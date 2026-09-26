"""Fixed, body-local BVHs for affine-body discrete collision detection.

The BVHs are built once from ``soup.xcs``.  At query time a primitive from
body A is mapped into body B's local frame and queried against B's immutable
BVH. Exact acceptance remains in world space. CCD uses conservative swept
queries against the same fixed BVHs and evaluates EE polynomial TOI on a
compact candidate array.
"""

import numpy as np
import warp as wp

from scalar_types import scalar, vec3, mat33
from contact import (
    ContactSolverBase,
    Contacts,
    Soup,
    _thickness,
    _add_dx,
    append,
    closest_point_triangle,
    verbose,
)
from ccd.ccd import ee_collision_time, pt_collision_time
from affine_body_dynamics import (
    AffineBodyDynamics,
    SCREW_ASSET_DIR,
    wrecking_balls_scene_args,
)


@wp.func
def _body_affine(q: wp.array(dtype=vec3), body: int):
    o = body * 4
    return wp.matrix_from_cols(q[o + 1], q[o + 2], q[o + 3])


@wp.func
def _inverse_padding(inv_a: mat33, world_padding: scalar):
    norm2 = scalar(0.0)
    for i in range(3):
        for j in range(3):
            norm2 += inv_a[i, j] * inv_a[i, j]
    return world_padding * wp.sqrt(norm2)


@wp.kernel
def affine_body_aabbs(
    q: wp.array(dtype=vec3),
    local_lower: wp.array(dtype=vec3),
    local_upper: wp.array(dtype=vec3),
    world_lower: wp.array(dtype=vec3),
    world_upper: wp.array(dtype=vec3),
):
    b = wp.tid()
    o = b * 4
    a = _body_affine(q, b)
    center = scalar(0.5) * (local_lower[b] + local_upper[b])
    extent = scalar(0.5) * (local_upper[b] - local_lower[b])
    world_center = q[o] + a @ center
    world_extent = vec3(
        wp.abs(a[0, 0]) * extent[0] + wp.abs(a[0, 1]) * extent[1] + wp.abs(a[0, 2]) * extent[2],
        wp.abs(a[1, 0]) * extent[0] + wp.abs(a[1, 1]) * extent[1] + wp.abs(a[1, 2]) * extent[2],
        wp.abs(a[2, 0]) * extent[0] + wp.abs(a[2, 1]) * extent[1] + wp.abs(a[2, 2]) * extent[2],
    )
    world_lower[b] = world_center - world_extent
    world_upper[b] = world_center + world_extent


@wp.func
def _aabbs_overlap(a_lower: vec3, a_upper: vec3,
                   b_lower: vec3, b_upper: vec3, padding: scalar):
    return (
        a_lower[0] - padding <= b_upper[0]
        and b_lower[0] - padding <= a_upper[0]
        and a_lower[1] - padding <= b_upper[1]
        and b_lower[1] - padding <= a_upper[1]
        and a_lower[2] - padding <= b_upper[2]
        and b_lower[2] - padding <= a_upper[2]
    )


@wp.kernel
def compact_overlapping_body_pairs(
    lower: wp.array(dtype=vec3), upper: wp.array(dtype=vec3),
    padding: scalar, pairs: wp.array(dtype=wp.vec2i), count: wp.array(dtype=int),
):
    a, b = wp.tid()
    if a < b and _aabbs_overlap(lower[a], upper[a], lower[b], upper[b], padding):
        output = wp.atomic_add(count, 0, 1)
        pairs[output] = wp.vec2i(a, b)


BODY_PAIR_TILE = 32


@wp.kernel
def compact_overlapping_body_pairs_tiled(
    lower: wp.array(dtype=vec3), upper: wp.array(dtype=vec3), n_bodies: int,
    padding: scalar, pairs: wp.array(dtype=wp.vec2i), count: wp.array(dtype=int),
):
    block_a, block_b, lane = wp.tid()
    a = block_a * BODY_PAIR_TILE + lane
    b_lower = wp.tile_load(
        lower, shape=BODY_PAIR_TILE, offset=block_b * BODY_PAIR_TILE, storage="shared"
    )
    b_upper = wp.tile_load(
        upper, shape=BODY_PAIR_TILE, offset=block_b * BODY_PAIR_TILE, storage="shared"
    )
    if a < n_bodies:
        a_lower = lower[a]
        a_upper = upper[a]
        for j in range(BODY_PAIR_TILE):
            b = block_b * BODY_PAIR_TILE + j
            if b < n_bodies and a < b:
                if _aabbs_overlap(a_lower, a_upper, b_lower[j], b_upper[j], padding):
                    output = wp.atomic_add(count, 0, 1)
                    pairs[output] = wp.vec2i(a, b)


@wp.func
def _target_motion_extent(
    q: wp.array(dtype=vec3), dq: wp.array(dtype=vec3), body: int,
    local_lower: vec3, local_upper: vec3,
):
    o = body * 4
    a0 = _body_affine(q, body)
    a1 = wp.matrix_from_cols(
        q[o + 1] - dq[o + 1], q[o + 2] - dq[o + 2], q[o + 3] - dq[o + 3]
    )
    inv_a0 = wp.inverse(a0)
    d = inv_a0 @ (a1 - a0)
    c = inv_a0 @ (-dq[o])
    center = scalar(0.5) * (local_lower + local_upper)
    extent = scalar(0.5) * (local_upper - local_lower)
    dc = d @ center + c
    return vec3(
        wp.abs(dc[0]) + wp.abs(d[0, 0]) * extent[0] + wp.abs(d[0, 1]) * extent[1] + wp.abs(d[0, 2]) * extent[2],
        wp.abs(dc[1]) + wp.abs(d[1, 0]) * extent[0] + wp.abs(d[1, 1]) * extent[1] + wp.abs(d[1, 2]) * extent[2],
        wp.abs(dc[2]) + wp.abs(d[2, 0]) * extent[0] + wp.abs(d[2, 1]) * extent[1] + wp.abs(d[2, 2]) * extent[2],
    )


@wp.kernel
def affine_swept_body_aabbs(
    q: wp.array(dtype=vec3), dq: wp.array(dtype=vec3),
    local_lower: wp.array(dtype=vec3), local_upper: wp.array(dtype=vec3),
    world_lower: wp.array(dtype=vec3), world_upper: wp.array(dtype=vec3),
    min_abs_det: wp.array(dtype=scalar),
):
    b = wp.tid()
    o = b * 4
    center = scalar(0.5) * (local_lower[b] + local_upper[b])
    extent = scalar(0.5) * (local_upper[b] - local_lower[b])
    a0 = _body_affine(q, b)
    a1 = wp.matrix_from_cols(
        q[o + 1] - dq[o + 1], q[o + 2] - dq[o + 2], q[o + 3] - dq[o + 3]
    )
    c0 = q[o] + a0 @ center
    c1 = q[o] - dq[o] + a1 @ center
    e0 = vec3(
        wp.abs(a0[0, 0]) * extent[0] + wp.abs(a0[0, 1]) * extent[1] + wp.abs(a0[0, 2]) * extent[2],
        wp.abs(a0[1, 0]) * extent[0] + wp.abs(a0[1, 1]) * extent[1] + wp.abs(a0[1, 2]) * extent[2],
        wp.abs(a0[2, 0]) * extent[0] + wp.abs(a0[2, 1]) * extent[1] + wp.abs(a0[2, 2]) * extent[2],
    )
    e1 = vec3(
        wp.abs(a1[0, 0]) * extent[0] + wp.abs(a1[0, 1]) * extent[1] + wp.abs(a1[0, 2]) * extent[2],
        wp.abs(a1[1, 0]) * extent[0] + wp.abs(a1[1, 1]) * extent[1] + wp.abs(a1[1, 2]) * extent[2],
        wp.abs(a1[2, 0]) * extent[0] + wp.abs(a1[2, 1]) * extent[1] + wp.abs(a1[2, 2]) * extent[2],
    )
    world_lower[b] = wp.min(c0 - e0, c1 - e1)
    world_upper[b] = wp.max(c0 + e0, c1 + e1)
    min_abs_det[b] = wp.min(wp.abs(wp.determinant(a0)), wp.abs(wp.determinant(a1)))


@wp.func
def _swept_edge_query_bounds(
    source_edge: int, target_body: int,
    q: wp.array(dtype=vec3), dq: wp.array(dtype=vec3),
    x0: wp.array(dtype=vec3), x1: wp.array(dtype=vec3),
    edges: wp.array(dtype=int),
    local_lower: wp.array(dtype=vec3), local_upper: wp.array(dtype=vec3),
    padding: scalar,
):
    target_p0 = q[target_body * 4]
    inv_a0 = wp.inverse(_body_affine(q, target_body))
    a0 = edges[2 * source_edge]
    a1 = edges[2 * source_edge + 1]
    p00 = inv_a0 @ (x0[a0] - target_p0)
    p01 = inv_a0 @ (x1[a0] - target_p0)
    p10 = inv_a0 @ (x0[a1] - target_p0)
    p11 = inv_a0 @ (x1[a1] - target_p0)
    motion = _target_motion_extent(
        q, dq, target_body, local_lower[target_body], local_upper[target_body]
    )
    inflate = motion + vec3(_inverse_padding(inv_a0, padding))
    lower = wp.min(wp.min(p00, p01), wp.min(p10, p11)) - inflate
    upper = wp.max(wp.max(p00, p01), wp.max(p10, p11)) + inflate
    return lower, upper


@wp.kernel
def batched_fixed_bvh_edge_edge_toi(
    pairs: wp.array(dtype=wp.vec2i), fixed: wp.array(dtype=int),
    body_edge_offsets: wp.array(dtype=int), edge_ids: wp.array(dtype=int),
    edge_bvh_ids: wp.array(dtype=wp.uint64),
    q: wp.array(dtype=vec3), dq: wp.array(dtype=vec3),
    x0: wp.array(dtype=vec3), x1: wp.array(dtype=vec3),
    edges: wp.array(dtype=int), local_lower: wp.array(dtype=vec3),
    local_upper: wp.array(dtype=vec3), padding: scalar,
    toi: wp.array(dtype=scalar),
):
    pair_index, local_source = wp.tid()
    pair = pairs[pair_index]
    source_body = pair[0]
    target_body = pair[1]
    if fixed[source_body] != 0 and fixed[target_body] == 0:
        source_body = pair[1]
        target_body = pair[0]
    source_begin = body_edge_offsets[source_body]
    source_count = body_edge_offsets[source_body + 1] - source_begin
    if local_source >= source_count:
        return

    source_edge = edge_ids[source_begin + local_source]
    lower, upper = _swept_edge_query_bounds(
        source_edge, target_body, q, dq, x0, x1, edges,
        local_lower, local_upper, padding,
    )
    query = wp.bvh_query_aabb(
        edge_bvh_ids[target_body], wp.vec3(lower), wp.vec3(upper)
    )
    target_begin = body_edge_offsets[target_body]
    local_id = int(0)
    while wp.bvh_query_next(query, local_id):
        target_edge = edge_ids[target_begin + local_id]
        a0 = edges[2 * source_edge]
        a1 = edges[2 * source_edge + 1]
        b0 = edges[2 * target_edge]
        b1 = edges[2 * target_edge + 1]
        t = ee_collision_time(
            x0[a0], x0[a1], x0[b0], x0[b1],
            x1[a0], x1[a1], x1[b0], x1[b1],
        )
        if t < scalar(1.0):
            wp.atomic_min(toi, 0, t)


@wp.kernel
def batched_fixed_bvh_point_triangle_toi(
    pairs: wp.array(dtype=wp.vec2i),
    body_vertex_offsets: wp.array(dtype=int), vertex_ids: wp.array(dtype=int),
    body_triangle_offsets: wp.array(dtype=int), triangle_ids: wp.array(dtype=int),
    triangle_bvh_ids: wp.array(dtype=wp.uint64),
    q: wp.array(dtype=vec3), dq: wp.array(dtype=vec3),
    x0: wp.array(dtype=vec3), x1: wp.array(dtype=vec3),
    triangles: wp.array(dtype=int), local_lower: wp.array(dtype=vec3),
    local_upper: wp.array(dtype=vec3), padding: scalar,
    toi: wp.array(dtype=scalar),
):
    pair_index, local_source, direction = wp.tid()
    pair = pairs[pair_index]
    source_body = pair[direction]
    target_body = pair[1 - direction]
    source_begin = body_vertex_offsets[source_body]
    source_count = body_vertex_offsets[source_body + 1] - source_begin
    if local_source >= source_count:
        return

    point = vertex_ids[source_begin + local_source]
    target_p0 = q[target_body * 4]
    inv_a0 = wp.inverse(_body_affine(q, target_body))
    p0_local = inv_a0 @ (x0[point] - target_p0)
    p1_local = inv_a0 @ (x1[point] - target_p0)
    motion = _target_motion_extent(
        q, dq, target_body, local_lower[target_body], local_upper[target_body]
    )
    inflate = motion + vec3(_inverse_padding(inv_a0, padding))
    query = wp.bvh_query_aabb(
        triangle_bvh_ids[target_body],
        wp.vec3(wp.min(p0_local, p1_local) - inflate),
        wp.vec3(wp.max(p0_local, p1_local) + inflate),
    )
    target_begin = body_triangle_offsets[target_body]
    local_id = int(0)
    while wp.bvh_query_next(query, local_id):
        triangle = triangle_ids[target_begin + local_id]
        t0 = triangles[3 * triangle]
        t1 = triangles[3 * triangle + 1]
        t2 = triangles[3 * triangle + 2]
        t = pt_collision_time(
            x0[point], x0[t0], x0[t1], x0[t2],
            x1[point], x1[t0], x1[t1], x1[t2],
        )
        if t < scalar(1.0):
            wp.atomic_min(toi, 0, t)


@wp.kernel
def batched_fixed_bvh_edge_edge_query(
    pairs: wp.array(dtype=wp.vec2i),
    body_edge_offsets: wp.array(dtype=int), edge_ids: wp.array(dtype=int),
    edge_bvh_ids: wp.array(dtype=wp.uint64), q: wp.array(dtype=vec3),
    soup: Soup, contacts: Contacts, thickness: float,
):
    pair_index, local_source = wp.tid()
    pair = pairs[pair_index]
    source_body = pair[0]
    target_body = pair[1]
    source_begin = body_edge_offsets[source_body]
    source_count = body_edge_offsets[source_body + 1] - source_begin
    if local_source >= source_count:
        return

    source_edge = edge_ids[source_begin + local_source]
    a0 = soup.edges[2 * source_edge]
    a1 = soup.edges[2 * source_edge + 1]
    target_a = _body_affine(q, target_body)
    target_inv = wp.inverse(target_a)
    target_p = q[target_body * 4]
    p0 = target_inv @ (soup.x_transformed[a0] - target_p)
    p1 = target_inv @ (soup.x_transformed[a1] - target_p)
    padding = _inverse_padding(target_inv, scalar(2.0) * scalar(thickness))
    query = wp.bvh_query_aabb(
        edge_bvh_ids[target_body],
        wp.vec3(wp.min(p0, p1) - vec3(padding)),
        wp.vec3(wp.max(p0, p1) + vec3(padding)),
    )
    target_begin = body_edge_offsets[target_body]
    local_id = int(0)
    while wp.bvh_query_next(query, local_id):
        target_edge = edge_ids[target_begin + local_id]
        b0 = soup.edges[2 * target_edge]
        b1 = soup.edges[2 * target_edge + 1]
        result = wp.closest_point_edge_edge(
            wp.vec3(soup.x_transformed[a0]), wp.vec3(soup.x_transformed[a1]),
            wp.vec3(soup.x_transformed[b0]), wp.vec3(soup.x_transformed[b1]),
            1.0e-6,
        )
        if scalar(result[2]) < scalar(2.0) * scalar(thickness):
            append(contacts, a0, a1, b0, b1, thickness, source_edge, target_edge)


@wp.kernel
def batched_fixed_bvh_point_triangle_query(
    pairs: wp.array(dtype=wp.vec2i),
    body_vertex_offsets: wp.array(dtype=int), vertex_ids: wp.array(dtype=int),
    body_triangle_offsets: wp.array(dtype=int), triangle_ids: wp.array(dtype=int),
    triangle_bvh_ids: wp.array(dtype=wp.uint64), q: wp.array(dtype=vec3),
    soup: Soup, contacts: Contacts, thickness: float,
):
    pair_index, local_source, direction = wp.tid()
    pair = pairs[pair_index]
    source_body = pair[direction]
    target_body = pair[1 - direction]
    source_begin = body_vertex_offsets[source_body]
    source_count = body_vertex_offsets[source_body + 1] - source_begin
    if local_source >= source_count:
        return

    point = vertex_ids[source_begin + local_source]
    target_a = _body_affine(q, target_body)
    target_inv = wp.inverse(target_a)
    target_p = q[target_body * 4]
    local_point = target_inv @ (soup.x_transformed[point] - target_p)
    padding = _inverse_padding(target_inv, scalar(2.0) * scalar(thickness))
    query = wp.bvh_query_aabb(
        triangle_bvh_ids[target_body],
        wp.vec3(local_point - vec3(padding)),
        wp.vec3(local_point + vec3(padding)),
    )
    target_begin = body_triangle_offsets[target_body]
    local_id = int(0)
    while wp.bvh_query_next(query, local_id):
        triangle = triangle_ids[target_begin + local_id]
        t0 = soup.triangles[3 * triangle]
        t1 = soup.triangles[3 * triangle + 1]
        t2 = soup.triangles[3 * triangle + 2]
        result, unused_type = closest_point_triangle(
            wp.vec3(soup.x_transformed[point]),
            wp.vec3(soup.x_transformed[t0]),
            wp.vec3(soup.x_transformed[t1]),
            wp.vec3(soup.x_transformed[t2]),
        )
        if scalar(result[2]) < scalar(2.0) * scalar(thickness):
            append(contacts, point, t1, t0, t2, thickness, point, triangle)


@wp.kernel
def compact_edge_query_tasks(
    pairs: wp.array(dtype=wp.vec2i), body_edge_offsets: wp.array(dtype=int),
    edge_ids: wp.array(dtype=int), tasks: wp.array(dtype=wp.vec2i),
    task_count: wp.array(dtype=int),
):
    pair_index, local_source = wp.tid()
    pair = pairs[pair_index]
    source_body = pair[0]
    source_begin = body_edge_offsets[source_body]
    source_count = body_edge_offsets[source_body + 1] - source_begin
    if local_source < source_count:
        output = wp.atomic_add(task_count, 0, 1)
        tasks[output] = wp.vec2i(edge_ids[source_begin + local_source], pair[1])


@wp.kernel
def compact_point_query_tasks(
    pairs: wp.array(dtype=wp.vec2i), body_vertex_offsets: wp.array(dtype=int),
    vertex_ids: wp.array(dtype=int), tasks: wp.array(dtype=wp.vec2i),
    task_count: wp.array(dtype=int),
):
    pair_index, local_source, direction = wp.tid()
    pair = pairs[pair_index]
    source_body = pair[direction]
    target_body = pair[1 - direction]
    source_begin = body_vertex_offsets[source_body]
    source_count = body_vertex_offsets[source_body + 1] - source_begin
    if local_source < source_count:
        output = wp.atomic_add(task_count, 0, 1)
        tasks[output] = wp.vec2i(vertex_ids[source_begin + local_source], target_body)


@wp.kernel
def tiled_fixed_bvh_edge_edge_query(
    tasks: wp.array(dtype=wp.vec2i), body_edge_offsets: wp.array(dtype=int),
    edge_ids: wp.array(dtype=int), edge_bvh_ids: wp.array(dtype=wp.uint64),
    q: wp.array(dtype=vec3), soup: Soup, contacts: Contacts, thickness: float,
):
    task_index, lane = wp.tid()
    task = tasks[task_index]
    source_edge = task[0]
    target_body = task[1]
    a0 = soup.edges[2 * source_edge]
    a1 = soup.edges[2 * source_edge + 1]
    target_a = _body_affine(q, target_body)
    target_inv = wp.inverse(target_a)
    target_p = q[target_body * 4]
    p0 = target_inv @ (soup.x_transformed[a0] - target_p)
    p1 = target_inv @ (soup.x_transformed[a1] - target_p)
    padding = _inverse_padding(target_inv, scalar(2.0) * scalar(thickness))
    query = wp.tile_bvh_query_aabb(
        edge_bvh_ids[target_body],
        wp.vec3(wp.min(p0, p1) - vec3(padding)),
        wp.vec3(wp.max(p0, p1) + vec3(padding)),
    )
    result_tile = wp.tile_bvh_query_next(query)
    target_begin = body_edge_offsets[target_body]
    while wp.tile_max(result_tile)[0] >= 0:
        local_id = wp.untile(result_tile)
        if local_id >= 0:
            target_edge = edge_ids[target_begin + local_id]
            b0 = soup.edges[2 * target_edge]
            b1 = soup.edges[2 * target_edge + 1]
            result = wp.closest_point_edge_edge(
                wp.vec3(soup.x_transformed[a0]), wp.vec3(soup.x_transformed[a1]),
                wp.vec3(soup.x_transformed[b0]), wp.vec3(soup.x_transformed[b1]),
                1.0e-6,
            )
            if scalar(result[2]) < scalar(2.0) * scalar(thickness):
                append(contacts, a0, a1, b0, b1, thickness, source_edge, target_edge)
        result_tile = wp.tile_bvh_query_next(query)


@wp.kernel
def tiled_fixed_bvh_point_triangle_query(
    tasks: wp.array(dtype=wp.vec2i), body_triangle_offsets: wp.array(dtype=int),
    triangle_ids: wp.array(dtype=int), triangle_bvh_ids: wp.array(dtype=wp.uint64),
    q: wp.array(dtype=vec3), soup: Soup, contacts: Contacts, thickness: float,
):
    task_index, lane = wp.tid()
    task = tasks[task_index]
    point = task[0]
    target_body = task[1]
    target_a = _body_affine(q, target_body)
    target_inv = wp.inverse(target_a)
    target_p = q[target_body * 4]
    local_point = target_inv @ (soup.x_transformed[point] - target_p)
    padding = _inverse_padding(target_inv, scalar(2.0) * scalar(thickness))
    query = wp.tile_bvh_query_aabb(
        triangle_bvh_ids[target_body],
        wp.vec3(local_point - vec3(padding)),
        wp.vec3(local_point + vec3(padding)),
    )
    result_tile = wp.tile_bvh_query_next(query)
    target_begin = body_triangle_offsets[target_body]
    while wp.tile_max(result_tile)[0] >= 0:
        local_id = wp.untile(result_tile)
        if local_id >= 0:
            triangle = triangle_ids[target_begin + local_id]
            t0 = soup.triangles[3 * triangle]
            t1 = soup.triangles[3 * triangle + 1]
            t2 = soup.triangles[3 * triangle + 2]
            result, unused_type = closest_point_triangle(
                wp.vec3(soup.x_transformed[point]),
                wp.vec3(soup.x_transformed[t0]),
                wp.vec3(soup.x_transformed[t1]),
                wp.vec3(soup.x_transformed[t2]),
            )
            if scalar(result[2]) < scalar(2.0) * scalar(thickness):
                append(contacts, point, t1, t0, t2, thickness, point, triangle)
        result_tile = wp.tile_bvh_query_next(query)


class FixedBodyContactSolver(ContactSolverBase):
    """Contact solver whose DCD broad phase uses immutable per-body BVHs.

    ``soup.xcs`` must contain body-local coordinates and ``affine_q`` must be
    an array of four vec3 blocks per body: ``[p, A[:,0], A[:,1], A[:,2]]``.
    The inherited swept BVHs are retained only as a near-singular fallback.
    """

    def initialize_fixed_body_bvhs(self, affine_q):
        self.affine_q = affine_q
        rest = self.soup.xcs.numpy()
        body = self.soup.body.numpy().astype(np.int32)
        edges = self.soup.edges.numpy().reshape(-1, 2)
        triangles = self.soup.triangles.numpy().reshape(-1, 3)

        local_lower = np.zeros((self.n_bodies, 3), dtype=np.float64)
        local_upper = np.zeros((self.n_bodies, 3), dtype=np.float64)
        self.fixed_vertex_ids = []
        self.fixed_edge_ids = []
        self.fixed_triangle_ids = []
        self.fixed_edge_bvhs = []
        self.fixed_triangle_bvhs = []

        for b in range(self.n_bodies):
            vertex_ids = np.flatnonzero(body == b).astype(np.int32)
            edge_ids = np.flatnonzero(body[edges[:, 0]] == b).astype(np.int32)
            triangle_ids = np.flatnonzero(body[triangles[:, 0]] == b).astype(np.int32)
            local_lower[b] = rest[vertex_ids].min(axis=0)
            local_upper[b] = rest[vertex_ids].max(axis=0)

            self.fixed_vertex_ids.append(wp.array(vertex_ids, dtype=int))
            self.fixed_edge_ids.append(wp.array(edge_ids, dtype=int))
            self.fixed_triangle_ids.append(wp.array(triangle_ids, dtype=int))

            edge_lower = np.minimum(rest[edges[edge_ids, 0]], rest[edges[edge_ids, 1]]).astype(np.float32)
            edge_upper = np.maximum(rest[edges[edge_ids, 0]], rest[edges[edge_ids, 1]]).astype(np.float32)
            edge_lower_wp = wp.array(edge_lower, dtype=wp.vec3)
            edge_upper_wp = wp.array(edge_upper, dtype=wp.vec3)
            # Retain bounds arrays because the BVH references their storage.
            self.fixed_edge_bvhs.append((wp.Bvh(edge_lower_wp, edge_upper_wp), edge_lower_wp, edge_upper_wp))

            tri = triangles[triangle_ids]
            tri_lower = np.min(rest[tri], axis=1).astype(np.float32)
            tri_upper = np.max(rest[tri], axis=1).astype(np.float32)
            tri_lower_wp = wp.array(tri_lower, dtype=wp.vec3)
            tri_upper_wp = wp.array(tri_upper, dtype=wp.vec3)
            self.fixed_triangle_bvhs.append((wp.Bvh(tri_lower_wp, tri_upper_wp), tri_lower_wp, tri_upper_wp))

        def flatten_by_body(arrays):
            counts = np.asarray([array.shape[0] for array in arrays], dtype=np.int32)
            offsets = np.zeros(self.n_bodies + 1, dtype=np.int32)
            offsets[1:] = np.cumsum(counts)
            if offsets[-1]:
                values = np.concatenate([array.numpy() for array in arrays]).astype(np.int32)
            else:
                values = np.zeros(0, dtype=np.int32)
            return wp.array(offsets, dtype=int), wp.array(values, dtype=int), int(counts.max(initial=0))

        self.fixed_vertex_offsets, self.fixed_vertex_ids_flat, self.fixed_max_vertices = flatten_by_body(
            self.fixed_vertex_ids
        )
        self.fixed_edge_offsets, self.fixed_edge_ids_flat, self.fixed_max_edges = flatten_by_body(
            self.fixed_edge_ids
        )
        self.fixed_triangle_offsets, self.fixed_triangle_ids_flat, self.fixed_max_triangles = flatten_by_body(
            self.fixed_triangle_ids
        )
        self.fixed_edge_bvh_ids = wp.array(
            np.asarray([entry[0].id for entry in self.fixed_edge_bvhs], dtype=np.uint64),
            dtype=wp.uint64,
        )
        self.fixed_triangle_bvh_ids = wp.array(
            np.asarray([entry[0].id for entry in self.fixed_triangle_bvhs], dtype=np.uint64),
            dtype=wp.uint64,
        )
        max_body_pairs = self.n_bodies * (self.n_bodies - 1) // 2
        self.fixed_body_pairs = wp.empty(max(max_body_pairs, 1), dtype=wp.vec2i)
        self.fixed_body_pair_count = wp.zeros(1, dtype=int)
        self.fixed_pair_cull_mode = "kernel"
        self.fixed_query_mode = "scalar"
        self.tile_query_threads = 32
        self.fixed_edge_query_tasks = wp.empty(1, dtype=wp.vec2i)
        self.fixed_point_query_tasks = wp.empty(1, dtype=wp.vec2i)
        self.fixed_edge_query_task_count = wp.zeros(1, dtype=int)
        self.fixed_point_query_task_count = wp.zeros(1, dtype=int)

        self.fixed_local_lower = wp.array(local_lower, dtype=vec3)
        self.fixed_local_upper = wp.array(local_upper, dtype=vec3)
        self.fixed_world_lower = wp.zeros(self.n_bodies, dtype=vec3)
        self.fixed_world_upper = wp.zeros(self.n_bodies, dtype=vec3)
        self.fixed_swept_min_det = wp.zeros(self.n_bodies, dtype=scalar)
        if hasattr(self, "abd_fixed"):
            self.fixed_body_flags_wp = self.abd_fixed
        else:
            self.fixed_body_flags_wp = wp.zeros(self.n_bodies, dtype=int)

    def _overlapping_body_pairs(self):
        wp.launch(
            affine_body_aabbs,
            self.n_bodies,
            inputs=[self.affine_q, self.fixed_local_lower, self.fixed_local_upper,
                    self.fixed_world_lower, self.fixed_world_upper],
        )
        lower = self.fixed_world_lower.numpy()
        upper = self.fixed_world_upper.numpy()
        padding = 2.0 * float(_thickness)
        pairs = []
        for a in range(self.n_bodies):
            for b in range(a + 1, self.n_bodies):
                if np.all(lower[a] - padding <= upper[b]) and np.all(lower[b] - padding <= upper[a]):
                    pairs.append((a, b))
        return pairs

    def _overlapping_body_pairs_gpu(self, tiled=True):
        wp.launch(
            affine_body_aabbs,
            self.n_bodies,
            inputs=[self.affine_q, self.fixed_local_lower, self.fixed_local_upper,
                    self.fixed_world_lower, self.fixed_world_upper],
        )
        self.fixed_body_pair_count.zero_()
        padding = scalar(2.0) * scalar(_thickness)
        if tiled:
            blocks = (self.n_bodies + BODY_PAIR_TILE - 1) // BODY_PAIR_TILE
            wp.launch_tiled(
                compact_overlapping_body_pairs_tiled,
                dim=[blocks, blocks],
                inputs=[self.fixed_world_lower, self.fixed_world_upper, self.n_bodies,
                        padding, self.fixed_body_pairs, self.fixed_body_pair_count],
                block_dim=BODY_PAIR_TILE,
            )
        else:
            wp.launch(
                compact_overlapping_body_pairs,
                dim=(self.n_bodies, self.n_bodies),
                inputs=[self.fixed_world_lower, self.fixed_world_upper, padding,
                        self.fixed_body_pairs, self.fixed_body_pair_count],
            )
        return int(self.fixed_body_pair_count.numpy()[0])

    @staticmethod
    def _ensure_query_task_capacity(array, required):
        if required <= array.shape[0]:
            return array
        return wp.empty(1 << (required - 1).bit_length(), dtype=wp.vec2i)

    def _build_tiled_query_tasks(self, n_pairs):
        edge_upper = n_pairs * self.fixed_max_edges
        point_upper = n_pairs * self.fixed_max_vertices * 2
        self.fixed_edge_query_tasks = self._ensure_query_task_capacity(
            self.fixed_edge_query_tasks, edge_upper
        )
        self.fixed_point_query_tasks = self._ensure_query_task_capacity(
            self.fixed_point_query_tasks, point_upper
        )
        self.fixed_edge_query_task_count.zero_()
        self.fixed_point_query_task_count.zero_()
        wp.launch(
            compact_edge_query_tasks,
            dim=(n_pairs, self.fixed_max_edges),
            inputs=[self.fixed_body_pairs, self.fixed_edge_offsets,
                    self.fixed_edge_ids_flat, self.fixed_edge_query_tasks,
                    self.fixed_edge_query_task_count],
        )
        wp.launch(
            compact_point_query_tasks,
            dim=(n_pairs, self.fixed_max_vertices, 2),
            inputs=[self.fixed_body_pairs, self.fixed_vertex_offsets,
                    self.fixed_vertex_ids_flat, self.fixed_point_query_tasks,
                    self.fixed_point_query_task_count],
        )
        return (
            int(self.fixed_edge_query_task_count.numpy()[0]),
            int(self.fixed_point_query_task_count.numpy()[0]),
        )

    def _swept_overlapping_body_pairs(self, affine_dx):
        wp.launch(
            affine_swept_body_aabbs,
            self.n_bodies,
            inputs=[self.affine_q, affine_dx, self.fixed_local_lower,
                    self.fixed_local_upper, self.fixed_world_lower,
                    self.fixed_world_upper, self.fixed_swept_min_det],
        )
        min_det = self.fixed_swept_min_det.numpy()
        if np.any(min_det < 1.0e-8):
            return None
        self.fixed_body_pair_count.zero_()
        if self.fixed_pair_cull_mode == "tile":
            blocks = (self.n_bodies + BODY_PAIR_TILE - 1) // BODY_PAIR_TILE
            wp.launch_tiled(
                compact_overlapping_body_pairs_tiled,
                dim=[blocks, blocks],
                inputs=[self.fixed_world_lower, self.fixed_world_upper, self.n_bodies,
                        scalar(1.0e-7), self.fixed_body_pairs, self.fixed_body_pair_count],
                block_dim=BODY_PAIR_TILE,
            )
        else:
            wp.launch(
                compact_overlapping_body_pairs,
                dim=(self.n_bodies, self.n_bodies),
                inputs=[self.fixed_world_lower, self.fixed_world_upper, scalar(1.0e-7),
                        self.fixed_body_pairs, self.fixed_body_pair_count],
            )
        n_pairs = int(self.fixed_body_pair_count.numpy()[0])
        return n_pairs

    def collision_free_step(self, dx):
        """Fixed-local-BVH CCD with batched EE and PT evaluation."""
        if not hasattr(self, "abd_states"):
            return ContactSolverBase.collision_free_step(self, dx)

        padding = scalar(1.0e-7)
        with self.profile_timer("fixed CCD vertex prediction"):
            wp.launch(
                _add_dx,
                self.soup.x_transformed.shape[0],
                inputs=[self.ccd_x1, self.soup.x_transformed, dx],
            )
        with self.profile_timer("fixed CCD object cull"):
            body_pairs = self._swept_overlapping_body_pairs(self.abd_states.dx)
        if body_pairs is None:
            with self.profile_timer("fixed CCD singular fallback"):
                return ContactSolverBase.collision_free_step(self, dx)

        self.ccd_toi.fill_(1.0)
        if body_pairs:
            with self.profile_timer("fixed CCD EE query and TOI"):
                wp.launch(
                    batched_fixed_bvh_edge_edge_toi,
                    dim=(body_pairs, self.fixed_max_edges),
                    inputs=[self.fixed_body_pairs, self.fixed_body_flags_wp,
                            self.fixed_edge_offsets, self.fixed_edge_ids_flat,
                            self.fixed_edge_bvh_ids, self.affine_q, self.abd_states.dx,
                            self.soup.x_transformed, self.ccd_x1, self.soup.edges,
                            self.fixed_local_lower, self.fixed_local_upper,
                            padding, self.ccd_toi],
                )
            with self.profile_timer("fixed CCD PT query and TOI"):
                wp.launch(
                    batched_fixed_bvh_point_triangle_toi,
                    dim=(body_pairs, self.fixed_max_vertices, 2),
                    inputs=[self.fixed_body_pairs, self.fixed_vertex_offsets,
                            self.fixed_vertex_ids_flat, self.fixed_triangle_offsets,
                            self.fixed_triangle_ids_flat, self.fixed_triangle_bvh_ids,
                            self.affine_q, self.abd_states.dx,
                            self.soup.x_transformed, self.ccd_x1, self.soup.triangles,
                            self.fixed_local_lower, self.fixed_local_upper,
                            padding, self.ccd_toi],
                )

        with self.profile_timer("ccd toi host transfer"):
            toi = float(self.ccd_toi.numpy()[0])
        return 0.9 * toi if toi < 1.0 else 1.0

    def detect_collision(self):
        """Run fixed-local-BVH DCD; CCD remains inherited and swept."""
        self.compute_V(ret=False)
        with self.profile_timer("fixed DCD object cull"):
            n_pairs = self._overlapping_body_pairs_gpu(
                tiled=self.fixed_pair_cull_mode == "tile"
            )

        self.contacts_new.cnt.zero_()
        self.contacts_new.htable.fill_(-1)
        self.contacts_pt.cnt.zero_()
        self.contacts_pt.htable.fill_(-1)

        n_edge_tasks = 0
        n_point_tasks = 0
        if n_pairs and self.fixed_query_mode == "tile":
            with self.profile_timer("fixed DCD tile task compaction"):
                n_edge_tasks, n_point_tasks = self._build_tiled_query_tasks(n_pairs)

        with self.profile_timer("fixed DCD EE queries"):
            if n_edge_tasks:
                wp.launch_tiled(
                    tiled_fixed_bvh_edge_edge_query,
                    dim=[n_edge_tasks],
                    inputs=[self.fixed_edge_query_tasks, self.fixed_edge_offsets,
                            self.fixed_edge_ids_flat, self.fixed_edge_bvh_ids,
                            self.affine_q, self.soup, self.contacts_new, _thickness],
                    block_dim=self.tile_query_threads,
                )
            elif n_pairs and self.fixed_query_mode == "scalar":
                wp.launch(
                    batched_fixed_bvh_edge_edge_query,
                    dim=(n_pairs, self.fixed_max_edges),
                    inputs=[self.fixed_body_pairs, self.fixed_edge_offsets,
                            self.fixed_edge_ids_flat, self.fixed_edge_bvh_ids,
                            self.affine_q, self.soup, self.contacts_new, _thickness],
                )

        with self.profile_timer("fixed DCD PT queries"):
            if n_point_tasks:
                wp.launch_tiled(
                    tiled_fixed_bvh_point_triangle_query,
                    dim=[n_point_tasks],
                    inputs=[self.fixed_point_query_tasks, self.fixed_triangle_offsets,
                            self.fixed_triangle_ids_flat, self.fixed_triangle_bvh_ids,
                            self.affine_q, self.soup, self.contacts_pt, _thickness],
                    block_dim=self.tile_query_threads,
                )
            elif n_pairs and self.fixed_query_mode == "scalar":
                wp.launch(
                    batched_fixed_bvh_point_triangle_query,
                    dim=(n_pairs, self.fixed_max_vertices, 2),
                    inputs=[self.fixed_body_pairs, self.fixed_vertex_offsets,
                            self.fixed_vertex_ids_flat, self.fixed_triangle_offsets,
                            self.fixed_triangle_ids_flat, self.fixed_triangle_bvh_ids,
                            self.affine_q, self.soup, self.contacts_pt, _thickness],
                )

        with self.profile_timer("contact count host transfer"):
            self.n_contacts = int(self.contacts_new.cnt.numpy()[0])
            self.n_contacts_pt = int(self.contacts_pt.cnt.numpy()[0])

        if self.n_contacts > self.contacts_new.capacity:
            self._grow_contact_list(self.contacts_new, self.n_contacts)
            return self.detect_collision()
        if self.n_contacts_pt > self.contacts_pt.capacity:
            self._grow_contact_list(self.contacts_pt, self.n_contacts_pt, point_triangle=True)
            return self.detect_collision()
        if verbose:
            print(f"fixed BVH contacts: EE={self.n_contacts}, PT={self.n_contacts_pt}, body pairs={n_pairs}")


class FixedBVHAffineBodyDynamics(AffineBodyDynamics, FixedBodyContactSolver):
    """ABD using fixed local BVHs for both DCD and conservative CCD."""

    def __init__(self, *args, **kwargs):
        AffineBodyDynamics.__init__(self, *args, **kwargs)
        self.initialize_fixed_body_bvhs(self.abd_states.q)

    def detect_collision(self):
        return FixedBodyContactSolver.detect_collision(self)


class FixedBVHScrewAndNut(FixedBVHAffineBodyDynamics):
    def __init__(self, h=0.01, linear_solver=None):
        super().__init__(
            h,
            [SCREW_ASSET_DIR / "screw-big-2.obj", SCREW_ASSET_DIR / "nut-big-2.obj"],
            fixed_bodies=[1],
            gravity=(0, 0, 0),
            affine_stiffness=1.0e8,
            motors={0: (0.0, -np.pi, 0.0)},
            linear_solver=linear_solver,
        )


class FixedBVHWreckingBalls(FixedBVHAffineBodyDynamics):
    def __init__(self, h=0.01, body_limit=0, linear_solver="cg"):
        super().__init__(
            h,
            linear_solver=linear_solver,
            **wrecking_balls_scene_args(body_limit=body_limit),
        )
        self.max_newton_iterations = 1024
        self.velocity_tolerance = 0.4
