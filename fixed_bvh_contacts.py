"""Fixed, body-local BVHs for affine-body discrete collision detection.

The BVHs are built once from ``soup.xcs``.  At query time a primitive from
body A is mapped into body B's local frame and queried against B's immutable
BVH.  Exact acceptance remains in world space, so this class only replaces
the broad phase.  CCD continues to use the swept world-space BVHs from
``ContactSolverBase``.
"""

import numpy as np
import warp as wp

from scalar_types import scalar, vec3
from contact import (
    ContactSolverBase,
    Contacts,
    Soup,
    _thickness,
    append,
    closest_point_triangle,
    verbose,
)
from affine_body_dynamics import AffineBodyDynamics, SCREW_ASSET_DIR


@wp.func
def _body_affine(q: wp.array(dtype=vec3), body: int):
    o = body * 4
    return wp.matrix_from_cols(q[o + 1], q[o + 2], q[o + 3])


@wp.func
def _inverse_padding(inv_a: wp.mat33d, world_padding: scalar):
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


@wp.kernel
def fixed_bvh_edge_edge_query(
    target_bvh: wp.uint64,
    source_edge_ids: wp.array(dtype=int),
    target_edge_ids: wp.array(dtype=int),
    target_body: int,
    q: wp.array(dtype=vec3),
    soup: Soup,
    contacts: Contacts,
    thickness: float,
):
    k = wp.tid()
    source_edge = source_edge_ids[k]
    a0 = soup.edges[2 * source_edge]
    a1 = soup.edges[2 * source_edge + 1]

    target_a = _body_affine(q, target_body)
    target_inv = wp.inverse(target_a)
    target_p = q[target_body * 4]
    p0 = target_inv @ (soup.x_transformed[a0] - target_p)
    p1 = target_inv @ (soup.x_transformed[a1] - target_p)
    padding = _inverse_padding(target_inv, scalar(2.0) * scalar(thickness))
    query = wp.bvh_query_aabb(
        target_bvh,
        wp.vec3(wp.min(p0, p1) - vec3(padding)),
        wp.vec3(wp.max(p0, p1) + vec3(padding)),
    )

    local_id = int(0)
    while wp.bvh_query_next(query, local_id):
        target_edge = target_edge_ids[local_id]
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
def fixed_bvh_point_triangle_query(
    target_bvh: wp.uint64,
    source_vertex_ids: wp.array(dtype=int),
    target_triangle_ids: wp.array(dtype=int),
    target_body: int,
    q: wp.array(dtype=vec3),
    soup: Soup,
    contacts: Contacts,
    thickness: float,
):
    k = wp.tid()
    point = source_vertex_ids[k]
    target_a = _body_affine(q, target_body)
    target_inv = wp.inverse(target_a)
    target_p = q[target_body * 4]
    local_point = target_inv @ (soup.x_transformed[point] - target_p)
    padding = _inverse_padding(target_inv, scalar(2.0) * scalar(thickness))
    query = wp.bvh_query_aabb(
        target_bvh,
        wp.vec3(local_point - vec3(padding)),
        wp.vec3(local_point + vec3(padding)),
    )

    local_id = int(0)
    while wp.bvh_query_next(query, local_id):
        triangle = target_triangle_ids[local_id]
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
            # Preserve the ordering used by point_triangle_collision.
            append(contacts, point, t1, t0, t2, thickness, point, triangle)


class FixedBodyContactSolver(ContactSolverBase):
    """Contact solver whose DCD broad phase uses immutable per-body BVHs.

    ``soup.xcs`` must contain body-local coordinates and ``affine_q`` must be
    an array of four vec3 blocks per body: ``[p, A[:,0], A[:,1], A[:,2]]``.
    The inherited swept BVHs are deliberately retained for CCD.
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

        self.fixed_local_lower = wp.array(local_lower, dtype=vec3)
        self.fixed_local_upper = wp.array(local_upper, dtype=vec3)
        self.fixed_world_lower = wp.zeros(self.n_bodies, dtype=vec3)
        self.fixed_world_upper = wp.zeros(self.n_bodies, dtype=vec3)

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

    def detect_collision(self):
        """Run fixed-local-BVH DCD; CCD remains inherited and swept."""
        self.compute_V(ret=False)
        with self.profile_timer("fixed DCD object cull"):
            pairs = self._overlapping_body_pairs()

        self.contacts_new.cnt.zero_()
        self.contacts_new.htable.fill_(-1)
        self.contacts_pt.cnt.zero_()
        self.contacts_pt.htable.fill_(-1)

        with self.profile_timer("fixed DCD EE queries"):
            for a, b in pairs:
                target_bvh = self.fixed_edge_bvhs[b][0]
                wp.launch(
                    fixed_bvh_edge_edge_query,
                    self.fixed_edge_ids[a].shape[0],
                    inputs=[target_bvh.id, self.fixed_edge_ids[a], self.fixed_edge_ids[b],
                            b, self.affine_q, self.soup, self.contacts_new, _thickness],
                )

        with self.profile_timer("fixed DCD PT queries"):
            for a, b in pairs:
                wp.launch(
                    fixed_bvh_point_triangle_query,
                    self.fixed_vertex_ids[a].shape[0],
                    inputs=[self.fixed_triangle_bvhs[b][0].id, self.fixed_vertex_ids[a],
                            self.fixed_triangle_ids[b], b, self.affine_q, self.soup,
                            self.contacts_pt, _thickness],
                )
                wp.launch(
                    fixed_bvh_point_triangle_query,
                    self.fixed_vertex_ids[b].shape[0],
                    inputs=[self.fixed_triangle_bvhs[a][0].id, self.fixed_vertex_ids[b],
                            self.fixed_triangle_ids[a], a, self.affine_q, self.soup,
                            self.contacts_pt, _thickness],
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
            print(f"fixed BVH contacts: EE={self.n_contacts}, PT={self.n_contacts_pt}, body pairs={len(pairs)}")


class FixedBVHAffineBodyDynamics(AffineBodyDynamics, FixedBodyContactSolver):
    """ABD using fixed local BVHs for DCD and the existing swept CCD."""

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
