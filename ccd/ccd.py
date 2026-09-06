from scalar_types import *
from ccd.cubic_roots import cubic_roots
from ipctkwp.distance.ee import beta_gamma_ee
from ipctkwp.distance.pt import beta_gamma_pt


@wp.kernel
def swept_edge_aabbs(
    x: wp.array(dtype=vec3),
    dx: wp.array(dtype=vec3),
    edges: wp.array(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    padding: scalar,
):
    i = wp.tid()
    i0 = edges[2 * i]
    i1 = edges[2 * i + 1]
    x0 = x[i0]
    x1 = x[i1]
    x0_end = x0 + dx[i0]
    x1_end = x1 + dx[i1]
    lo = wp.min(wp.min(x0, x1), wp.min(x0_end, x1_end)) - vec3(padding)
    hi = wp.max(wp.max(x0, x1), wp.max(x0_end, x1_end)) + vec3(padding)
    lower[i] = wp.vec3(lo)
    upper[i] = wp.vec3(hi)


@wp.kernel
def swept_triangle_aabbs(
    x: wp.array(dtype=vec3),
    dx: wp.array(dtype=vec3),
    triangles: wp.array(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    padding: scalar,
):
    i = wp.tid()
    i0 = triangles[3 * i]
    i1 = triangles[3 * i + 1]
    i2 = triangles[3 * i + 2]
    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x0_end = x0 + dx[i0]
    x1_end = x1 + dx[i1]
    x2_end = x2 + dx[i2]
    lo = wp.min(wp.min(wp.min(x0, x1), x2), wp.min(wp.min(x0_end, x1_end), x2_end)) - vec3(padding)
    hi = wp.max(wp.max(wp.max(x0, x1), x2), wp.max(wp.max(x0_end, x1_end), x2_end)) + vec3(padding)
    lower[i] = wp.vec3(lo)
    upper[i] = wp.vec3(hi)
        
@wp.func
def verify_root_pt(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    beta, gamma = beta_gamma_pt(x0, x1, x2, x3)
    tol = scalar(1e-8)
    cond = beta >= -tol and gamma >= -tol and beta + gamma <= scalar(1.0) + tol
    return cond

@wp.func
def verify_root_ee(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    beta, gamma = beta_gamma_ee(x0, x1, x2, x3)
    tol = scalar(1e-8)
    cond = beta >= -tol and beta <= scalar(1.0) + tol and -gamma >= -tol and -gamma <= scalar(1.0) + tol
    return cond

@wp.func
def build_and_solve_4_points_coplanar(
    p0_t0: vec3, p1_t0: vec3, p2_t0: vec3, p3_t0: vec3,
    p0_t1: vec3, p1_t1: vec3, p2_t1: vec3, p3_t1: vec3
):
    a1 = wp.matrix_from_cols(p1_t1, p2_t1, p3_t1)
    a2 = wp.matrix_from_cols(p0_t1, p2_t1, p3_t1)
    a3 = wp.matrix_from_cols(p0_t1, p1_t1, p3_t1)
    a4 = wp.matrix_from_cols(p0_t1, p1_t1, p2_t1)

    b1 = wp.matrix_from_cols(p1_t0, p2_t0, p3_t0)
    b2 = wp.matrix_from_cols(p0_t0, p2_t0, p3_t0)
    b3 = wp.matrix_from_cols(p0_t0, p1_t0, p3_t0)
    b4 = wp.matrix_from_cols(p0_t0, p1_t0, p2_t0)

    a1 -= b1
    a2 -= b2
    a3 -= b3
    a4 -= b4

    t = det_polynomial(a1, b1) - det_polynomial(a2, b2) + det_polynomial(a3, b3) - det_polynomial(a4, b4)

    found, roots = cubic_roots(t, scalar(0.0), scalar(1.0))
    return found, roots


@wp.func
def det_polynomial(a: mat33, b: mat33) -> vec4:
    z = scalar(0.0)
    pos_polynomial = vec4()
    neg_polynomial = vec4()

    c11c22c33 = mat23(
        a[0, 0], a[1, 1], a[2, 2],
        b[0, 0], b[1, 1], b[2, 2]
    )
    c12c23c31 = mat23(
        a[0, 1], a[1, 2], a[2, 0],
        b[0, 1], b[1, 2], b[2, 0]
    )
    c13c21c32 = mat23(
        a[0, 2], a[1, 0], a[2, 1],
        b[0, 2], b[1, 0], b[2, 1]
    )
    c11c23c32 = mat23(
        a[0, 0], a[1, 2], a[2, 1],
        b[0, 0], b[1, 2], b[2, 1]
    )
    c12c21c33 = mat23(
        a[0, 1], a[1, 0], a[2, 2],
        b[0, 1], b[1, 0], b[2, 2]
    )
    c13c22c31 = mat23(
        a[0, 2], a[1, 1], a[2, 0],
        b[0, 2], b[1, 1], b[2, 0]
    )

    pos_polynomial += cubic_binomial(c11c22c33[0], c11c22c33[1])
    pos_polynomial += cubic_binomial(c12c23c31[0], c12c23c31[1])
    pos_polynomial += cubic_binomial(c13c21c32[0], c13c21c32[1])
    neg_polynomial += cubic_binomial(c11c23c32[0], c11c23c32[1])
    neg_polynomial += cubic_binomial(c12c21c33[0], c12c21c33[1])
    neg_polynomial += cubic_binomial(c13c22c31[0], c13c22c31[1])

    return pos_polynomial - neg_polynomial


@wp.func
def cubic_binomial(a: vec3, b:vec3):
    return vec4(
        b[0] * b[1] * b[2],
        a[0] * b[1] * b[2] + b[0] * b[1] * a[2] + b[0] * a[1] * b[2],
        a[0] * a[1] * b[2] + a[0] * b[1] * a[2] + b[0] * a[1] * a[2],
        a[0] * a[1] * a[2]
    )



@wp.func
def pt_collision_time(
    p0_t0: vec3, p1_t0: vec3, p2_t0: vec3, p3_t0: vec3,

    p0_t1: vec3, p1_t1: vec3, p2_t1: vec3, p3_t1: vec3
):
    n_roots, roots = build_and_solve_4_points_coplanar(p0_t0, p1_t0, p2_t0, p3_t0, p0_t1, p1_t1, p2_t1, p3_t1)

    root = scalar(1.0)
    true_root = bool(False)
    for i in range(n_roots):
        root = roots[i]
        p0t = wp.lerp(p0_t0, p0_t1, root)
        p1t = wp.lerp(p1_t0, p1_t1, root)
        p2t = wp.lerp(p2_t0, p2_t1, root)
        p3t = wp.lerp(p3_t0, p3_t1, root)
        true_root = verify_root_pt(p0t, p1t, p2t, p3t)
        if true_root:
            break

    if not true_root:
        root = scalar(1.0)

    return root

@wp.func
def ee_collision_time(
    ei0_t0: vec3,  
    ei1_t0: vec3, 
    ej0_t0: vec3, 
    ej1_t0: vec3, 
    ei0_t1: vec3,
    ei1_t1: vec3,
    ej0_t1: vec3,
    ej1_t1: vec3
):
    n_roots, roots = build_and_solve_4_points_coplanar(ei0_t0, ei1_t0, ej0_t0, ej1_t0, ei0_t1, ei1_t1, ej0_t1, ej1_t1)

    root = scalar(1.0)
    true_root = bool(False)
    for i in range(n_roots):
        root = roots[i]
        ei0 = wp.lerp(ei0_t0, ei0_t1, root)
        ei1 = wp.lerp(ei1_t0, ei1_t1, root)
        ej0 = wp.lerp(ej0_t0, ej0_t1, root)
        ej1 = wp.lerp(ej1_t0, ej1_t1, root)
        true_root = verify_root_ee(ei0, ei1, ej0, ej1)
        if true_root:
            break

    if not true_root:
        root = scalar(1.0)

    return root


@wp.kernel
def point_triangle_toi(
    triangle_bvh: wp.uint64,
    x: wp.array(dtype=vec3),
    dx: wp.array(dtype=vec3),
    triangles: wp.array(dtype=int),
    body: wp.array(dtype=int),
    toi: wp.array(dtype=scalar),
    padding: scalar,
    exclude_same_body: bool,
):
    i = wp.tid()
    p0 = x[i]
    p1 = p0 + dx[i]
    query = wp.bvh_query_aabb(
        triangle_bvh,
        wp.vec3(wp.min(p0, p1) - vec3(padding)),
        wp.vec3(wp.max(p0, p1) + vec3(padding)),
    )

    j = int(0)
    while wp.bvh_query_next(query, j):
        t0 = triangles[3 * j]
        t1 = triangles[3 * j + 1]
        t2 = triangles[3 * j + 2]
        connected = i == t0 or i == t1 or i == t2
        filtered = exclude_same_body and body[i] == body[t0]
        if not connected and not filtered:
            t = pt_collision_time(
                p0, x[t0], x[t1], x[t2],
                p1, x[t0] + dx[t0], x[t1] + dx[t1], x[t2] + dx[t2],
            )
            if t < scalar(1.0):
                wp.atomic_min(toi, 0, t)


@wp.kernel
def edge_edge_toi(
    edge_bvh: wp.uint64,
    x: wp.array(dtype=vec3),
    dx: wp.array(dtype=vec3),
    edges: wp.array(dtype=int),
    body: wp.array(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    toi: wp.array(dtype=scalar),
    exclude_same_body: bool,
):
    i = wp.tid()
    a0 = edges[2 * i]
    a1 = edges[2 * i + 1]
    query = wp.bvh_query_aabb(edge_bvh, lower[i], upper[i])

    j = int(0)
    while wp.bvh_query_next(query, j):
        b0 = edges[2 * j]
        b1 = edges[2 * j + 1]
        connected = a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1
        filtered = exclude_same_body and body[a0] == body[b0]
        if i < j and not connected and not filtered:
            t = ee_collision_time(
                x[a0], x[a1], x[b0], x[b1],
                x[a0] + dx[a0], x[a1] + dx[a1], x[b0] + dx[b0], x[b1] + dx[b1],
            )
            if t < scalar(1.0):
                wp.atomic_min(toi, 0, t)
