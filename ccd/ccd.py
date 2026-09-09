from scalar_types import *
from ccd.cubic_roots import cubic_roots
from ipctkwp.distance.ee import beta_gamma_ee
from ipctkwp.distance.pt import beta_gamma_pt


@wp.func
def _segment_segment_distance_sq(a0: vec3, a1: vec3, b0: vec3, b1: vec3):
    """Robust squared segment distance, including nearly parallel edges."""
    da = a1 - a0
    db = b1 - b0
    r = a0 - b0
    aa = wp.dot(da, da)
    bb = wp.dot(db, db)
    f = wp.dot(db, r)
    s = scalar(0.0)
    t = scalar(0.0)
    if aa <= scalar(1.0e-20) and bb <= scalar(1.0e-20):
        return wp.length_sq(r)
    if aa <= scalar(1.0e-20):
        t = wp.clamp(f / bb, scalar(0.0), scalar(1.0))
    elif bb <= scalar(1.0e-20):
        s = wp.clamp(-wp.dot(da, r) / aa, scalar(0.0), scalar(1.0))
    else:
        ab = wp.dot(da, db)
        c = wp.dot(da, r)
        denom = aa * bb - ab * ab
        if denom > scalar(1.0e-20):
            s = wp.clamp((ab * f - c * bb) / denom, scalar(0.0), scalar(1.0))
        t = (ab * s + f) / bb
        if t < scalar(0.0):
            t = scalar(0.0)
            s = wp.clamp(-c / aa, scalar(0.0), scalar(1.0))
        elif t > scalar(1.0):
            t = scalar(1.0)
            s = wp.clamp((ab - c) / aa, scalar(0.0), scalar(1.0))
    d = (a0 + s * da) - (b0 + t * db)
    return wp.length_sq(d)


@wp.func
def _point_triangle_distance_sq(p: vec3, a: vec3, b: vec3, c: vec3):
    """Squared point-triangle distance with Voronoi feature handling."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= scalar(0.0) and d2 <= scalar(0.0):
        return wp.length_sq(ap)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= scalar(0.0) and d4 <= d3:
        return wp.length_sq(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= scalar(0.0) and d1 >= scalar(0.0) and d3 <= scalar(0.0):
        v = d1 / (d1 - d3)
        return wp.length_sq(p - (a + v * ab))
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= scalar(0.0) and d5 <= d6:
        return wp.length_sq(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= scalar(0.0) and d2 >= scalar(0.0) and d6 <= scalar(0.0):
        w = d2 / (d2 - d6)
        return wp.length_sq(p - (a + w * ac))
    va = d3 * d6 - d5 * d4
    if va <= scalar(0.0) and d4 >= d3 and d5 >= d6:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.length_sq(p - (b + w * (c - b)))
    denom = va + vb + vc
    if wp.abs(denom) <= scalar(1.0e-20):
        # Degenerate triangle: its three edges still define a safe distance.
        d_ab = _segment_segment_distance_sq(p, p, a, b)
        d_ac = _segment_segment_distance_sq(p, p, a, c)
        d_bc = _segment_segment_distance_sq(p, p, b, c)
        return wp.min(d_ab, wp.min(d_ac, d_bc))
    inv_denom = scalar(1.0) / denom
    v = vb * inv_denom
    w = vc * inv_denom
    return wp.length_sq(p - (a + v * ab + w * ac))


@wp.func
def conservative_ee_toi(
    a0: vec3, a1: vec3, b0: vec3, b1: vec3,
    a0_end: vec3, a1_end: vec3, b0_end: vec3, b1_end: vec3,
):
    """Distance-based conservative advancement from warp-ipc/libuipc."""
    da0 = a0_end - a0
    da1 = a1_end - a1
    db0 = b0_end - b0
    db1 = b1_end - b1
    mean = (da0 + da1 + db0 + db1) * scalar(0.25)
    da0 -= mean
    da1 -= mean
    db0 -= mean
    db1 -= mean
    max_a = wp.sqrt(wp.max(wp.length_sq(da0), wp.length_sq(da1)))
    max_b = wp.sqrt(wp.max(wp.length_sq(db0), wp.length_sq(db1)))
    max_disp = max_a + max_b
    if max_disp <= scalar(1.0e-20):
        return scalar(1.0)
    initial_distance = wp.sqrt(_segment_segment_distance_sq(a0, a1, b0, b1))
    gap = scalar(0.1) * initial_distance
    distance = initial_distance
    toi = scalar(0.0)
    for iteration in range(100):
        lower_bound = scalar(0.9) * distance / max_disp
        if lower_bound <= scalar(1.0e-15):
            return toi
        a0 += lower_bound * da0
        a1 += lower_bound * da1
        b0 += lower_bound * db0
        b1 += lower_bound * db1
        distance = wp.sqrt(_segment_segment_distance_sq(a0, a1, b0, b1))
        if toi > scalar(0.0) and distance < gap:
            return toi
        toi += lower_bound
        if toi >= scalar(1.0):
            return scalar(1.0)
    return wp.min(toi, scalar(1.0))


@wp.func
def conservative_pt_toi(
    p: vec3, a: vec3, b: vec3, c: vec3,
    p_end: vec3, a_end: vec3, b_end: vec3, c_end: vec3,
):
    dp = p_end - p
    da = a_end - a
    db = b_end - b
    dc = c_end - c
    mean = (dp + da + db + dc) * scalar(0.25)
    dp -= mean
    da -= mean
    db -= mean
    dc -= mean
    max_tri = wp.sqrt(wp.max(wp.length_sq(da), wp.max(wp.length_sq(db), wp.length_sq(dc))))
    max_disp = wp.sqrt(wp.length_sq(dp)) + max_tri
    if max_disp <= scalar(1.0e-20):
        return scalar(1.0)
    initial_distance = wp.sqrt(_point_triangle_distance_sq(p, a, b, c))
    gap = scalar(0.1) * initial_distance
    distance = initial_distance
    toi = scalar(0.0)
    for iteration in range(100):
        lower_bound = scalar(0.9) * distance / max_disp
        if lower_bound <= scalar(1.0e-15):
            return toi
        p += lower_bound * dp
        a += lower_bound * da
        b += lower_bound * db
        c += lower_bound * dc
        distance = wp.sqrt(_point_triangle_distance_sq(p, a, b, c))
        if toi > scalar(0.0) and distance < gap:
            return toi
        toi += lower_bound
        if toi >= scalar(1.0):
            return scalar(1.0)
    return wp.min(toi, scalar(1.0))


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
            # t = conservative_pt_toi(
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
            # t = conservative_ee_toi(
            t = ee_collision_time(
                x[a0], x[a1], x[b0], x[b1],
                x[a0] + dx[a0], x[a1] + dx[a1], x[b0] + dx[b0], x[b1] + dx[b1],
            )
            if t < scalar(1.0):
                wp.atomic_min(toi, 0, t)
