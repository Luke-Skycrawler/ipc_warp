import warp as wp 
import numpy as np 
from scalar_types import *
from fem.geometry import Soup
from fem.params import NewtonState
from ccd.ccd import (
    edge_edge_toi,
    point_triangle_toi,
    swept_edge_aabbs,
    swept_triangle_aabbs,
    pt_collision_time,
    ee_collision_time,
)
_thickness = 0.01
contact_volume = 10000
# buffer = 0.01
buffer = _thickness
eps = 1e-6
FLT_MAX = 1e5
ZERO = 1e-6
disable_self_collision = False
verbose = False

'''
TODO: make sure max_unroll = 0 before importing this module

(see `fix_interference` kernel)
'''

@wp.kernel 
def edge_aabb(x: wp.array(dtype = vec3), edges: wp.array(dtype = int), aabb_lower: wp.array(dtype = wp.vec3), aabb_upper: wp.array(dtype = wp.vec3), thickness: scalar):
    i = wp.tid()
    p0 = x[edges[i * 2]]
    p1 = x[edges[i * 2 + 1]]

    aabb_lower[i] = wp.vec3(wp.min(p0, p1) - vec3(thickness))
    aabb_upper[i] = wp.vec3(wp.max(p0, p1) + vec3(thickness))
    
@wp.kernel
def c_gets_i_mod_2(color: wp.array(dtype = int)):
    i = wp.tid() 
    color[i] = i % 2

@wp.kernel
def _copy(dst: wp.array(dtype = wp.vec3), src: wp.array(dtype = vec3)):
    i = wp.tid()
    dst[i] = wp.vec3(src[i])

@wp.kernel
def _add_dx(dst: wp.array(dtype = vec3), x0: wp.array(dtype = vec3), dx: wp.array(dtype = vec3)):
    i = wp.tid()
    dst[i] = x0[i] - dx[i]

# @wp.struct 
# class ContactInfo:
#     a1a2b1b2: wp.vec4i
#     lam: scalar
#     k: scalar
#     cj: scalar

@wp.struct 
class XConstraint: 
    a1a2b1b2: wp.vec4i
    e0e1: wp.vec2i 
    l0: scalar
    alpha: scalar
    lam: scalar

ContactInfo = XConstraint
# @wp.struct
# class HTableEntry: 
#     list_idx: int
#     # updated_stamp: int

@wp.struct 
class Contacts:
    list: wp.array(dtype = ContactInfo)
    cnt: wp.array(dtype = int)
    htable: wp.array(dtype = int)
    capacity: int

@wp.struct 
class ContactRet:
    points: wp.array(dtype = vec3)
    dists: wp.array(dtype = scalar)


@wp.func 
def fetch_dist_v0v1(p: NewtonState, soup: Soup, c: XConstraint):
    l0 = c.l0
    i0 = c.a1a2b1b2[0]
    i1 = c.a1a2b1b2[1]
    i2 = c.a1a2b1b2[2]
    i3 = c.a1a2b1b2[3]
    
    b0 = soup.body[i0]
    b1 = soup.body[i2]

    x0 = soup.x_transformed[i0]
    x1 = soup.x_transformed[i1]
    x2 = soup.x_transformed[i2]
    x3 = soup.x_transformed[i3]

    dab = wp.closest_point_edge_edge(wp.vec3(x0), wp.vec3(x1), wp.vec3(x2), wp.vec3(x3), eps)
    v0 = wp.lerp(x0, x1, scalar(dab[0]))
    v1 = wp.lerp(x2, x3, scalar(dab[1]))

    dist = scalar(dab[2])

    return dist, v0, v1

@wp.func 
def triangle_normal(x0: vec3, x1: vec3, x2: vec3):
    return wp.normalize(wp.cross(x1 - x0, x2 - x0))

@wp.func 
def fetch_dist_v0v1_pt(p: NewtonState, soup: Soup, c: XConstraint):
    '''
    i, t0, t1, t2 
    '''
    o = scalar(1.0)
    z = scalar(0.0)

    l0 = c.l0
    i = c.a1a2b1b2[0]
    t0 = c.a1a2b1b2[1]
    t1 = c.a1a2b1b2[2]
    t2 = c.a1a2b1b2[3]
    
    
    x0 = soup.x_transformed[i]
    x1 = soup.x_transformed[t0]
    x2 = soup.x_transformed[t1]
    x3 = soup.x_transformed[t2]

    dab, type = closest_point_triangle(wp.vec3(x0), wp.vec3(x1), wp.vec3(x2), wp.vec3(x3))

    v0 = x0
    alpha = scalar(dab[0])
    beta = scalar(dab[1])
    v1 = alpha * x1 + beta * x2 + (scalar(1.0) - alpha - beta) * x3

    dist = scalar(dab[2])

    return dist, v0, v1

@wp.kernel
def get_contact_points(p: NewtonState, soup: Soup, xconstraints: wp.array(dtype = XConstraint), contact_ret: ContactRet):
    i = wp.tid()
    c = xconstraints[i]
    dist, v0, v1 = fetch_dist_v0v1(p, soup, c)
    contact_ret.points[i] = (v0 + v1) * scalar(0.5)
    contact_ret.dists[i] = dist

@wp.kernel
def get_contact_points_pt(p: NewtonState, soup: Soup, xconstraints: wp.array(dtype = XConstraint), contact_ret: ContactRet):
    i = wp.tid()
    c = xconstraints[i]
    dist, v0, v1 = fetch_dist_v0v1_pt(p, soup, c)
    contact_ret.points[i] = (v0 + v1) * scalar(0.5)
    contact_ret.dists[i] = dist

@wp.func
def _hash(a1: int, b1: int) -> int:
    '''
    Hash function in "Optimized Spatial Hashing for Collision Detection of Deformable Objects"
    '''
    h = wp.bit_xor(a1 * 73856093, b1 * 19349663)
    return h % 8191

@wp.func
def append(contacts: Contacts, a1: int, a2: int, b1: int, b2: int, thickness: float, ei: int, ej: int):
    '''
    for edges ei < ej; for point-triangle i, j is the index for point and triangle respectively 
    '''
    idx = wp.atomic_add(contacts.cnt, 0, 1)
    if idx < contacts.capacity:
        h = _hash(a1, b1)
        contacts.list[idx].a1a2b1b2 = wp.vec4i(a1, a2, b1, b2)
        contacts.list[idx].l0 = scalar(thickness) * scalar(2.0)
        contacts.list[idx].alpha = scalar(1e-6)
        contacts.list[idx].e0e1 = wp.vec2i(ei, ej)
        contacts.htable[h] = idx

@wp.func
def append_ccd(contacts: Contacts, a1: int, a2: int, b1: int, b2: int, thickness: scalar, ei: int, ej: int, toi: scalar):
    '''
    for edges ei < ej; for point-triangle i, j is the index for point and triangle respectively 
    '''
    idx = wp.atomic_add(contacts.cnt, 0, 1)
    if idx < contacts.capacity:
        h = _hash(a1, b1)
        contacts.list[idx].a1a2b1b2 = wp.vec4i(a1, a2, b1, b2)
        contacts.list[idx].l0 = thickness * scalar(2.0)
        contacts.list[idx].alpha = toi
        contacts.list[idx].e0e1 = wp.vec2i(ei, ej)
        contacts.htable[h] = idx

@wp.kernel
def edge_edge_collision(bvh: wp.uint64, soup: Soup, contacts: Contacts, thickness: float):
    i = wp.tid()
    if True:
        # edge exists
        a1 = soup.edges[i * 2]
        a2 = soup.edges[i * 2 + 1]
        p1 = wp.vec3(soup.x_transformed[a1])
        p2 = wp.vec3(soup.x_transformed[a2])

        low = wp.min(p1, p2) - wp.vec3(thickness + buffer)
        high = wp.max(p1, p2) + wp.vec3(thickness + buffer)
        query = wp.bvh_query_aabb(bvh, low, high)
        j = int(0) 
        while wp.bvh_query_next(query, j):
            connected = soup.edges[i * 2] == soup.edges[j * 2] or soup.edges[i * 2] == soup.edges[j * 2 + 1] or soup.edges[i * 2 + 1] == soup.edges[j * 2] or soup.edges[i * 2 + 1] == soup.edges[j * 2 + 1]

            self_collision = False
            if wp.static(disable_self_collision):
                self_collision = soup.body[a1] == soup.body[soup.edges[j * 2]]
            if i < j and not connected and not self_collision: 
                b1 = soup.edges[j * 2]
                b2 = soup.edges[j * 2 + 1]
                
                q1 = wp.vec3(soup.x_transformed[b1])
                q2 = wp.vec3(soup.x_transformed[b2])
                std = wp.closest_point_edge_edge(p1, p2, q1, q2, 1e-6)
                dist = std[2]
                if dist < thickness * 2.0:
                    append(contacts, a1, a2, b1, b2, thickness, i, j)
@wp.func
def closest_point_triangle(
    p: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
):
    ret = wp.vec3(0.0)
    type = int(-1)
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    # Vertex region A
    if d1 <= 0.0 and d2 <= 0.0:
        ret = wp.vec3(1.0, 0.0, wp.length(p - a))
        type = 0

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    # Vertex region B
    if d3 >= 0.0 and d4 <= d3 and type == -1:
        ret = wp.vec3(0.0, 1.0, wp.length(p - b))
        type = 1
    # Edge region AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0 and type == -1:
        v = d1 / (d1 - d3)
        point = a + v * ab
        ret = wp.vec3(1.0 - v, v, wp.length(p - point))
        type = 3

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    # Vertex region C
    if d6 >= 0.0 and d5 <= d6 and type == -1:
        ret = wp.vec3(0.0, 0.0, wp.length(p - c))
        type = 2

    # Edge region AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0 and type == -1:
        w = d2 / (d2 - d6)
        point = a + w * ac
        ret = wp.vec3(1.0 - w, 0.0, wp.length(p - point))
        type = 4

    # Edge region BC
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0 and type == -1:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        point = b + w * (c - b)
        ret = wp.vec3(0.0, 1.0 - w, wp.length(p - point))
        type = 5

    if type == -1:
        # Face region
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        u = 1.0 - v - w

        point = a + ab * v + ac * w
        ret = wp.vec3(u, v, wp.length(p - point))
        type = 6

    return ret, type
        
@wp.kernel
def point_triangle_collision(bvh: wp.uint64, soup: Soup, contacts: Contacts, thickness: float):
    i = wp.tid()
    if True:
        # point not inverted 
        xi = wp.vec3(soup.x_transformed[i])
        low = xi - wp.vec3((thickness + buffer) * 2.0)
        high = xi + wp.vec3((thickness + buffer) * 2.0)
        query = wp.mesh_query_aabb(bvh, low, high)

        j = int(0) 
        while wp.mesh_query_aabb_next(query, j):
            connected = soup.triangles[j * 3] == i or soup.triangles[j * 3 + 1] == i or soup.triangles[j * 3 + 2] == i

            self_collision = False
            if wp.static(disable_self_collision):
                self_collision = soup.body[i] == soup.body[soup.triangles[j * 3]]
            
            if not connected and not self_collision: 
                t1 = soup.triangles[j * 3]
                t2 = soup.triangles[j * 3 + 1]
                t3 = soup.triangles[j * 3 + 2]

                q1 = wp.vec3(soup.x_transformed[t1])
                q2 = wp.vec3(soup.x_transformed[t2])
                q3 = wp.vec3(soup.x_transformed[t3])

                std, _ = closest_point_triangle(xi, q1, q2, q3)

                dist = std[2]
                if dist < thickness * 2.0:
                    append(contacts, i, t2, t1, t3, thickness, i, j)

@wp.func 
def fix_interference(v: wp.vec4i, color: wp.array(dtype = int), dirty: wp.array(dtype = bool)):
    colors = wp.vec4i(color[v.x], color[v.y], color[v.z], color[v.w])
    cm = wp.max(colors)

    for ii in range(1, 4):
        for jj in range(ii):
            if color[v[jj]] == color[v[ii]]:
                color[v[ii]] = cm + 1
                cm += 1
                dirty[v[ii]] = True

@wp.kernel
def color_contacts(contacts: Contacts, color: wp.array(dtype = int), dirty: wp.array(dtype = bool)):
    i = wp.tid() 
    if i < contacts.cnt[0]:
        fix_interference(contacts.list[i].a1a2b1b2, color, dirty)

@wp.kernel
def compute_normal_kernel(x: wp.array(dtype = vec3), triangles: wp.array(dtype = int), N: wp.array(dtype = vec3)):
    i = wp.tid()
    t1 = triangles[i * 3]
    t2 = triangles[i * 3 + 1]
    t3 = triangles[i * 3 + 2]

    x1 = x[t1]
    x2 = x[t2]
    x3 = x[t3]

    N[i] = triangle_normal(x1, x2, x3)


@wp.kernel
def point_triangle_intersections(
    triangle_bvh: wp.uint64,
    x0: wp.array(dtype=vec3),
    x1: wp.array(dtype=vec3),
    triangles: wp.array(dtype=int),
    body: wp.array(dtype=int),
    toi: wp.array(dtype=scalar),
    padding: scalar,
    exclude_same_body: bool,
    contacts: Contacts,
    thickness: scalar
):
    i = wp.tid()
    p0 = x0[i]
    p1 = x1[i]
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
                p0, x0[t0], x0[t1], x0[t2],
                p1, x1[t0], x1[t1], x1[t2],
            )
            if t < scalar(1.0):
                wp.atomic_min(toi, 0, t)
                append_ccd(contacts, i, t0, t1, t2, thickness, i, j, t)

@wp.kernel
def edge_edge_intersections(
    edge_bvh: wp.uint64,
    x0: wp.array(dtype=vec3),
    x1: wp.array(dtype=vec3),
    edges: wp.array(dtype=int),
    body: wp.array(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    toi: wp.array(dtype=scalar),
    exclude_same_body: bool,
    contacts: Contacts,
    thickness: scalar
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
                x0[a0], x0[a1], x0[b0], x0[b1],
                x1[a0], x1[a1], x1[b0], x1[b1],
            )
            if t < scalar(1.0):
                wp.atomic_min(toi, 0, t)
                append_ccd(contacts, a0, a1, b0, b1, thickness, i, j, t)

class ContactSolverBase:
    def __init__(self):
        '''
        Base contact interface to detect the edge-edge contacts. Contains an optional colorization function

        need to have self.soup: Soup defined prior to calling this constructor
        '''
        self.soup: Soup

        n_edges = self.soup.edges.shape[0] // 2
        n_nodes = self.soup.xcs.shape[0]
        
        # edges bvh
        self.bvh_edges_lower = wp.zeros((n_edges, ), dtype = wp.vec3) 
        self.bvh_edges_upper = wp.zeros((n_edges, ), dtype = wp.vec3)
        self.compute_edge_aabbs()
        self.bvh_edges = wp.Bvh(self.bvh_edges_lower, self.bvh_edges_upper)

        # Separate swept BVHs used to cap a Newton step before energy backtracking.
        self.ccd_edge_lower = wp.zeros((n_edges,), dtype=wp.vec3)
        self.ccd_edge_upper = wp.zeros((n_edges,), dtype=wp.vec3)
        self.ccd_edge_bvh = wp.Bvh(self.ccd_edge_lower, self.ccd_edge_upper)
        self.ccd_toi = wp.ones((1,), dtype=scalar)
        self.ccd_x1 = wp.zeros_like(self.soup.x_transformed)
        
        # triangles 
        self.has_triangles = self.soup.triangles.shape[0] > 0
        if self.has_triangles:
            self.x_mesh = wp.zeros((self.soup.x_transformed.shape[0], ), dtype = wp.vec3)
            # must be wp.vec3 type to construct the mesh bvh 
            # sync with self.soup.x_transformed in compute_V()
            self.tri_mesh = wp.Mesh(self.x_mesh, self.soup.triangles)
            n_triangles = self.soup.triangles.shape[0] // 3
            self.ccd_triangle_lower = wp.zeros((n_triangles,), dtype=wp.vec3)
            self.ccd_triangle_upper = wp.zeros((n_triangles,), dtype=wp.vec3)
            self.ccd_triangle_bvh = wp.Bvh(self.ccd_triangle_lower, self.ccd_triangle_upper)
        

        # color 
        self.color = wp.zeros((n_nodes, ), dtype = int)
        # self.color_cnt = wp.zeros((1,), dtype = int)
        self.dirty_bit = wp.zeros((n_nodes, ), dtype = bool)



        self.contacts_list_new = wp.zeros((contact_volume,), dtype = ContactInfo)
        self.contacts_cnt_new = wp.zeros((1,), dtype = int)
        self.contacts_htable_new = wp.zeros((8191,), dtype = int)
        self.contacts_new = Contacts()

        self.contacts_new.list = self.contacts_list_new
        self.contacts_new.cnt = self.contacts_cnt_new
        self.contacts_new.htable = self.contacts_htable_new
        self.contacts_new.capacity = contact_volume

        self.n_contacts = 0
        self.n_contacts_pt = 0

        self.contact_ret = ContactRet()
        self.contact_ret.points = wp.zeros((contact_volume,), dtype = vec3)
        self.contact_ret.dists = wp.zeros((contact_volume,), dtype = scalar)


        self.contacts_list_pt = wp.zeros((contact_volume, ), dtype = ContactInfo)
        self.contacts_cnt_pt = wp.zeros((1,), dtype = int)
        self.contacts_htable_pt = wp.zeros((8191,), dtype = int)
        self.contacts_pt = Contacts()

        self.contacts_pt.list = self.contacts_list_pt
        self.contacts_pt.cnt = self.contacts_cnt_pt
        self.contacts_pt.htable = self.contacts_htable_pt
        self.contacts_pt.capacity = contact_volume

        self.n_contacts_pt = 0

    def _grow_contact_list(self, contacts, required, point_triangle=False):
        """Grow a GPU contact list after an overflowed counting pass."""
        capacity = 1 << (required - 1).bit_length()
        storage = wp.zeros((capacity,), dtype=ContactInfo)
        contacts.list = storage
        contacts.capacity = capacity
        if point_triangle:
            self.contacts_list_pt = storage
        else:
            self.contacts_list_new = storage

        if self.contact_ret.points.shape[0] < capacity:
            self.contact_ret.points = wp.zeros((capacity,), dtype=vec3)
            self.contact_ret.dists = wp.zeros((capacity,), dtype=scalar)
        print(f"contact buffer grown to {capacity}")

    def collision_free_step(self, dx):
        """Return a zero-thickness CCD upper bound for the solver update x -= alpha * dx."""
        n_edges = self.soup.edges.shape[0] // 2
        padding = scalar(1e-7)
        wp.launch(_add_dx, self.soup.x_transformed.shape[0], inputs=[self.ccd_x1, self.soup.x_transformed, dx])
        wp.launch(
            swept_edge_aabbs,
            n_edges,
            inputs=[self.soup.x_transformed, self.ccd_x1, self.soup.edges,
                    self.ccd_edge_lower, self.ccd_edge_upper, padding],
        )
        self.ccd_edge_bvh.refit()
        self.ccd_toi.fill_(1.0)
        wp.launch(
            edge_edge_toi,
            n_edges,
            inputs=[self.ccd_edge_bvh.id, self.soup.x_transformed, self.ccd_x1,
                    self.soup.edges, self.soup.body, self.ccd_edge_lower,
                    self.ccd_edge_upper, self.ccd_toi, disable_self_collision],
        )

        if self.has_triangles:
            n_triangles = self.soup.triangles.shape[0] // 3
            wp.launch(
                swept_triangle_aabbs,
                n_triangles,
                inputs=[self.soup.x_transformed, self.ccd_x1, self.soup.triangles,
                        self.ccd_triangle_lower, self.ccd_triangle_upper, padding],
            )
            self.ccd_triangle_bvh.refit()
            wp.launch(
                point_triangle_toi,
                self.soup.x_transformed.shape[0],
                inputs=[self.ccd_triangle_bvh.id, self.soup.x_transformed, self.ccd_x1,
                        self.soup.triangles, self.soup.body, self.ccd_toi,
                        padding, disable_self_collision],
            )

        with self.profile_timer("ccd toi host transfer"):
            toi = float(self.ccd_toi.numpy()[0])
        return 0.9 * toi if toi < 1.0 else 1.0

    def new_intersections(self, x0, x1):
        '''
        Only used for Augmented Lagrangian solver (`augmented_lagrangian.py`) to detect intersections and update active contact set. (Alg. 3)
        '''

        self.contacts_new.cnt.zero_()
        self.contacts_new.htable.fill_(-1)
        
        n_edges = self.soup.edges.shape[0] // 2
        padding = scalar(1e-7)
        
        wp.launch(
            swept_edge_aabbs,
            n_edges,
            inputs=[x0, x1, self.soup.edges,
                    self.ccd_edge_lower, self.ccd_edge_upper, padding],
        )
        self.ccd_edge_bvh.refit()
        self.ccd_toi.fill_(1.0)
        wp.launch(
            edge_edge_intersections,
            n_edges,
            inputs=[self.ccd_edge_bvh.id, x0, x1,
                    self.soup.edges, self.soup.body, self.ccd_edge_lower,
                    self.ccd_edge_upper, self.ccd_toi, disable_self_collision, self.contacts_new, _thickness],
        )
        self.n_contacts = int(self.contacts_new.cnt.numpy()[0])
        if self.n_contacts > self.contacts_new.capacity:
            self._grow_contact_list(self.contacts_new, self.n_contacts)
            self.contacts_new.cnt.zero_()
            self.contacts_new.htable.fill_(-1)
            self.ccd_toi.fill_(1.0)
            wp.launch(
                edge_edge_intersections,
                n_edges,
                inputs=[self.ccd_edge_bvh.id, x0, x1,
                        self.soup.edges, self.soup.body, self.ccd_edge_lower,
                        self.ccd_edge_upper, self.ccd_toi,
                        disable_self_collision, self.contacts_new, _thickness],
            )
            self.n_contacts = int(self.contacts_new.cnt.numpy()[0])

        if self.has_triangles:

            self.contacts_pt.cnt.zero_()
            self.contacts_pt.htable.fill_(-1)

            n_triangles = self.soup.triangles.shape[0] // 3
            wp.launch(
                swept_triangle_aabbs,
                n_triangles,
                inputs=[x0, x1, self.soup.triangles,
                        self.ccd_triangle_lower, self.ccd_triangle_upper, padding],
            )
            self.ccd_triangle_bvh.refit()
            wp.launch(
                point_triangle_intersections,
                self.soup.x_transformed.shape[0],
                inputs=[self.ccd_triangle_bvh.id, x0, x1,
                        self.soup.triangles, self.soup.body, self.ccd_toi,
                        padding, disable_self_collision, self.contacts_pt, _thickness],
            )
            self.n_contacts_pt = int(self.contacts_pt.cnt.numpy()[0]) 
            if self.n_contacts_pt > self.contacts_pt.capacity:
                self._grow_contact_list(
                    self.contacts_pt, self.n_contacts_pt, point_triangle=True
                )
                self.contacts_pt.cnt.zero_()
                self.contacts_pt.htable.fill_(-1)
                wp.launch(
                    point_triangle_intersections,
                    self.soup.x_transformed.shape[0],
                    inputs=[self.ccd_triangle_bvh.id, x0, x1,
                            self.soup.triangles, self.soup.body, self.ccd_toi,
                            padding, disable_self_collision, self.contacts_pt,
                            _thickness],
                )
                self.n_contacts_pt = int(self.contacts_pt.cnt.numpy()[0])
        else:
            self.n_contacts_pt = 0

        with self.profile_timer("ccd toi host transfer"):
            toi = float(self.ccd_toi.numpy()[0])
        return 0.9 * toi if toi < 1.0 else 1.0

    def update_bvh(self):
        self.compute_edge_aabbs()
        self.bvh_edges.refit()
        if self.has_triangles:
            wp.launch(_copy, dim = self.soup.x_transformed.shape[0], inputs = [self.x_mesh, self.soup.x_transformed])
            self.tri_mesh.refit()
    
    def compute_edge_aabbs(self):
        n_edges = self.soup.edges.shape[0] // 2
        wp.launch(edge_aabb, n_edges, inputs = [self.soup.x_transformed, self.soup.edges, self.bvh_edges_lower, self.bvh_edges_upper, _thickness + buffer])

    def compute_V(self, ret = True): 
        # nothing to do since self.soup.x_transformed is tied to  RodComplexBC.states.x
        return None
        
    def detect_collision(self): 
        self.compute_V(ret = False)
        self.update_bvh()
        self.contacts_new.cnt.zero_()
        self.contacts_new.htable.fill_(-1)
        n_edges = self.soup.edges.shape[0] // 2
        
        wp.launch(edge_edge_collision, n_edges, inputs = [self.bvh_edges.id, self.soup, self.contacts_new, _thickness])
        with self.profile_timer("contact count host transfer"):
            self.n_contacts = int(self.contacts_new.cnt.numpy()[0])
        if self.n_contacts > self.contacts_new.capacity:
            self._grow_contact_list(self.contacts_new, self.n_contacts)
            self.contacts_new.cnt.zero_()
            wp.launch(edge_edge_collision, n_edges, inputs = [self.bvh_edges.id, self.soup, self.contacts_new, _thickness])
            with self.profile_timer("contact count host transfer"):
                self.n_contacts = int(self.contacts_new.cnt.numpy()[0])
        if verbose:
            print(f"n ee contacts = {self.n_contacts}")

        self.contacts_pt.cnt.zero_()
        self.contacts_pt.htable.fill_(-1)
        n_pts = self.soup.xcs.shape[0]
        if self.has_triangles:
            wp.launch(point_triangle_collision, n_pts, inputs = [self.tri_mesh.id, self.soup, self.contacts_pt, _thickness])


            with self.profile_timer("contact count host transfer"):
                self.n_contacts_pt = int(self.contacts_pt.cnt.numpy()[0])
            if self.n_contacts_pt > self.contacts_pt.capacity:
                self._grow_contact_list(self.contacts_pt, self.n_contacts_pt, point_triangle=True)
                self.contacts_pt.cnt.zero_()
                wp.launch(point_triangle_collision, n_pts, inputs = [self.tri_mesh.id, self.soup, self.contacts_pt, _thickness])
                with self.profile_timer("contact count host transfer"):
                    self.n_contacts_pt = int(self.contacts_pt.cnt.numpy()[0])
            if verbose:
                print(f"n pt contacts = {self.n_contacts_pt}")

    def get_contact_points(self):
        wp.launch(get_contact_points, (self.n_contacts,), inputs = [self.history, self.soup, self.contacts_new.list, self.contact_ret])
        # filter d > 2 * thickness
        dists = self.contact_ret.dists.numpy()[:self.n_contacts]
        points = self.contact_ret.points.numpy()[:self.n_contacts]
        
        valid = dists < _thickness * 2.0
        magnitudes = np.abs(dists[valid] - _thickness * 2.0)

        points_valid = points[valid]
        if self.has_triangles:
            wp.launch(get_contact_points_pt, (self.n_contacts_pt,), inputs = [self.history, self.soup, self.contacts_pt.list, self.contact_ret])
            
            dists_pt = self.contact_ret.dists.numpy()[:self.n_contacts_pt]

            points_pt = self.contact_ret.points.numpy()[:self.n_contacts_pt]
            valid_pt = dists_pt < _thickness * 2.0
            magnitudes_pt = np.abs(dists_pt[valid_pt] - _thickness * 2.0)
            # print(f"dists pt = {dists_pt}")
            return points_pt[valid_pt], magnitudes_pt
            
            # points_valid = np.vstack([points_valid, points_pt[valid_pt]])
            # magnitudes = np.concatenate([magnitudes, magnitudes_pt])
        return points_valid, magnitudes
