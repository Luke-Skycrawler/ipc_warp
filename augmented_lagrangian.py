import numpy as np 
import warp as wp 
from fem.interface import RodComplex
from scalar_types import *
from fem.geometry import Soup
from time_integrator_base import *
from contact import ContactSolverBase, XConstraint, fetch_dist_v0v1, fetch_dist_v0v1_pt, closest_point_triangle
from ipctkwp.distance.edge_edge import x_to_grad_psd_hess_ee
from ipctkwp.distance.point_triangle import x_to_grad_psd_hess_pt
from ipctkwp.distance.mollifier import ee_mollifier_derivatives, ee_mollifier_value, ee_mollifier_threshold
from geometry.static_scene import StaticScene
from dynamic_contacts import edge_edge_distance_gradient_hessian, point_triangle_distance_gradient_hessian
'''
reference: [1] Robust and Efficient Penetration-Free Elastodynamics without Barriers
'''

Gamma = scalar(0.9)

@wp.struct 
class ALConstraint:
    a1a2b1b2: wp.vec4i
    e0e1: wp.vec2i 
    l0: scalar
    alpha: scalar
    lam: scalar
    gamma: scalar
    
@wp.struct 
class Hash: 
    '''
    Hash table that supports O(1) query, O(n) traversal and O(n) creation
    reference: https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/11-hashing.html
    '''
    table_size: int 
    cell_start: wp.array(dtype = int)
    cell_entries: wp.array(dtype = ALConstraint)

@wp.func
def hash_pos(hash: Hash, e0: int, e1: int) -> int: 
    return wp.bit_xor(e0 * 73856093, e1 * 19349663) % hash.table_size

@wp.func 
def filter_accept(c: XConstraint, tv: wp.array(dtype = scalar)) -> bool:
    i0 = c.a1a2b1b2[0]
    i1 = c.a1a2b1b2[1]
    i2 = c.a1a2b1b2[2]
    i3 = c.a1a2b1b2[3]

    ti = c.alpha - scalar(1e-6)

    t0 = tv[i0]
    t1 = tv[i1]
    t2 = tv[i2]
    t3 = tv[i3]

    if ti <= t0 or ti <= t1 or ti <= t2 or ti <= t3: 
        return True
    return False
    
@wp.kernel 
def count(hash: Hash, constraints: wp.array(dtype = XConstraint), tv: wp.array(dtype = scalar)): 
    i = wp.tid()
    if filter_accept(constraints[i], tv):
        e0e1 = constraints[i].e0e1
        idx = hash_pos(hash, e0e1[0], e0e1[1])
        wp.atomic_add(hash.cell_start, idx, 1)

@wp.func 
def to_al_constraint(xc: XConstraint) -> ALConstraint: 
    al = ALConstraint()
    al.a1a2b1b2 = xc.a1a2b1b2
    al.e0e1 = xc.e0e1
    al.l0 = xc.l0
    al.alpha = xc.alpha

    al.lam = scalar(0.0)
    al.gamma = scalar(1.0)
    return al

@wp.kernel 
def register_hash(hash: Hash, constraints: wp.array(dtype = XConstraint), tv: wp.array(dtype = scalar)): 
    i = wp.tid()
    if filter_accept(constraints[i], tv):
        e0e1 = constraints[i].e0e1
        idx = hash_pos(hash, e0e1[0], e0e1[1])

        fill_in_idx = wp.atomic_add(hash.cell_start, idx, -1) - 1

        hash.cell_entries[fill_in_idx] = to_al_constraint(constraints[i])

@wp.func
def filter_old(c: ALConstraint) -> bool:
    return c.gamma >= scalar(0.01)

@wp.func 
def filter_new(old: Hash, c: ALConstraint) -> bool:
    '''
    decline existing constraints that are already in the old hash table
    '''
    e0e1 = c.e0e1
    idx = hash_pos(old, e0e1[0], e0e1[1])
    for i in range(old.cell_start[idx], old.cell_start[idx + 1]): 
        if old.cell_entries[i].e0e1[0] == e0e1[0] and old.cell_entries[i].e0e1[1] == e0e1[1]: 
            return False
    return True

@wp.kernel
def count_merge(new: Hash, old: Hash, dst: Hash, n_old: int, n_new: int):
    i = wp.tid() 
    old = i < n_old

    if old: 
        if filter_old(old.cell_entries[i]):
            e0e1 = old.cell_entries[i].e0e1
            idx = hash_pos(dst, e0e1[0], e0e1[1])
            wp.atomic_add(dst.cell_start, idx, 1)
    else:
        if filter_new(old, new.cell_entries[i - n_old]): 
            e0e1 = new.cell_entries[i - n_old].e0e1
            idx = hash_pos(dst, e0e1[0], e0e1[1])
            wp.atomic_add(dst.cell_start, idx, 1)
            
@wp.kernel 
def register_merge(new: Hash, old: Hash, dst: Hash, n_old: int, n_new: int):
    i = wp.tid()
    old = i < n_old
    if old: 
        if filter_old(old.cell_entries[i]):
            e0e1 = old.cell_entries[i].e0e1
            idx = hash_pos(dst, e0e1[0], e0e1[1])
            fill_in_idx = wp.atomic_add(dst.cell_start, idx, -1) - 1
            dst.cell_entries[fill_in_idx] = old.cell_entries[i]
    else: 
        if filter_new(old, new.cell_entries[i - n_old]): 
            e0e1 = new.cell_entries[i - n_old].e0e1
            idx = hash_pos(dst, e0e1[0], e0e1[1])
            fill_in_idx = wp.atomic_add(dst.cell_start, idx, -1) - 1
            dst.cell_entries[fill_in_idx] = new.cell_entries[i - n_old]


@wp.func 
def fetch_dist_squared_ee(soup: Soup, c: ALConstraint):
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

    return dist * dist


@wp.func 
def fetch_dist_squared_pt(soup: Soup, c: ALConstraint):
    '''
    i, t0, t1, t2 
    '''
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

    return dist * dist

@wp.kernel
def update_active_set_forces(hash: Hash, is_pt: bool, mu: scalar): 
    i = wp.tid()
    z = scalar(0.0)
    hi = hash.cell_entries[i]

    d = 0.0
    if is_pt: 
        d = fetch_dist_squared_pt() 
    else:
        d = fetch_dist_squared_ee()
    delta = hi.l0
    ci = d - delta * delta

    si = wp.max(z, ci - hi.lam / mu)

    if si == z: 
        hash.cell_entries[i].lam = hi.lam - mu * ci
        hash.cell_entries[i].gamma *= Gamma
    else: 
        hash.cell_entries[i].lam = z
        hash.cell_entries[i].gamma = scalar(1.0)


class ActiveSet: 
    '''
    Hash table 
    '''
    def __init__(self, n_edges, n_triangles, n_surface_nodes, contact_volume): 
        self.hash = self.reserve(contact_volume)

    def reserve(self, contact_volume):
        self.contact_volume = contact_volume
        hash = Hash()
        hash.table_size = 2 * contact_volume 
        hash.cell_start = wp.zeros((hash.table_size + 1, ), dtype = int)
        hash.cell_entries = wp.zeros((contact_volume, ), dtype = ALConstraint)
        return hash

    def create_hash(self, constraints, n_constraints): 
        self.hash.cell_start.zero_()
        self.hash.cell_entries.zero_()

        wp.launch(count, dim = (n_constraints, ), inputs = [self.hash, constraints])

        # prefix sum using wp.tile_scan_inclusive
        wp.launch_tiled(tile_scan_inclusive, dim=[], inputs=[self.hash.cell_entries], block_dim=64)


        # fill in contact information, init with gamma = 1.0, lambda = 0.0
        wp.launch(register_hash, dim = (n_constraints, ), inputs = [self.hash, constraints])

    def copy_from(self, a: ActiveSet):

        if self.contact_volume < a.contact_volume:
            self.hash = self.reserve(a.contact_volume)
        
        wp.copy(self.hash.cell_start, a.hash.cell_start)
        wp.copy(self.hash.cell_entries, a.hash.cell_entries)

    def merge(self, a: ActiveSet): 
        new_hash = self.reserve(self.contact_volume)

        n_constraints_old = a.hash.cell_start.numpy()[-1]
        n_constraints_new = self.hash.cell_start.numpy()[-1]

        wp.launch(count_merge, dim = (n_constraints_old + n_constraints_new, ), inputs = [a.hash, self.hash, new_hash, n_constraints_old, n_constraints_new])

        wp.launch_tiled(tile_scan_inclusive, dim=[], inputs=[new_hash.cell_entries], block_dim=64)

        wp.launch(register_merge, dim = (n_constraints_old + n_constraints_new, ), inputs = [a.hash, self.hash, new_hash, n_constraints_old, n_constraints_new])

        return new_hash

    def update_forces(self, is_pt, mu):
        n_constraints = self.hash.cell_start.numpy()[-1]
        wp.launch(update_active_set_forces, dim = (n_constraints, ), inputs = [self.hash, is_pt, mu])   
        
@wp.kernel
def contact_gauss_newton_ee(states: NewtonState, soup: Soup, constraints: wp.array(dtype = ALConstraint), triplets: Triplets, b: wp.array(dtype = vec3), mu: scalar):
    z = scalar(0.0)
    i = wp.tid() 
    c = constraints[i]

    dist_sqr = fetch_dist_squared_ee(soup, c)
    x0 = states.x[c.a1a2b1b2[0]]
    x1 = states.x[c.a1a2b1b2[1]]
    x2 = states.x[c.a1a2b1b2[2]]
    x3 = states.x[c.a1a2b1b2[3]]
    grad, hess = edge_edge_distance_gradient_hessian(x0, x1, x2, x3)

    ci = dist_sqr - c.l0 * c.l0
    si = wp.max(z, ci - c.lam / mu)

    term_hess = mu * c.gamma
    term = term_hess * (ci - c.lam / mu - si)
    grad_output *= term

    for ii in range(4): 
        gii = vec3(grad_output[ii * 3 + 0], grad_output[ii * 3 + 1], grad_output[ii * 3 + 2])
        wp.atomic_add(b, c.a1a2b1b2[ii], gii)
        for jj in range(4):
            triplets.rows[i * 16 + ii * 4 + jj] = c.a1a2b1b2[ii]
            triplets.cols[i * 16 + ii * 4 + jj] = c.a1a2b1b2[jj]
            triplets.vals[i * 16 + ii * 4 + jj] = wp.outer(grad[ii], grad[jj]) * term_hess
    
class RodComplexAL(RodBCBase, RodComplex, ContactSolverBase): 
    def __init__(self, h, meshes = [], transforms = [], static_meshes:StaticScene = None):
        self.meshes_filename = meshes 
        self.transforms = transforms
        RodBCBase.__init__(self, h)
        self.soup.x_transformed = self.states.x
        ContactSolverBase.__init__(self)
        self.eps = 1e-3
        self.k_min = 2

        self.x_t = wp.zeros_like(self.states.x)
        self.x_tilde = wp.zeros_like(self.states.x)

        n_edges = self.soup.edges.shape[0] // 2
        n_triangles = self.soup.triangles.shape[0] // 3
        n_pts = self.n_nodes
        # fixme: replace it with number of surface nodes
        contact_volume_ee = self.contacts_new.capacity
    
        self.active_set_ee = ActiveSet(n_edges, n_triangles, n_pts, contact_volume_ee)

        self.active_set_pt = ActiveSet(n_edges, n_triangles, n_pts, contact_volume_ee)

        self.active_set_ee_old = ActiveSet(n_edges, n_triangles, n_pts, contact_volume_ee)

        self.active_set_pt_old = ActiveSet(n_edges, n_triangles, n_pts, contact_volume_ee)

    def compute_mu(self):
        return 1.0
        
    def compute_A(self):
        super().compute_A()
        # collision set is the penetrated constraints captured by ccd
        self.collision_triplets = Triplets()
        nnz = (self.n_contacts + self.n_contacts_pt) * 4 * 4
        self.collision_triplets.rows = wp.zeros((nnz,), dtype = int)
        self.collision_triplets.cols = wp.zeros_like(self.collision_triplets.rows)
        self.collision_triplets.vals = wp.zeros((nnz,), dtype = mat33)
        nnz = (self.n_contacts + self.n_contacts_pt) * 16 

        self.mu = self.compute_mu()

        n_contact_ee = self.active_set_ee.hash.cell_start.numpy()[-1]
        n_contact_pt = self.active_set_pt.hash.cell_start.numpy()[-1]

        wp.launch(contact_gauss_newton_ee, dim = (n_contact_ee, ), inputs = [self.states, self.soup, self.active_set_ee.hash.cell_entries, self.collision_triplets, self.b])
        wp.launch(contact_gauss_newton_pt, dim = (n_contact_pt, ), inputs = [self.states, self.soup, self.active_set_pt.hash.cell_entries, self.collision_triplets, self.b, n_contact_ee])
        

    def compute_collision_energy(self):
        pass 


    def step(self):
        '''
        Alg. 1 in [1] 
        '''
        self.compute_tilde_x()
        self.move_boundary()

        beta = 1.0 
        iter = 0
        while beta > self.eps:
            self.solve_subproblem() 
            self.update_active_set()

            alpha = self.collision_free_step(self.states.dx)

            wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])

            if iter + 1 >= self.k_min:
                beta *= 1 - alpha 

            iter += 1 

        wp.copy(self.states.x0, self.states.x_t)
        self.update_x0_xdot()
        self.theta += self.h
        self.frame += 1


    def compute_tilde_x(self):
        wp.copy(self.x_t, self.states.x)
        wp.launch(compute_tilde_x, dim = (self.n_nodes, ), inputs = [self.x_tilde, self.states])

    def move_boundary(self):
        pass

    def solve_subproblem(self):
        '''
        Alg. 2 in [1] 
        '''
        super().step()
        # fixme: don't advance self.theta and self.frame

        self.active_set_ee.update_forces(False, self.mu)
        self.active_set_pt.update_forces(True, self.mu)

    def update_active_set(self):
        '''
        Alg. 3 in [1] 
        '''
        self.new_intersections(self.states.x0, self.states.x)

        self.active_set_ee.create_hash(self.contacts_new)
        self.active_set_pt.create_hash(self.contacts_new)

        self.active_set_ee_old = self.active_set_ee.merge(self.active_set_ee_old)
        self.active_set_pt_old = self.active_set_pt.merge(self.active_set_pt_old)

        self.active_set_ee.copy_from(self.active_set_ee_old)
        self.active_set_pt.copy_from(self.active_set_pt_old)
