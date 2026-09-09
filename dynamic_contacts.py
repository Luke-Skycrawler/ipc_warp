import numpy as np
import warp as wp
from fem.interface import RodComplex
from geometry.static_scene import StaticScene
from scalar_types import *
from contact import ContactSolverBase, XConstraint, fetch_dist_v0v1, fetch_dist_v0v1_pt, closest_point_triangle
from ipctkwp.distance.edge_edge import x_to_grad_psd_hess_ee
from ipctkwp.distance.point_triangle import x_to_grad_psd_hess_pt
from ipctkwp.distance.barrier import ipc_barrier, ipc_barrier_derivative, ipc_barrier_derivative2
from ipctkwp.distance.mollifier import ee_mollifier_derivatives, ee_mollifier_value, ee_mollifier_threshold
from fem.geometry import Soup
from time_integrator_base import *

@wp.func
def embed_point_point_distance(x0: vec3, x1: vec3, o0: int, o1: int):
    grad = vec12()
    hess = mat12()
    d = x0 - x1
    for i in range(3):
        grad[o0 * 3 + i] = scalar(2.0) * d[i]
        grad[o1 * 3 + i] = scalar(-2.0) * d[i]
        hess[o0 * 3 + i, o0 * 3 + i] = scalar(2.0)
        hess[o0 * 3 + i, o1 * 3 + i] = scalar(-2.0)
        hess[o1 * 3 + i, o0 * 3 + i] = scalar(-2.0)
        hess[o1 * 3 + i, o1 * 3 + i] = scalar(2.0)
    return grad, hess

@wp.func
def embed_point_edge_distance(p: vec3, e0: vec3, e1: vec3, op: int, oe0: int, oe1: int):
    edge = e1 - e0
    r = p - e0
    inv_edge_len2 = scalar(1.0) / wp.dot(edge, edge)
    alpha = wp.dot(r, edge) * inv_edge_len2
    q = r - alpha * edge
    coefficients = vec3(scalar(1.0), alpha - scalar(1.0), -alpha)
    alpha_signs = vec3(scalar(0.0), scalar(1.0), scalar(-1.0))
    two_alpha_edge = scalar(2.0) * alpha * edge
    alpha_grad = wp.matrix_from_rows(
        edge * inv_edge_len2,
        (two_alpha_edge - r - edge) * inv_edge_len2,
        (r - two_alpha_edge) * inv_edge_len2,
    )
    offsets = wp.vec3i(op, oe0, oe1)
    grad = vec12()
    hess = mat12()
    for i in range(3):
        oi = offsets[i]
        for k in range(3):
            grad[oi * 3 + k] = scalar(2.0) * coefficients[i] * q[k]
        for j in range(3):
            oj = offsets[j]
            block = scalar(2.0) * wp.outer(
                alpha_signs[i] * q - coefficients[i] * edge,
                alpha_grad[j],
            )
            for k in range(3):
                for l in range(3):
                    value = block[k, l]
                    if k == l:
                        value += scalar(2.0) * coefficients[i] * coefficients[j]
                    hess[oi * 3 + k, oj * 3 + l] = value
    return grad, hess

@wp.func
def edge_edge_distance_gradient_hessian(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    dab = wp.closest_point_edge_edge(wp.vec3(x0), wp.vec3(x1), wp.vec3(x2), wp.vec3(x3), 1.0e-6)
    s = scalar(dab[0])
    t = scalar(dab[1])
    feature_eps = scalar(1.0e-6)
    s_boundary = s <= feature_eps or s >= scalar(1.0) - feature_eps
    t_boundary = t <= feature_eps or t >= scalar(1.0) - feature_eps

    if s_boundary and t_boundary:
        oa = int(0)
        ob = int(2)
        pa = x0
        pb = x2
        if s >= scalar(1.0) - feature_eps:
            oa = 1
            pa = x1
        if t >= scalar(1.0) - feature_eps:
            ob = 3
            pb = x3
        return embed_point_point_distance(pa, pb, oa, ob)

    if s_boundary:
        oa = int(0)
        pa = x0
        if s >= scalar(1.0) - feature_eps:
            oa = 1
            pa = x1
        return embed_point_edge_distance(pa, x2, x3, oa, 2, 3)

    if t_boundary:
        ob = int(2)
        pb = x2
        if t >= scalar(1.0) - feature_eps:
            ob = 3
            pb = x3
        return embed_point_edge_distance(pb, x0, x1, ob, 0, 1)

    return x_to_grad_psd_hess_ee(x0, x1, x2, x3)

@wp.func
def point_triangle_distance_gradient_hessian(x0: vec3, x1: vec3, x2: vec3, x3: vec3):
    dab, feature = closest_point_triangle(wp.vec3(x0), wp.vec3(x1), wp.vec3(x2), wp.vec3(x3))
    if feature == 0:
        return embed_point_point_distance(x0, x1, 0, 1)
    if feature == 1:
        return embed_point_point_distance(x0, x2, 0, 2)
    if feature == 2:
        return embed_point_point_distance(x0, x3, 0, 3)
    if feature == 3:
        return embed_point_edge_distance(x0, x1, x2, 0, 1, 2)
    if feature == 4:
        return embed_point_edge_distance(x0, x1, x3, 0, 1, 3)
    if feature == 5:
        return embed_point_edge_distance(x0, x2, x3, 0, 2, 3)
    return x_to_grad_psd_hess_pt(x0, x1, x2, x3)


@wp.func
def ee_mollifier_threshold_rest(soup: Soup, c: XConstraint):
    i0 = c.a1a2b1b2[0]
    i1 = c.a1a2b1b2[1]
    i2 = c.a1a2b1b2[2]
    i3 = c.a1a2b1b2[3]
    return ee_mollifier_threshold(soup.xcs[i0], soup.xcs[i1], soup.xcs[i2], soup.xcs[i3])

@wp.func
def project_spd_12(hess: mat12):
    # Cyclic Jacobi EVD, matching warp-ipc's numerical PSD projection.
    A = hess
    V = mat12()
    eigenvalues = vec12()
    accumulated = vec12()
    corrections = vec12()
    for i in range(12):
        V[i, i] = scalar(1.0)
        eigenvalues[i] = A[i, i]
        accumulated[i] = eigenvalues[i]

    sweep = int(0)
    while sweep < 4:
        threshold = scalar(0.0)
        for j in range(12):
            for i in range(j):
                threshold += A[i, j] * A[i, j]
        threshold = wp.sqrt(threshold) / scalar(48.0)
        if threshold == scalar(0.0):
            sweep = 4
        else:
            for p in range(12):
                for q in range(p + 1, 12):
                    gap = scalar(10.0) * wp.abs(A[p, q])
                    if threshold <= wp.abs(A[p, q]):
                        delta = eigenvalues[q] - eigenvalues[p]
                        t = scalar(0.0)
                        if wp.abs(delta) + gap == wp.abs(delta):
                            t = A[p, q] / delta
                        else:
                            theta = scalar(0.5) * delta / A[p, q]
                            t = scalar(1.0) / (wp.abs(theta) + wp.sqrt(scalar(1.0) + theta * theta))
                            if theta < scalar(0.0):
                                t = -t
                        c = scalar(1.0) / wp.sqrt(scalar(1.0) + t * t)
                        s = t * c
                        tau = s / (scalar(1.0) + c)
                        rotation = t * A[p, q]
                        corrections[p] -= rotation
                        corrections[q] += rotation
                        eigenvalues[p] -= rotation
                        eigenvalues[q] += rotation
                        A[p, q] = scalar(0.0)
                        for j in range(p):
                            g = A[j, p]
                            h = A[j, q]
                            A[j, p] = g - s * (h + g * tau)
                            A[j, q] = h + s * (g - h * tau)
                        for j in range(p + 1, q):
                            g = A[p, j]
                            h = A[j, q]
                            A[p, j] = g - s * (h + g * tau)
                            A[j, q] = h + s * (g - h * tau)
                        for j in range(q + 1, 12):
                            g = A[p, j]
                            h = A[q, j]
                            A[p, j] = g - s * (h + g * tau)
                            A[q, j] = h + s * (g - h * tau)
                        for j in range(12):
                            g = V[j, p]
                            h = V[j, q]
                            V[j, p] = g - s * (h + g * tau)
                            V[j, q] = h + s * (g - h * tau)
            for i in range(12):
                accumulated[i] += corrections[i]
                eigenvalues[i] = accumulated[i]
                corrections[i] = scalar(0.0)
            sweep += 1

    result = mat12()
    for k in range(12):
        eigenvalue = wp.max(eigenvalues[k], scalar(0.0))
        for i in range(12):
            for j in range(12):
                result[i, j] += eigenvalue * V[i, k] * V[j, k]
    return result

@wp.kernel
def contact_hessian_ee(states: NewtonState, soup: Soup, contacts: wp.array(dtype = XConstraint), triplets: Triplets, b: wp.array(dtype = vec3)):
    i = wp.tid()
    c = contacts[i]
    
    dist, v0, v1 = fetch_dist_v0v1(states, soup, c)
    
    if dist < c.l0:
        x0 = states.x[c.a1a2b1b2[0]]
        x1 = states.x[c.a1a2b1b2[1]]
        x2 = states.x[c.a1a2b1b2[2]]
        x3 = states.x[c.a1a2b1b2[3]]
        grad, hess = edge_edge_distance_gradient_hessian(x0, x1, x2, x3)
        d2 = dist * dist
        d02 = c.l0 * c.l0
        barrier_grad = ipc_barrier_derivative(d2, d02, contact_stiffness)
        barrier_hess = ipc_barrier_derivative2(d2, d02, contact_stiffness)
        eps_x = ee_mollifier_threshold_rest(soup, c)
        mollifier, mollifier_grad, mollifier_hess = ee_mollifier_derivatives(
            x0, x1, x2, x3, eps_x
        )
        barrier = ipc_barrier(d2, d02, contact_stiffness)
        barrier_grad_vec = barrier_grad * grad
        hess = mollifier * (barrier_hess * wp.outer(grad, grad) + barrier_grad * hess)
        hess += barrier * mollifier_hess
        hess += wp.outer(mollifier_grad, barrier_grad_vec)
        hess += wp.outer(barrier_grad_vec, mollifier_grad)
        hess = project_spd_12(hess)
        # self.b stores force; compute_rhs later converts it to an energy gradient.
        grad = -(mollifier * barrier_grad_vec + barrier * mollifier_grad)

        for ii in range(4):
            gii = vec3(grad[ii * 3 + 0], grad[ii * 3 + 1], grad[ii * 3 + 2])
            wp.atomic_add(b, c.a1a2b1b2[ii], gii)
            for jj in range(4): 
                triplets.rows[i * 16 + ii * 4 + jj] = c.a1a2b1b2[ii]
                triplets.cols[i * 16 + ii * 4 + jj] = c.a1a2b1b2[jj]
                block = mat33(
                    hess[ii * 3 + 0, jj * 3 + 0], hess[ii * 3 + 0, jj * 3 + 1], hess[ii * 3 + 0, jj * 3 + 2],
                    hess[ii * 3 + 1, jj * 3 + 0], hess[ii * 3 + 1, jj * 3 + 1], hess[ii * 3 + 1, jj * 3 + 2],
                    hess[ii * 3 + 2, jj * 3 + 0], hess[ii * 3 + 2, jj * 3 + 1], hess[ii * 3 + 2, jj * 3 + 2]
                )
                triplets.vals[i * 16 + ii * 4 + jj] = block

@wp.kernel
def contact_hessian_pt(states: NewtonState, soup: Soup, contacts: wp.array(dtype = XConstraint), triplets: Triplets, b: wp.array(dtype = vec3), offset: int):
    i = wp.tid()
    c = contacts[i]
    
    dist, v0, v1 = fetch_dist_v0v1_pt(states, soup, c)
    
    if dist < c.l0:
        i0 = c.a1a2b1b2[0]
        i1 = c.a1a2b1b2[1]
        i2 = c.a1a2b1b2[2]
        i3 = c.a1a2b1b2[3]
        x0 = states.x[i0]
        x1 = states.x[i1]
        x2 = states.x[i2]
        x3 = states.x[i3]
        grad, hess = point_triangle_distance_gradient_hessian(x0, x1, x2, x3)
        d2 = dist * dist
        d02 = c.l0 * c.l0
        barrier_grad = ipc_barrier_derivative(d2, d02, contact_stiffness)
        barrier_hess = ipc_barrier_derivative2(d2, d02, contact_stiffness)
        hess = barrier_hess * wp.outer(grad, grad) + barrier_grad * hess
        hess = project_spd_12(hess)
        # self.b stores force; compute_rhs later converts it to an energy gradient.
        grad *= -barrier_grad

        for ii in range(4):
            gii = vec3(grad[ii * 3 + 0], grad[ii * 3 + 1], grad[ii * 3 + 2])
            wp.atomic_add(b, c.a1a2b1b2[ii], gii)
            for jj in range(4): 
                idx = (offset + i) * 16 + ii * 4 + jj
                triplets.rows[idx] = c.a1a2b1b2[ii]
                triplets.cols[idx] = c.a1a2b1b2[jj]
                block = mat33(
                    hess[ii * 3 + 0, jj * 3 + 0], hess[ii * 3 + 0, jj * 3 + 1], hess[ii * 3 + 0, jj * 3 + 2],
                    hess[ii * 3 + 1, jj * 3 + 0], hess[ii * 3 + 1, jj * 3 + 1], hess[ii * 3 + 1, jj * 3 + 2],
                    hess[ii * 3 + 2, jj * 3 + 0], hess[ii * 3 + 2, jj * 3 + 1], hess[ii * 3 + 2, jj * 3 + 2]
                )
                triplets.vals[idx] = block
                

class RodComplexBC(RodBCBase, RodComplex, ContactSolverBase):
    def __init__(self, h, meshes = [], transforms = [], static_meshes:StaticScene = None):
        self.meshes_filename = meshes 
        self.transforms = transforms
        RodBCBase.__init__(self, h)
        self.soup.x_transformed = self.states.x
        ContactSolverBase.__init__(self)
        self._ldlt_solver = None
        self._ldlt_collision_pattern = None
        self._ldlt_offsets = None
        self._ldlt_columns = None
        self._ldlt_values = None
        self._ldlt_scalar_nnz = 0
        self.ldlt_symbolic_factorizations = 0
        self.ldlt_refactorizations = 0

    def on_checkpoint_loaded(self):
        # The restored geometry may have a different active contact pattern.
        # Force one fresh symbolic analysis; subsequent Newton iterations can
        # resume the normal collision-pattern reuse path.
        self._ldlt_collision_pattern = None

    def line_search_upper_bound(self):
        return self.collision_free_step(self.states.dx)
    
    def compute_A(self):
        with self.profile_timer("compute A"):
            with self.profile_timer("detect collision"):
                self.detect_collision()
            with self.profile_timer("compute elastic hessian"):
                super().compute_A()
            with self.profile_timer("compute contact hessian"):
                self.collision_triplets = Triplets()
                nnz = (self.n_contacts + self.n_contacts_pt) * 4 * 4
                self.collision_triplets.rows = wp.zeros((nnz,), dtype = int)
                self.collision_triplets.cols = wp.zeros_like(self.collision_triplets.rows)
                self.collision_triplets.vals = wp.zeros((nnz,), dtype = mat33)
                wp.launch(contact_hessian_ee, dim = (self.n_contacts, ), inputs = [self.states, self.soup, self.contacts_new.list, self.collision_triplets, self.b])
                wp.launch(contact_hessian_pt, dim = (self.n_contacts_pt, ), inputs = [self.states, self.soup, self.contacts_pt.list, self.collision_triplets, self.b, self.n_contacts])

                self.collision_hessian = bsr_from_triplets(
                    self.n_nodes,
                    self.n_nodes,
                    self.collision_triplets.rows,
                    self.collision_triplets.cols,
                    self.collision_triplets.vals,
                )
                bsr_axpy(self.collision_hessian, self.K_sparse, self.h * self.h, 1.0)

    def compute_collision_energy(self):
        self.detect_collision()
        e = wp.zeros((1,), dtype = scalar)
        wp.launch(contact_energy_ee, dim = (self.n_contacts, ), inputs = [self.states, self.soup, self.contacts_new.list, e])
        wp.launch(contact_energy_pt, dim = (self.n_contacts_pt, ), inputs = [self.states, self.soup, self.contacts_pt.list, e])
        with self.profile_timer("energy host transfer"):
            energy_host = e.numpy()
        return energy_host[0] * self.h * self.h

    def solve_ldlt(self):
        # bsr_from_triplets canonicalizes the contact triplets, so this pattern
        # comparison is insensitive to the atomic order used to discover the
        # same contact set.
        with self.profile_timer("collision pattern host transfer"):
            collision_nnz = self.collision_hessian.nnz_sync()
            collision_offsets = self.collision_hessian.offsets.numpy().copy()
            collision_columns = self.collision_hessian.columns.numpy()[:collision_nnz].copy()

        previous = self._ldlt_collision_pattern
        same_collision_pattern = (
            previous is not None
            and np.array_equal(collision_offsets, previous[0])
            and np.array_equal(collision_columns, previous[1])
        )

        n = self.A.shape[0]
        block_nnz = self.A.nnz_sync()
        scalar_nnz = block_nnz * 9
        reuse_symbolic = (
            self._ldlt_solver is not None
            and same_collision_pattern
            and scalar_nnz == self._ldlt_scalar_nnz
        )

        if not reuse_symbolic:
            self._ldlt_offsets = wp.empty(n + 1, dtype=int, device=self.A.device)
            self._ldlt_columns = wp.empty(scalar_nnz, dtype=int, device=self.A.device)
            self._ldlt_values = wp.empty(scalar_nnz, dtype=scalar, device=self.A.device)

        wp.launch(
            bsr_to_scalar_csr,
            dim=n + 1,
            inputs=[
                self.A.offsets,
                self.A.columns,
                self.A.values,
                self._ldlt_offsets,
                self._ldlt_columns,
                self._ldlt_values,
                self.A.nrow,
            ],
            device=self.A.device,
        )

        if reuse_symbolic:
            self._ldlt_solver.refactorize(self._ldlt_values.ptr)
            self.ldlt_refactorizations += 1
        else:
            self._ldlt_solver = CUSolverDevice(
                self._ldlt_offsets.ptr,
                self._ldlt_columns.ptr,
                self._ldlt_values.ptr,
                n,
                scalar_nnz,
            )
            self._ldlt_solver.analyze_pattern()
            self._ldlt_solver.factorize()
            self._ldlt_scalar_nnz = scalar_nnz
            self.ldlt_symbolic_factorizations += 1

        self._ldlt_collision_pattern = (collision_offsets, collision_columns)
        self._ldlt_solver.solve(self.b.ptr, self.states.dx.ptr)

@wp.kernel
def contact_energy_ee(states: NewtonState, soup: Soup, contacts: wp.array(dtype = XConstraint), e: wp.array(dtype = scalar)):
    i = wp.tid()
    c = contacts[i]
    
    dist, v0, v1 = fetch_dist_v0v1(states, soup, c)
    
    if dist < c.l0:
        eps_x = ee_mollifier_threshold_rest(soup, c)
        mollifier = ee_mollifier_value(
            soup.x_transformed[c.a1a2b1b2[0]], soup.x_transformed[c.a1a2b1b2[1]],
            soup.x_transformed[c.a1a2b1b2[2]], soup.x_transformed[c.a1a2b1b2[3]], eps_x
        )
        energy = mollifier * ipc_barrier(dist * dist, c.l0 * c.l0, contact_stiffness)
        wp.atomic_add(e, 0, energy)
        
@wp.kernel
def contact_energy_pt(states: NewtonState, soup: Soup, contacts: wp.array(dtype = XConstraint), e: wp.array(dtype = scalar)):
    i = wp.tid()
    c = contacts[i]
    dist, v0, v1 = fetch_dist_v0v1_pt(states, soup, c)

    if dist < c.l0:
        energy = ipc_barrier(dist * dist, c.l0 * c.l0, contact_stiffness)
        wp.atomic_add(e, 0, energy)
