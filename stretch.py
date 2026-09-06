import polyscope as ps
import polyscope.imgui as gui 
import numpy as np
import warp as wp 
from fem.interface import Rod, default_tobj, RodComplex
from fem.params import NewtonState
import igl
from warp.sparse import *
from fem.params import FEMMesh, mu, lam, gravity, gravity_np
from fem.fem import tet_kernel, tet_kernel_sparse, Triplets, psi
from warp.optim.linear import bicgstab, cg
from dxslv import CUSolverDevice
from geometry.static_scene import StaticScene
from warp.fem.linalg import array_axpy
from scalar_types import *
from contact import ContactSolverBase, XConstraint, fetch_dist_v0v1, fetch_dist_v0v1_pt, closest_point_triangle

from ipctkwp.distance.edge_edge import x_to_grad_psd_hess_ee
from ipctkwp.distance.point_triangle import x_to_grad_psd_hess_pt

from fem.geometry import Soup

vel_tol = 5e-2
eps = 3e-4
h = 8e-3
rho = 1e3
omega = 3.0
boundary_v = 1.0

quasi_static = False
twist = True
dirichlet_boundary = True
attachment_stiffness = scalar(1e7)

contact_stiffness = scalar(1e8)
solver_choice = "cg"
wp.config.max_unroll = 1
wp.config.enable_backward = False


@wp.kernel
def bsr_to_scalar_csr(
    block_offsets: wp.array(dtype=int),
    block_columns: wp.array(dtype=int),
    block_values: wp.array(dtype=mat33),
    scalar_offsets: wp.array(dtype=int),
    scalar_columns: wp.array(dtype=int),
    scalar_values: wp.array(dtype=scalar),
    n_block_rows: int,
):
    scalar_row = wp.tid()
    if scalar_row == n_block_rows * 3:
        scalar_offsets[scalar_row] = block_offsets[n_block_rows] * 9
        return

    block_row = scalar_row // 3
    row_in_block = scalar_row % 3
    block_begin = block_offsets[block_row]
    block_end = block_offsets[block_row + 1]
    blocks_in_row = block_end - block_begin
    scalar_begin = block_begin * 9 + row_in_block * blocks_in_row * 3
    scalar_offsets[scalar_row] = scalar_begin

    for block_index in range(block_begin, block_end):
        block_column = block_columns[block_index]
        output = scalar_begin + (block_index - block_begin) * 3
        for column_in_block in range(3):
            scalar_columns[output + column_in_block] = block_column * 3 + column_in_block
            scalar_values[output + column_in_block] = block_values[block_index][
                row_in_block, column_in_block
            ]

@wp.kernel
def set_M_diag(d: wp.array(dtype = scalar), M: wp.array(dtype = mat33)):  
    i =  wp.tid()
    mii = wp.identity(3, dtype = scalar)
    mii *= d[i]
    M[i] = mii


@wp.func
def x_minus_tilde(state: NewtonState, h: scalar, i: int) -> vec3:
    ret = vec3()
    if quasi_static:
        ret = gravity
    else: 
        return state.x[i] - (state.x0[i] + h * state.xdot[i] + h * h * gravity)
    return ret

@wp.kernel
def compute_rhs(state: NewtonState, h: scalar, M: wp.array(dtype = scalar), b: wp.array(dtype = vec3)):
    '''
    before execution, b[i] stores the elastic forces 
    turns rhs into df/dx, where f is the argmin function Vh^2 + 0.5 M(x - x+tilde) ^ 2 
    '''
    i = wp.tid()
    if quasi_static: 
        b[i] = -b[i] - M[i] * x_minus_tilde(state, h, i)
    else: 
        b[i] = -b[i] * h * h + M[i] * x_minus_tilde(state, h, i)


@wp.func
def should_fix(x: vec3): 
    ret = False
    if wp.static(twist): 
        ret = x[0] < -0.5 + eps or x[0] > 0.5 - eps
    elif wp.static(dirichlet_boundary):
        ret = x[0] < -0.5 + eps 
    return ret

    # v0 = vec3(-56.273449910216, 94.689259419722, -19.03583034376)
    # return wp.length_sq(x - v0) < eps
@wp.func
def moving_boundary(x: vec3):
    return x[0] < -0.5 + eps# or x[0] > 0.5 - eps
    

@wp.kernel
def set_b_fixed(geo: FEMMesh,b: wp.array(dtype = vec3)):
    i = wp.tid()
    # set fixed points rhs to 0
    if should_fix(geo.xcs[i]): 
        b[i] = vec3()

@wp.kernel
def set_K_fixed(geo: FEMMesh, triplets: Triplets):
    eij = wp.tid()
    e = eij // 16
    ii = (eij // 4) % 4
    jj = eij % 4

    i = geo.T[e, ii]
    j = geo.T[e, jj]
    
    if should_fix(geo.xcs[i]) or should_fix(geo.xcs[j]):        
        if ii == jj:
            triplets.vals[eij] += wp.identity(3, dtype = scalar) * attachment_stiffness
        # else:
        #     triplets.vals[eij] = mat33(0.0)

@wp.kernel
def add_dx(state: NewtonState, alpha :scalar):
    i = wp.tid()
    state.x[i] -= state.dx[i] * alpha

@wp.kernel
def update_x0_xdot(state: NewtonState, h: scalar):
    i = wp.tid()
    state.xdot[i] = (state.x[i] - state.x0[i]) / h
    state.x0[i] = state.x[i]

@wp.kernel
def compute_Psi(x: wp.array(dtype = vec3), geo: FEMMesh, Bm: wp.array(dtype = mat33), W: wp.array(dtype = scalar), Psi: wp.array(dtype = scalar)):
    e = wp.tid()
    t0 = x[geo.T[e, 0]]
    t1 = x[geo.T[e, 1]]
    t2 = x[geo.T[e, 2]]
    t3 = x[geo.T[e, 3]]
    
    Ds = wp.matrix_from_cols(t0 - t3, t1 - t3, t2 - t3)
    
    F = Ds @ Bm[e]
    psie = psi(F)
    # wp.atomic_add(Psi, 0, W[e] * psi)
    Psi[e] = W[e] * psie

@wp.kernel
def compute_inertia(geo: FEMMesh, state: NewtonState, M: wp.array(dtype = scalar), inert: wp.array(dtype = scalar), comp_x: wp.array(dtype = vec3), h: scalar):
    i = wp.tid()
    de = scalar(0.0)
    if not should_fix(geo.xcs[i]):
    # if True:
        dx = x_minus_tilde(state, h, i)
        de = wp.length_sq(dx) * M[i] * scalar(0.5)
        # de = wp.dot(comp_x[i], state.x[i])
    else: 
        de = wp.dot(comp_x[i], state.x[i])
    wp.atomic_add(inert, 0, de)

@wp.kernel
def compute_compensation(state: NewtonState, geo: FEMMesh, theta: scalar, comp_x: wp.array(dtype = vec3)):
    i = wp.tid()
    z = scalar(0.0)
    xi = state.x[i]
    c = wp.cos(theta * scalar(omega))
    s = wp.sin(theta * scalar(omega))
    rot = mat22(
        c, s,
        -s, c
    )
    x_rst= geo.xcs[i]
    if moving_boundary(x_rst):
    # if False:
        yz_rst = vec2(x_rst[1], x_rst[2])
        yz = rot @ yz_rst
        if x_rst[0] < 0.0:
            yz = wp.transpose(rot) @ yz_rst
        target = vec3(x_rst[0], yz[0], yz[1]) # + vec3(z, theta, z)
        # target = x_rst + vec3(z, theta, z)
        target += vec3(scalar(boundary_v) * theta, z, z)
        comp_x[i] = xi - target
    else:
        comp_x[i] = vec3(z)

class PSViewer:
    def __init__(self, rod, static_mesh: StaticScene = None):
        self.V = rod.xcs.numpy()
        self.F = rod.F

        self.ps_mesh = ps.register_surface_mesh("rod", self.V, self.F)
        self.frame = 0
        self.rod = rod
        self.ui_pause = True
        self.animate = False
        
        self.end_frame = 5000
        self.capture_interval = 1
        if static_mesh is not None:
            Vs = static_mesh.xcs.numpy()
            Fs = static_mesh.indices.numpy().reshape((-1, 3))
            self.static_mesh = ps.register_surface_mesh("static", Vs, Fs)
            # self.static_mesh.add_vector_quantity("normal", static_mesh.N, defined_on="faces")
            if static_mesh.has_medials:
                self.static_medials = ps.register_curve_network("static medial", static_mesh.V_medial, static_mesh.E_medial)
                self.static_spheres = ps.register_point_cloud("static spheres", static_mesh.V_medial)
                self.static_spheres.add_scalar_quantity("radius", static_mesh.R)
                self.static_spheres.set_point_radius_quantity("radius", autoscale=False)

    def save(self):
        # ps.screenshot(f"output/{self.frame:04d}.jpg")
        self.frame = self.rod.frame
        if self.frame % 4 == 0:
            igl.write_obj(f"output/obj/{self.frame:04d}.obj", self.V, self.F)
        if hasattr(self.rod, "save_states"):
            self.rod.save_states()

    def callback(self):
        changed, self.ui_pause = gui.Checkbox("Pause", self.ui_pause)
        self.animate = gui.Button("Step") or not self.ui_pause
        if gui.Button("Reset"):
            self.rod.reset()
            self.frame = self.rod.frame
            self.ui_pause = True
            self.animate = True

        if gui.Button("Save"):
            np.save(f"output/x_{self.frame}.npy", self.V)
            print(f"output/x_{self.frame}.npy saved")
        if self.animate: 
            self.rod.step()
            self.V = self.rod.states.x.numpy()
            self.ps_mesh.update_vertex_positions(self.V)
            self.frame = self.rod.frame
            
            print("frame = ", self.frame)

            # if self.frame % self.capture_interval == 0:
            #     self.save()
        if self.frame >= self.end_frame:
            print(f"end frame = {self.frame} reached, exiting")
            quit()

        
class RodBCBase:
    '''
    fem with boundary condition and dynamic attributes
    '''

    def  __init__(self, h):
        super().__init__()
        self.define_M()
        self.states = NewtonState()
        self.states.x = wp.zeros_like(self.xcs)
        self.states.x0 = wp.zeros_like(self.xcs)
        self.states.dx = wp.zeros_like(self.xcs)
        self.states.xdot = wp.zeros_like(self.xcs)
        self.states.Psi = wp.zeros((self.n_tets,), dtype = scalar)

        self.comp_x = wp.zeros_like(self.states.dx)
        
        self.reset()
        self.h = h
        print(f"timestep set to {h}")
        
    def reset(self):
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)
        self.states.xdot.zero_()
        

        self.theta = 0.0
        self.frame = 0

    def define_M(self):
        V = self.xcs.numpy()
        T = self.T.numpy()
        # self.M is a vector composed of diagonal elements 
        self.Mnp = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
        self.M = wp.zeros((self.n_nodes,), dtype = scalar)
        self.M.assign(self.Mnp * rho)
        self.M.fill_(1.0)

        self.M_sparse = bsr_zeros(self.n_nodes, self.n_nodes, mat33)
        M_diag = wp.zeros((self.n_nodes,), dtype = mat33)
        wp.launch(set_M_diag, (self.n_nodes,), inputs = [self.M, M_diag])
        bsr_set_diag(self.M_sparse, M_diag)


    def step(self):
        newton_iter = True
        n_iter = 0
        max_iter = 100
        # while n_iter < max_iter:
        while newton_iter:
            self.compute_A()
            self.compute_rhs()

            self.solve()
            # wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, 1.0])
            
            
            # line search stuff, not converged yet
            alpha = self.line_search()
            if alpha == 0.0:
                break

            dxnp = self.states.dx.numpy()
            norm_dx = np.max(dxnp)
            newton_iter = norm_dx > vel_tol * h and n_iter < max_iter
            print(f"norm = {np.linalg.norm(dxnp)}, {n_iter}")
            n_iter += 1
        self.update_x0_xdot()
        self.theta += self.h
        self.frame += 1

    def update_x0_xdot(self):
        wp.launch(update_x0_xdot, dim = (self.n_nodes,), inputs = [self.states, self.h])

    def compute_A(self):

        self.compute_K()

        h = self.h
        if not quasi_static:
            # A = h^2 * K + M
            bsr_axpy(self.M_sparse, self.K_sparse, 1.0, h * h)

        self.A = self.K_sparse
        # self.A = self.M_sparse 

    def compute_K(self):
        self.triplets.vals.zero_()
        self.b.zero_()
        wp.launch(tet_kernel_sparse, (self.n_tets * 4 * 4,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.triplets, self.b]) 
        # now self.b has the elastic forces

        self.set_bc_fixed_hessian()
        bsr_set_zero(self.K_sparse)
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals)        
        
    def compute_rhs(self):
        wp.launch(compute_rhs, (self.n_nodes, ), inputs = [self.states, self.h, self.M, self.b])
        self.set_bc_fixed_grad()
        if twist: 
            self.comp_x.zero_()
            wp.launch(compute_compensation, self.n_nodes, inputs= [self.states, self.geo, self.theta, self.comp_x])
            # bsr_mv(self.A, self.comp_x, self.b, beta = 1.0)
            print(f"compensation = {np.linalg.norm(self.comp_x.numpy())}")

            tmp = bsr_mv(self.A, self.comp_x)
            array_axpy(tmp, self.b, 1.0, 1.0)
            wp.copy(self.comp_x, tmp)

    def set_bc_fixed_grad(self):
        wp.launch(set_b_fixed, (self.n_nodes,), inputs = [self.geo, self.b])
    
    def set_bc_fixed_hessian(self):
        wp.launch(set_K_fixed, (self.n_tets * 4 * 4,), inputs = [self.geo, self.triplets])

    def solve(self):
        with wp.ScopedTimer("solve"):
            if solver_choice == "cg":
                self.states.dx.zero_()
                # bicgstab(self.A, self.b, self.states.dx, 1e-6, maxiter = 100)
                cg(self.A, self.b, self.states.dx, 1e-6, use_cuda_graph = True)
            elif solver_choice == "ldlt":
                n = self.A.shape[0]
                block_nnz = self.A.nnz_sync()
                scalar_nnz = block_nnz * 9
                offsets = wp.empty(n + 1, dtype=int, device=self.A.device)
                columns = wp.empty(scalar_nnz, dtype=int, device=self.A.device)
                values = wp.empty(scalar_nnz, dtype=scalar, device=self.A.device)

                wp.launch(
                    bsr_to_scalar_csr,
                    dim=n + 1,
                    inputs=[
                        self.A.offsets,
                        self.A.columns,
                        self.A.values,
                        offsets,
                        columns,
                        values,
                        self.A.nrow,
                    ],
                    device=self.A.device,
                )

                direct_solver = CUSolverDevice(
                    offsets.ptr, columns.ptr, values.ptr, n, scalar_nnz
                )
                direct_solver.analyze_pattern()
                direct_solver.factorize()
                direct_solver.solve(self.b.ptr, self.states.dx.ptr)
    def line_search_fixed(self):
        alpha = self.line_search_upper_bound()
        wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])
        return alpha
        
    def line_search(self):
        # if twist: 
        #     return self.line_search_fixed() 
        # FIXME: not converged
        x_tmp = wp.clone(self.states.x)
        E0 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
        upper_bound = self.line_search_upper_bound()
        alpha = upper_bound
        while True:
            wp.copy(self.states.x, x_tmp)
            wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])
            E1 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
            
            if E1 < E0:
                break
            if alpha < 1e-2:
                wp.copy(self.states.x, x_tmp)
                alpha = 0.0
                break
            alpha *= 0.5

        print(f"alpha = {alpha}, E0 = {E0}, E1 = {E1}, upper bound = {upper_bound}")
        return alpha

    def line_search_upper_bound(self):
        return 1.0

    def compute_collision_energy(self):
        return 0.0

    def compute_psi(self):
        h = self.h
        self.states.Psi.zero_()
        wp.launch(compute_Psi, (self.n_tets,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.states.Psi])
        return np.sum(self.states.Psi.numpy()) * h * h
    
    def compute_inertia(self):
        inert = wp.zeros((1,), dtype = scalar)
        wp.launch(compute_inertia, (self.n_nodes, ), inputs = [self.geo, self.states, self.M, inert, self.comp_x, self.h])
        return inert.numpy()[0]

class RodBC(RodBCBase, Rod):
    def __init__(self, h, filename = default_tobj):
        self.filename = filename
        super().__init__(h)

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
        scale = scalar(2.0) * contact_stiffness
        hess = scale * (wp.outer(grad, grad) + (d2 - d02) * hess)
        hess = project_spd_12(hess)
        # self.b stores force; compute_rhs later converts it to an energy gradient.
        grad *= scale * (d02 - d2)

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
        scale = scalar(2.0) * contact_stiffness
        hess = scale * (wp.outer(grad, grad) + (d2 - d02) * hess)
        hess = project_spd_12(hess)
        # self.b stores force; compute_rhs later converts it to an energy gradient.
        grad *= scale * (d02 - d2)

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

    def line_search_upper_bound(self):
        return self.collision_free_step(self.states.dx)
    
    def compute_A(self):
        self.detect_collision()
        super().compute_A()
        
        triplets = Triplets()
        nnz = (self.n_contacts + self.n_contacts_pt) * 4 * 4
        triplets.rows = wp.zeros((nnz,), dtype = int)
        triplets.cols = wp.zeros_like(triplets.rows)
        triplets.vals = wp.zeros((nnz,), dtype = mat33)
        wp.launch(contact_hessian_ee, dim = (self.n_contacts, ), inputs = [self.states, self.soup, self.contacts_new.list, triplets, self.b])
        
        wp.launch(contact_hessian_pt, dim = (self.n_contacts_pt, ), inputs = [self.states, self.soup, self.contacts_pt.list, triplets, self.b, self.n_contacts])

        collision_hess = bsr_from_triplets(self.n_nodes, self.n_nodes, triplets.rows, triplets.cols, triplets.vals)

        bsr_axpy(collision_hess, self.K_sparse, h * h, 1.0)
    def compute_collision_energy(self):
        self.detect_collision()
        e = wp.zeros((1,), dtype = scalar)
        wp.launch(contact_energy_ee, dim = (self.n_contacts, ), inputs = [self.states, self.soup, self.contacts_new.list, e])
        wp.launch(contact_energy_pt, dim = (self.n_contacts_pt, ), inputs = [self.states, self.soup, self.contacts_pt.list, e])
        return e.numpy()[0] * self.h * self.h

@wp.kernel
def contact_energy_ee(states: NewtonState, soup: Soup, contacts: wp.array(dtype = XConstraint), e: wp.array(dtype = scalar)):
    i = wp.tid()
    c = contacts[i]
    
    dist, v0, v1 = fetch_dist_v0v1(states, soup, c)
    
    if dist < c.l0:
        dl = dist * dist - c.l0 * c.l0
        energy = contact_stiffness * dl * dl
        wp.atomic_add(e, 0, energy)

@wp.kernel
def contact_energy_pt(states: NewtonState, soup: Soup, contacts: wp.array(dtype = XConstraint), e: wp.array(dtype = scalar)):
    i = wp.tid()
    c = contacts[i]
    dist, v0, v1 = fetch_dist_v0v1_pt(states, soup, c)

    if dist < c.l0:
        dl = dist * dist - c.l0 * c.l0
        energy = contact_stiffness * dl * dl
        wp.atomic_add(e, 0, energy)

def drape():
    # rod = RodBC(h, "assets/elephant.mesh")
    # rod = RodBC(h)

    # n_meshes = 2 
    # meshes = ["assets/bar2.tobj"] * n_meshes
    # # meshes = ["assets/bunny_5.tobj"] * n_meshes
    # transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    # transforms[1][:3, :3] = np.zeros((3, 3))
    # transforms[1][0, 1] = 1
    # transforms[1][1, 0] = 1
    # transforms[1][2, 2] = 1

    # for i in range(n_meshes):
    #     # transforms[i][0, 3] = i * 0.5
    #     transforms[i][1, 3] = 1.2 + i * 0.2
    #     transforms[i][2, 3] = i * 1.0

    rod = RodComplexBC(h, meshes = ["assets/bar2.tobj"], transforms = [np.eye(4)])
    # rod = RodComplexBC(h, meshes = meshes, transforms = transforms)

    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.set_ground_plane_mode("none")
    ps.show()


if __name__ == "__main__":
    ps.init()
    wp.init()
    drape()
