import numpy as np
import igl
import warp as wp
from fem.interface import Rod, default_tobj
from fem.params import NewtonState, FEMMesh, gravity, gravity_np
from fem.fem import tet_kernel, tet_kernel_sparse, Triplets, psi
from warp.sparse import *
from warp.optim.linear import bicgstab, cg
from dxslv import CUSolverDevice
from warp.fem.linalg import array_axpy
from scalar_types import *

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
contact_stiffness = scalar(1e9)
solver_choice = "ldlt"
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
def moving_boundary(x: vec3):
    return x[0] < -0.5 + eps# or x[0] > 0.5 - eps
    

@wp.kernel
def set_b_fixed(geo: FEMMesh, b: wp.array(dtype = vec3)):
    i = wp.tid()
    # set fixed points rhs to 0
    if geo.fixed[i] != 0:
        b[i] = vec3()

@wp.kernel
def set_K_fixed(geo: FEMMesh, triplets: Triplets):
    eij = wp.tid()
    e = eij // 16
    ii = (eij // 4) % 4
    jj = eij % 4

    i = geo.T[e, ii]
    j = geo.T[e, jj]
    
    if geo.fixed[i] != 0 or geo.fixed[j] != 0:
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
    if geo.fixed[i] == 0:
    # if True:
        dx = x_minus_tilde(state, h, i)
        de = wp.length_sq(dx) * M[i] * scalar(0.5)
        # de = wp.dot(comp_x[i], state.x[i])
    else: 
        de = wp.dot(comp_x[i], state.x[i])
    wp.atomic_add(inert, 0, de)

@wp.kernel
def compute_twist_compensation(state: NewtonState, geo: FEMMesh, theta: scalar, comp_x: wp.array(dtype = vec3)):
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
        target = vec3(x_rst[0], yz[0], yz[1])
        target += vec3(scalar(boundary_v) * theta, z, z)
        comp_x[i] = xi - target
    else:
        comp_x[i] = vec3(z)

        
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
        self.set_fixed_boundary()
        
        self.reset()
        self.h = h
        self.profile_enabled = False
        self.profile_print = True
        self.profile_synchronize = True
        self.profile_timings = {}
        print(f"timestep set to {h}")

    def configure_profiling(self, enabled=True, print_timings=True, synchronize=True):
        self.profile_enabled = enabled
        self.profile_print = print_timings
        self.profile_synchronize = synchronize
        self.profile_timings = {}

    def profile_timer(self, name):
        return wp.ScopedTimer(
            name,
            active=self.profile_enabled,
            print=self.profile_print,
            dict=self.profile_timings,
            synchronize=self.profile_synchronize,
        )
        
    def reset(self):
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)
        self.states.xdot.zero_()
        

        self.theta = 0.0
        self.frame = 0

    def set_fixed_boundary(self):
        """Initialize the per-node fixed mask once before simulation.

        Override this method to assign a different ``int32[n_nodes]`` mask to
        ``self.geo.fixed``.
        Boundary selection remains host-side so changing it does not alter any
        compiled Warp kernel.
        """
        x = self.xcs.numpy()
        fixed = np.zeros(self.n_nodes, dtype=np.int32)
        if twist:
            fixed[(x[:, 0] < -0.5 + eps) | (x[:, 0] > 0.5 - eps)] = 1
        elif dirichlet_boundary:
            fixed[x[:, 0] < -0.5 + eps] = 1
        self.geo.fixed.assign(fixed)

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
            with self.profile_timer("total newton iteration"):
                self.compute_A()
                self.compute_rhs()

                self.solve()
                # wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, 1.0])

                # line search stuff, not converged yet
                alpha = self.line_search()
                if alpha == 0.0:
                    break

                with self.profile_timer("dx host transfer"):
                    dxnp = self.states.dx.numpy()
                norm_dx = np.max(dxnp)
                newton_iter = norm_dx > vel_tol * self.h and n_iter < max_iter
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
        with self.profile_timer("compute rhs"):
            wp.launch(compute_rhs, (self.n_nodes, ), inputs = [self.states, self.h, self.M, self.b])
            self.set_bc_fixed_grad()
            self.compute_compensation()

    def compute_compensation(self):
        """Apply the prescribed-boundary displacement to the Newton RHS.

        Override this hook for a different moving boundary. An override should
        leave ``comp_x`` holding the compensation force used by the line-search
        energy, as the default implementation does.
        """
        self.comp_x.zero_()
        if not twist:
            return
        wp.launch(
            compute_twist_compensation,
            self.n_nodes,
            inputs=[self.states, self.geo, self.theta, self.comp_x],
        )
        print(f"compensation = {np.linalg.norm(self.comp_x.numpy())}")
        tmp = bsr_mv(self.A, self.comp_x)
        array_axpy(tmp, self.b, 1.0, 1.0)
        wp.copy(self.comp_x, tmp)

    def set_bc_fixed_grad(self):
        wp.launch(set_b_fixed, (self.n_nodes,), inputs = [self.geo, self.b])
    
    def set_bc_fixed_hessian(self):
        wp.launch(set_K_fixed, (self.n_tets * 4 * 4,), inputs = [self.geo, self.triplets])

    def solve(self):
        with self.profile_timer("solve"):
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
        with self.profile_timer("line search"):
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
        with self.profile_timer("energy host transfer"):
            psi_host = self.states.Psi.numpy()
        return np.sum(psi_host) * h * h
    
    def compute_inertia(self):
        inert = wp.zeros((1,), dtype = scalar)
        wp.launch(compute_inertia, (self.n_nodes, ), inputs = [self.geo, self.states, self.M, inert, self.comp_x, self.h])
        with self.profile_timer("energy host transfer"):
            inert_host = inert.numpy()
        return inert_host[0]

class RodBC(RodBCBase, Rod):
    def __init__(self, h, filename = default_tobj):
        self.filename = filename
        super().__init__(h)
