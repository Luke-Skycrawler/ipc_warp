import numpy as np
import igl
import warp as wp
from pathlib import Path
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
h = 16e-3
rho = 1e3
omega = 3.0
boundary_v = 1.0

quasi_static = False
twist = True
dirichlet_boundary = True
attachment_stiffness = scalar(1e11)
contact_stiffness = scalar(1e9)
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
def moving_boundary(x: vec3):
    return x[0] < -0.5 + eps# or x[0] > 0.5 - eps
    

@wp.kernel
def initialize_attachment_triplets(
    geo: FEMMesh,
    triplets: Triplets,
    stiffness: scalar,
):
    i = wp.tid()
    triplets.rows[i] = i
    triplets.cols[i] = i
    if geo.fixed[i] != 0:
        triplets.vals[i] = wp.identity(3, dtype=scalar) * stiffness
    else:
        triplets.vals[i] = mat33(scalar(0.0))

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
def compute_inertia(geo: FEMMesh, state: NewtonState, M: wp.array(dtype = scalar), inert: wp.array(dtype = scalar), h: scalar):
    i = wp.tid()
    de = scalar(0.0)
    if geo.fixed[i] == 0:
        dx = x_minus_tilde(state, h, i)
        de = wp.length_sq(dx) * M[i] * scalar(0.5)
    wp.atomic_add(inert, 0, de)


@wp.kernel
def subtract_gradient(
    gradient: wp.array(dtype=vec3),
    base_gradient: wp.array(dtype=vec3),
    correction: wp.array(dtype=vec3),
):
    i = wp.tid()
    correction[i] = gradient[i] - base_gradient[i]


@wp.kernel
def compute_attachment_energy(
    state: NewtonState,
    geo: FEMMesh,
    reference_x: wp.array(dtype=vec3),
    attachment_residual: wp.array(dtype=vec3),
    stiffness: scalar,
    energy: wp.array(dtype=scalar),
):
    i = wp.tid()
    if geo.fixed[i] != 0:
        # The target is fixed during a Newton iteration.  The residual at the
        # reference position is x_ref - x_target, so this evaluates the full
        # quadratic penalty at the trial position rather than only its frozen
        # linearization.
        residual = attachment_residual[i] + state.x[i] - reference_x[i]
        wp.atomic_add(energy, 0, scalar(0.5) * stiffness * wp.dot(residual, residual))

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
        self.h = h
        self.define_M()
        self.states = NewtonState()
        self.states.x = wp.zeros_like(self.xcs)
        self.states.x0 = wp.zeros_like(self.xcs)
        self.states.dx = wp.zeros_like(self.xcs)
        self.states.xdot = wp.zeros_like(self.xcs)
        self.states.Psi = wp.zeros((self.n_tets,), dtype = scalar)

        self.comp_x = wp.zeros_like(self.states.dx)
        # Residual x - x_target used to assemble the attachment force.  Keep
        # it separate from comp_x, which is overwritten with the force/RHS.
        self.attachment_residual = wp.zeros_like(self.states.dx)
        # Retain the assembled base gradient for diagnostics and contact/RHS
        # bookkeeping.  Attachment compensation is represented by the exact
        # quadratic energy below.
        self.energy_gradient_correction = wp.zeros_like(self.states.dx)
        self.line_search_reference_x = wp.zeros_like(self.states.x)
        self.set_fixed_boundary()
        self.initialize_attachment_matrix()
        
        self.reset()
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

    def save_checkpoint(self, filename):
        """Store the complete time-integration state needed to rerun a frame."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            x=self.states.x.numpy(),
            x0=self.states.x0.numpy(),
            xdot=self.states.xdot.numpy(),
            theta=np.asarray(self.theta, dtype=np.float64),
            frame=np.asarray(self.frame, dtype=np.int64),
            timestep=np.asarray(self.h, dtype=np.float64),
            n_nodes=np.asarray(self.n_nodes, dtype=np.int64),
            n_tets=np.asarray(self.n_tets, dtype=np.int64),
            rest_sum=np.asarray(np.sum(self.xcs.numpy()), dtype=np.float64),
        )
        print(f"checkpoint saved: {path}")

    def load_checkpoint(self, filename):
        """Restore a checkpoint; contacts and matrices are rebuilt on demand."""
        path = Path(filename)
        with np.load(path, allow_pickle=False) as data:
            x = np.asarray(data["x"])
            x0 = np.asarray(data["x0"])
            xdot = np.asarray(data["xdot"])
            expected_shape = (self.n_nodes, 3)
            if x.shape != expected_shape or x0.shape != expected_shape or xdot.shape != expected_shape:
                raise ValueError(
                    f"Checkpoint state shape does not match this mesh: "
                    f"{x.shape}, expected {expected_shape}"
                )
            if int(data["n_tets"]) != self.n_tets:
                raise ValueError("Checkpoint tetrahedron count does not match this mesh")
            if not np.isclose(float(data["timestep"]), self.h):
                raise ValueError("Checkpoint timestep does not match this simulation")
            if not np.isclose(float(data["rest_sum"]), np.sum(self.xcs.numpy())):
                raise ValueError("Checkpoint rest geometry does not match this simulation")
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(x0)) and np.all(np.isfinite(xdot))):
                raise ValueError("Checkpoint contains a non-finite state")

            self.states.x.assign(x)
            self.states.x0.assign(x0)
            self.states.xdot.assign(xdot)
            self.states.dx.zero_()
            self.states.Psi.zero_()
            self.theta = float(data["theta"])
            self.frame = int(data["frame"])

        self.on_checkpoint_loaded()
        print(f"checkpoint loaded: {path} (frame {self.frame})")

    def on_checkpoint_loaded(self):
        """Hook for invalidating state-dependent caches after a restore."""
        pass

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

    def initialize_attachment_matrix(self):
        """Build the fixed-node diagonal matrix once after the mask is set."""
        self.attachment_triplets = Triplets()
        self.attachment_triplets.rows = wp.zeros(self.n_nodes, dtype=int)
        self.attachment_triplets.cols = wp.zeros(self.n_nodes, dtype=int)
        self.attachment_triplets.vals = wp.zeros(self.n_nodes, dtype=mat33)

        # Previously the attachment was part of K and therefore received the
        # same h^2 scaling as the elastic Hessian in dynamic mode.
        scale = attachment_stiffness
        if not quasi_static:
            scale *= scalar(self.h * self.h)
        self.attachment_energy_stiffness = scale
        wp.launch(
            initialize_attachment_triplets,
            self.n_nodes,
            inputs=[self.geo, self.attachment_triplets, scale],
        )
        self.attachment_matrix = bsr_from_triplets(
            self.n_nodes,
            self.n_nodes,
            self.attachment_triplets.rows,
            self.attachment_triplets.cols,
            self.attachment_triplets.vals,
        )


    def step(self):
        newton_iter = True
        n_iter = 0
        max_iter = 100
        # while n_iter < max_iter:
        while newton_iter and n_iter < max_iter:
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
                norm_dx = np.max(np.abs(dxnp))
                newton_iter = norm_dx > vel_tol * self.h
                print(f"    norm = {norm_dx:1.2e}, iter = {n_iter}")
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
        bsr_axpy(self.attachment_matrix, self.A, 1.0, 1.0)
        # self.A = self.M_sparse 

    def compute_K(self):
        self.triplets.vals.zero_()
        self.b.zero_()
        wp.launch(tet_kernel_sparse, (self.n_tets * 4 * 4,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.triplets, self.b]) 
        # now self.b has the elastic forces

        bsr_set_zero(self.K_sparse)
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals)        
        
    def compute_rhs(self):
        with self.profile_timer("compute rhs"):
            wp.launch(compute_rhs, (self.n_nodes, ), inputs = [self.states, self.h, self.M, self.b])
            wp.copy(self.energy_gradient_correction, self.b)
            self.set_bc_fixed_grad()
            self.compute_compensation()
            wp.launch(
                subtract_gradient,
                self.n_nodes,
                inputs=[self.b, self.energy_gradient_correction, self.energy_gradient_correction],
            )

    def compute_compensation(self):
        """Apply the prescribed-boundary displacement to the Newton RHS.

        Override this hook for a different moving boundary. An override should
        leave ``comp_x`` holding the compensation force used by the line-search
        energy, as the default implementation does.
        """
        self.attachment_residual.zero_()
        if not twist:
            self.comp_x.zero_()
            return
        wp.launch(
            compute_twist_compensation,
            self.n_nodes,
            inputs=[self.states, self.geo, self.theta, self.attachment_residual],
        )
        # print(f"    compensation = {np.linalg.norm(self.comp_x.numpy())}")
        # The prescribed target is an attachment penalty, so its force must be
        # generated by the attachment matrix itself (not the full Newton
        # matrix).  This keeps the Newton RHS consistent with the energy used
        # by the line search.
        tmp = bsr_mv(self.attachment_matrix, self.attachment_residual)
        array_axpy(tmp, self.b, 1.0, 1.0)
        wp.copy(self.comp_x, tmp)

    def set_bc_fixed_grad(self):
        # The new attachment formulation is a penalty, not hard elimination:
        # retain the physical gradient and add A(x - x_target) below.
        pass

    def solve_ldlt(self): 
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

    def solve(self):
        with self.profile_timer("solve"):
            if solver_choice == "cg":
                self.states.dx.zero_()
                # bicgstab(self.A, self.b, self.states.dx, 1e-6, maxiter = 100)
                cg(self.A, self.b, self.states.dx, 1e-4, use_cuda_graph = True)
            elif solver_choice == "ldlt":
                self.solve_ldlt()

    def line_search_fixed(self):
        alpha = self.line_search_upper_bound()
        wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])
        return alpha
        
    def line_search(self):
        with self.profile_timer("line search"):
            x_tmp = wp.clone(self.states.x)
            wp.copy(self.line_search_reference_x, x_tmp)
            E0 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
            E0 += self.compute_bc_energy()
            upper_bound = self.line_search_upper_bound()
            alpha = upper_bound
            E1 = np.inf
            accepted = False
            # A contact barrier can make the useful step many orders of
            # magnitude smaller than the CCD upper bound.  A fixed 1e-2
            # cutoff incorrectly rejects valid descent directions precisely
            # when contact becomes stiff.
            for _ in range(64):
                wp.copy(self.states.x, x_tmp)
                wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])
                E1 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
                E1 += self.compute_bc_energy()

                if np.isfinite(E1) and E1 < E0:
                    accepted = True
                    break
                alpha *= 0.5
                if alpha <= np.finfo(np.float64).eps * max(1.0, upper_bound):
                    break

            if not accepted:
                wp.copy(self.states.x, x_tmp)
                alpha = 0.0

            print(f"    alpha = {alpha:1.2e}, E0 = {E0:1.2e}, E1 = {E1:1.2e}, upper bound = {upper_bound:1.2e}")
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
        wp.launch(compute_inertia, (self.n_nodes, ), inputs = [self.geo, self.states, self.M, inert, self.h])
        with self.profile_timer("energy host transfer"):
            inert_host = inert.numpy()
        return inert_host[0]

    def compute_bc_energy(self):
        """Exact quadratic energy of the prescribed fixed-node attachments."""
        energy = wp.zeros((1,), dtype=scalar)
        wp.launch(
            compute_attachment_energy,
            self.n_nodes,
            inputs=[
                self.states,
                self.geo,
                self.line_search_reference_x,
                self.attachment_residual,
                self.attachment_energy_stiffness,
                energy,
            ],
        )
        with self.profile_timer("energy host transfer"):
            energy_host = energy.numpy()
        return energy_host[0]

class RodBC(RodBCBase, Rod):
    def __init__(self, h, filename = default_tobj):
        self.filename = filename
        super().__init__(h)
