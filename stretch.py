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
from geometry.static_scene import StaticScene
from warp.fem.linalg import array_axpy
from scalar_types import *
from contact import ContactSolverBase, XConstraint, fetch_dist_v0v1, fetch_dist_v0v1_pt

from ipctkwp.distance.edge_edge import x_to_grad_psd_hess_ee
from ipctkwp.distance.point_triangle import x_to_grad_psd_hess_pt
from ipctkwp.distance.point_edge import point_edge_distance_gradient_hessian

from fem.geometry import Soup

eps = 3e-4
h = 1e-2
rho = 1e3
omega = 3.0
boundary_v = 1.0

quasi_static = False
twist = True
attachment_stiffness = scalar(1e8)

contact_stiffness = scalar(1e5)
wp.config.max_unroll = 1
wp.config.enable_backward = False

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
    else:
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

    # if not should_fix(geo.xcs[i]):
    if True:
        dx = x_minus_tilde(state, h, i)
        de = wp.length_sq(dx) * M[i] * scalar(0.5) + wp.dot(comp_x[i], state.x[i])
        # de = wp.dot(comp_x[i], state.x[i])
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
        max_iter = 8
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
            norm_dx = np.linalg.norm(dxnp)
            newton_iter = norm_dx > 1e-4 and n_iter < max_iter
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
            self.states.dx.zero_()
            # bicgstab(self.A, self.b, self.states.dx, 1e-6, maxiter = 100)
            cg(self.A, self.b, self.states.dx, 1e-6, use_cuda_graph = True)
    
    def line_search_fixed(self):
        alpha = 1.0
        wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])
        return alpha
        
    def line_search(self):
        # if twist: 
        #     return self.line_search_fixed() 
        # FIXME: not converged
        x_tmp = wp.clone(self.states.x)
        E0 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
        alpha = 1.0
        while True:
            wp.copy(self.states.x, x_tmp)
            wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])
            E1 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
            
            if E1 < E0:
                break
            if alpha < 1e-3:
                wp.copy(self.states.x, x_tmp)
                alpha = 0.0
                break
            alpha *= 0.5

        print(f"alpha = {alpha}, E0 = {E0}, E1 = {E1}")
        return alpha

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
        grad, hess = x_to_grad_psd_hess_ee(x0, x1, x2, x3)

        grad *= contact_stiffness
        hess *= contact_stiffness

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
        grad, hess = x_to_grad_psd_hess_pt(x0, x1, x2, x3)

        grad *= contact_stiffness
        hess *= contact_stiffness

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
    
    def compute_A(self):
        self.detect_collision()
        
        triplets = Triplets()
        nnz = (self.n_contacts + self.n_contacts_pt) * 4 * 4
        triplets.rows = wp.zeros((nnz,), dtype = int)
        triplets.cols = wp.zeros_like(triplets.rows)
        triplets.vals = wp.zeros((nnz,), dtype = mat33)
        wp.launch(contact_hessian_ee, dim = (self.n_contacts, ), inputs = [self.states, self.soup, self.contacts_new.list, triplets, self.b])
        
        wp.launch(contact_hessian_pt, dim = (self.n_contacts_pt, ), inputs = [self.states, self.soup, self.contacts_pt.list, triplets, self.b, self.n_contacts])

        collision_hess = bsr_from_triplets(self.n_nodes, self.n_nodes, triplets.rows, triplets.cols, triplets.vals)
        
        super().compute_A()
        bsr_axpy(collision_hess, self.K_sparse, h * h, 1.0)
        
def drape():
    # rod = RodBC(h, "assets/elephant.mesh")
    # rod = RodBC(h)
    rod = RodComplexBC(h, meshes = ["assets/bar2.tobj"], transforms = [np.eye(4)])
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.set_ground_plane_mode("none")
    ps.show()


if __name__ == "__main__":
    ps.init()
    wp.init()
    drape()
    