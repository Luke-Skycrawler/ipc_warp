"""Affine Body Dynamics on top of the existing IPC contact pipeline.

Each body has four vec3 unknowns ``[p, a0, a1, a2]`` and a surface vertex is

    x = p + r.x * a0 + r.y * a1 + r.z * a2.

Thus the affine matrix is stored by columns.  This is the transpose of the
row-proxy convention used by some ABD implementations, but represents the
same map and makes every 3x3 contact block reduce through scalar Jacobian
weights.  Collision detection, CCD, and IPC derivatives remain vertex based.
"""

from pathlib import Path

import numpy as np
import warp as wp
from warp.optim.linear import cg
from warp.sparse import bsr_axpy, bsr_from_triplets, bsr_zeros

from scalar_types import scalar, vec3, mat33
from fem.fem import Triplets
from fem.params import NewtonState
from time_integrator_base import RodBCBase
from dynamic_contacts import (
    RodComplexBC,
    contact_hessian_ee,
    contact_hessian_pt,
)


@wp.struct
class ABDState:
    q: wp.array(dtype=vec3)
    q0: wp.array(dtype=vec3)
    q_tilde: wp.array(dtype=vec3)
    qdot: wp.array(dtype=vec3)
    dx: wp.array(dtype=vec3)


@wp.kernel
def _sync_affine_vertices(q: wp.array(dtype=vec3), rest: wp.array(dtype=vec3),
                          body: wp.array(dtype=int), x: wp.array(dtype=vec3)):
    i = wp.tid()
    b = body[i] * 4
    r = rest[i]
    x[i] = q[b] + r[0] * q[b + 1] + r[1] * q[b + 2] + r[2] * q[b + 3]


@wp.kernel
def _affine_vertex_direction(dx: wp.array(dtype=vec3), rest: wp.array(dtype=vec3),
                             body: wp.array(dtype=int), vertex_dx: wp.array(dtype=vec3)):
    i = wp.tid()
    b = body[i] * 4
    r = rest[i]
    vertex_dx[i] = dx[b] + r[0] * dx[b + 1] + r[1] * dx[b + 2] + r[2] * dx[b + 3]


@wp.kernel
def _affine_add_step(q: wp.array(dtype=vec3), dx: wp.array(dtype=vec3), alpha: scalar):
    i = wp.tid()
    q[i] = q[i] - alpha * dx[i]


@wp.kernel
def _prepare_tilde(state: ABDState, h: scalar, gravity: vec3,
                   motor_omega: wp.array(dtype=vec3), motor_enabled: wp.array(dtype=int)):
    b = wp.tid()
    o = b * 4
    state.q_tilde[o] = state.q0[o] + h * state.qdot[o] + h * h * gravity
    w = motor_omega[b]
    for j in range(3):
        v = state.qdot[o + 1 + j]
        if motor_enabled[b] != 0:
            # A_dot = [omega]x A for the column representation.
            v = wp.cross(w, state.q0[o + 1 + j])
        state.q_tilde[o + 1 + j] = state.q0[o + 1 + j] + h * v


@wp.kernel
def _update_affine_velocity(state: ABDState, h: scalar):
    i = wp.tid()
    state.qdot[i] = (state.q[i] - state.q0[i]) / h
    state.q0[i] = state.q[i]


@wp.kernel
def _assemble_body_system(state: ABDState, mass: wp.array(dtype=mat33),
                          volume: wp.array(dtype=scalar), stiffness: wp.array(dtype=scalar),
                          fixed: wp.array(dtype=int), h2: scalar, triplets: Triplets,
                          gradient: wp.array(dtype=vec3)):
    tid = wp.tid()
    b = tid // 16
    local = tid - b * 16
    i = local // 4
    j = local - i * 4
    row = b * 4 + i
    col = b * 4 + j
    triplets.rows[tid] = row
    triplets.cols[tid] = col

    eye = wp.identity(n=3, dtype=scalar)
    if fixed[b] != 0:
        if i == j:
            triplets.vals[tid] = eye
        else:
            triplets.vals[tid] = mat33()
        if j == 0:
            gradient[row] = vec3(scalar(0.0))
        return

    block = mass[tid]
    if i > 0 and j > 0:
        ai = state.q[b * 4 + i]
        aj = state.q[b * 4 + j]
        scale = scalar(4.0) * stiffness[b] * volume[b] * h2
        if i == j:
            s = wp.dot(ai, ai) - scalar(1.0)
            block = block + scale * (s * eye + scalar(2.0) * wp.outer(ai, ai))
            for k in range(1, 4):
                if k != i:
                    ak = state.q[b * 4 + k]
                    block = block + scale * wp.outer(ak, ak)
        else:
            block = block + scale * (wp.dot(ai, aj) * eye + wp.outer(aj, ai))
    triplets.vals[tid] = block

    if j == 0:
        g = vec3(scalar(0.0))
        for k in range(4):
            g = g + mass[b * 16 + i * 4 + k] @ (state.q[b * 4 + k] - state.q_tilde[b * 4 + k])
        if i > 0:
            ai = state.q[b * 4 + i]
            og = (wp.dot(ai, ai) - scalar(1.0)) * ai
            for k in range(1, 4):
                if k != i:
                    ak = state.q[b * 4 + k]
                    og = og + wp.dot(ai, ak) * ak
            g = g + scalar(4.0) * stiffness[b] * volume[b] * h2 * og
        gradient[row] = g


@wp.kernel
def _reduce_contact_force(vertex_force: wp.array(dtype=vec3), rest: wp.array(dtype=vec3),
                          body: wp.array(dtype=int), fixed: wp.array(dtype=int),
                          reduced_force: wp.array(dtype=vec3)):
    i = wp.tid()
    b = body[i]
    if fixed[b] == 0:
        o = b * 4
        f = vertex_force[i]
        r = rest[i]
        wp.atomic_add(reduced_force, o, f)
        wp.atomic_add(reduced_force, o + 1, r[0] * f)
        wp.atomic_add(reduced_force, o + 2, r[1] * f)
        wp.atomic_add(reduced_force, o + 3, r[2] * f)


@wp.kernel
def _add_contact_gradient(gradient: wp.array(dtype=vec3), force: wp.array(dtype=vec3), h2: scalar):
    i = wp.tid()
    gradient[i] = gradient[i] - h2 * force[i]


@wp.kernel
def _reduce_contact_hessian(src: Triplets, rest: wp.array(dtype=vec3),
                            body: wp.array(dtype=int), fixed: wp.array(dtype=int), dst: Triplets):
    tid = wp.tid()
    source = tid // 16
    local = tid - source * 16
    a = local // 4
    c = local - a * 4
    vi = src.rows[source]
    vj = src.cols[source]
    bi = body[vi]
    bj = body[vj]
    dst.rows[tid] = bi * 4 + a
    dst.cols[tid] = bj * 4 + c
    if fixed[bi] != 0 or fixed[bj] != 0:
        dst.vals[tid] = mat33()
        return
    ca = scalar(1.0)
    cc = scalar(1.0)
    if a > 0:
        ca = rest[vi][a - 1]
    if c > 0:
        cc = rest[vj][c - 1]
    dst.vals[tid] = ca * cc * src.vals[source]


@wp.kernel
def _body_energy(state: ABDState, mass: wp.array(dtype=mat33), volume: wp.array(dtype=scalar),
                 stiffness: wp.array(dtype=scalar), fixed: wp.array(dtype=int), h2: scalar,
                 inertia: wp.array(dtype=scalar), ortho: wp.array(dtype=scalar)):
    b = wp.tid()
    if fixed[b] != 0:
        return
    e = scalar(0.0)
    for i in range(4):
        di = state.q[b * 4 + i] - state.q_tilde[b * 4 + i]
        for j in range(4):
            dj = state.q[b * 4 + j] - state.q_tilde[b * 4 + j]
            e = e + scalar(0.5) * wp.dot(di, mass[b * 16 + i * 4 + j] @ dj)
    wp.atomic_add(inertia, 0, e)

    oe = scalar(0.0)
    for i in range(1, 4):
        ai = state.q[b * 4 + i]
        d = wp.dot(ai, ai) - scalar(1.0)
        oe = oe + d * d
        for j in range(i + 1, 4):
            aj = state.q[b * 4 + j]
            d = wp.dot(ai, aj)
            oe = oe + scalar(2.0) * d * d
    wp.atomic_add(ortho, 0, stiffness[b] * volume[b] * h2 * oe)


class AffineBodyDynamics(RodComplexBC):
    """A clinical ABD specialization of :class:`RodComplexBC`."""

    def __init__(self, h, meshes, transforms=None, fixed_bodies=None, density=1000.0,
                 affine_stiffness=1.0e8, gravity=(0.0, 0.0, 0.0), motors=None):
        self._abd_initializing = True
        self._density = float(density)
        self._fixed_body_spec = set(fixed_bodies or [])
        self._gravity_np = np.asarray(gravity, dtype=np.float64)
        if transforms is None:
            transforms = [np.eye(4, dtype=np.float64) for _ in meshes]
        super().__init__(h, list(map(str, meshes)), transforms)
        self.disable_self_collision = True

        world = self.states.x.numpy().astype(np.float64, copy=True)
        body = self.body.numpy().astype(np.int32)
        faces = self.F.astype(np.int32, copy=False)
        centers, volumes, weights = self._mass_properties(world, faces, body)
        local = world - centers[body]
        self.xcs.assign(local)

        self.abd_states = ABDState()
        n_q = self.n_bodies * 4
        self.abd_states.q = wp.zeros(n_q, dtype=vec3)
        self.abd_states.q0 = wp.zeros(n_q, dtype=vec3)
        self.abd_states.q_tilde = wp.zeros(n_q, dtype=vec3)
        self.abd_states.qdot = wp.zeros(n_q, dtype=vec3)
        self.abd_states.dx = wp.zeros(n_q, dtype=vec3)
        q = np.zeros((n_q, 3), dtype=np.float64)
        for b in range(self.n_bodies):
            q[4 * b] = centers[b]
            q[4 * b + 1:4 * b + 4] = np.eye(3)
        self.abd_states.q.assign(q)
        self.abd_states.q0.assign(q)

        fixed = np.asarray([int(b in self._fixed_body_spec) for b in range(self.n_bodies)], dtype=np.int32)
        self.abd_fixed = wp.array(fixed, dtype=int)
        self.abd_volume = wp.array(volumes, dtype=scalar)
        stiff = np.broadcast_to(np.asarray(affine_stiffness, dtype=np.float64), (self.n_bodies,)).copy()
        self.abd_stiffness = wp.array(stiff, dtype=scalar)
        self.abd_mass = wp.array(self._build_mass_blocks(local, body, weights), dtype=mat33)
        self.gravity = vec3(*self._gravity_np)

        motor_w = np.zeros((self.n_bodies, 3), dtype=np.float64)
        motor_on = np.zeros(self.n_bodies, dtype=np.int32)
        for b, w in (motors or {}).items():
            motor_w[int(b)] = np.asarray(w, dtype=np.float64)
            motor_on[int(b)] = 1
        self.motor_omega = wp.array(motor_w, dtype=vec3)
        self.motor_enabled = wp.array(motor_on, dtype=int)

        self.reduced_b = wp.zeros(n_q, dtype=vec3)
        self.reduced_contact_force = wp.zeros(n_q, dtype=vec3)
        self.base_triplets = Triplets()
        self.base_triplets.rows = wp.zeros(self.n_bodies * 16, dtype=int)
        self.base_triplets.cols = wp.zeros_like(self.base_triplets.rows)
        self.base_triplets.vals = wp.zeros(self.n_bodies * 16, dtype=mat33)
        self._abd_initializing = False
        self._sync_vertices()
        self._prepare_prediction()

    # RodBCBase invokes these virtual methods before ABD state exists.
    def define_M(self):
        self.Mnp = np.ones(self.n_nodes, dtype=np.float64)
        self.M = wp.ones(self.n_nodes, dtype=scalar)
        self.M_sparse = bsr_zeros(self.n_nodes, self.n_nodes, mat33)

    def set_fixed_boundary(self):
        self.geo.fixed.zero_()

    def initialize_attachment_matrix(self):
        self.attachment_matrix = bsr_zeros(self.n_nodes, self.n_nodes, mat33)

    def reset(self):
        if not hasattr(self, "abd_states"):
            RodBCBase.reset(self)
            return
        wp.copy(self.abd_states.q, self.abd_states.q0)
        self.abd_states.qdot.zero_()
        self.abd_states.dx.zero_()
        self._sync_vertices()
        self.theta = 0.0
        self.frame = 0

    @staticmethod
    def _mass_properties(vertices, faces, body):
        n_bodies = int(body.max()) + 1
        centers = np.zeros((n_bodies, 3), dtype=np.float64)
        volumes = np.zeros(n_bodies, dtype=np.float64)
        area_weights = np.zeros(len(vertices), dtype=np.float64)
        for b in range(n_bodies):
            fb = faces[body[faces[:, 0]] == b]
            if len(fb):
                a, c, d = vertices[fb[:, 0]], vertices[fb[:, 1]], vertices[fb[:, 2]]
                signed = np.einsum("ij,ij->i", a, np.cross(c, d)) / 6.0
                total = signed.sum()
                if abs(total) > 1.0e-12:
                    centers[b] = np.sum(signed[:, None] * (a + c + d) / 4.0, axis=0) / total
                    volumes[b] = abs(total)
                else:
                    centers[b] = vertices[body == b].mean(axis=0)
                    ext = np.ptp(vertices[body == b], axis=0)
                    volumes[b] = max(float(np.prod(ext)), 1.0e-12)
                area = 0.5 * np.linalg.norm(np.cross(c - a, d - a), axis=1)
                np.add.at(area_weights, fb.reshape(-1), np.repeat(area / 3.0, 3))
            else:
                centers[b] = vertices[body == b].mean(axis=0)
                volumes[b] = 1.0
                area_weights[body == b] = 1.0
        return centers, volumes, area_weights

    def _build_mass_blocks(self, local, body, area_weights):
        blocks = np.zeros((self.n_bodies * 16, 3, 3), dtype=np.float64)
        eye = np.eye(3)
        volumes = self.abd_volume.numpy() if hasattr(self, "abd_volume") else None
        for b in range(self.n_bodies):
            ids = np.flatnonzero(body == b)
            w = area_weights[ids]
            total_mass = self._density * (float(volumes[b]) if volumes is not None else 1.0)
            w = np.full(len(ids), total_mass / len(ids)) if w.sum() == 0 else w / w.sum() * total_mass
            coeff = np.column_stack((np.ones(len(ids)), local[ids]))
            moment = coeff.T @ (w[:, None] * coeff)
            for i in range(4):
                for j in range(4):
                    blocks[b * 16 + i * 4 + j] = moment[i, j] * eye
        return blocks

    def _sync_vertices(self):
        wp.launch(_sync_affine_vertices, self.n_nodes,
                  inputs=[self.abd_states.q, self.xcs, self.body, self.states.x])

    def _prepare_prediction(self):
        wp.launch(_prepare_tilde, self.n_bodies,
                  inputs=[self.abd_states, self.h, self.gravity, self.motor_omega, self.motor_enabled])

    def set_initial_velocity(self, body=0, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)):
        q = self.abd_states.q.numpy()
        qdot = self.abd_states.qdot.numpy()
        o = int(body) * 4
        qdot[o] = np.asarray(linear, dtype=np.float64)
        w = np.asarray(angular, dtype=np.float64)
        for j in range(3):
            qdot[o + 1 + j] = np.cross(w, q[o + 1 + j])
        self.abd_states.qdot.assign(qdot)

    def step(self):
        self._prepare_prediction()
        RodBCBase.step(self)

    def compute_A(self):
        with self.profile_timer("compute A"):
            with self.profile_timer("detect collision"):
                self.detect_collision()
            self.reduced_b.zero_()
            wp.launch(_assemble_body_system, self.n_bodies * 16,
                      inputs=[self.abd_states, self.abd_mass, self.abd_volume,
                              self.abd_stiffness, self.abd_fixed, self.h * self.h,
                              self.base_triplets, self.reduced_b])
            self.A = bsr_from_triplets(self.n_bodies * 4, self.n_bodies * 4,
                                       self.base_triplets.rows, self.base_triplets.cols,
                                       self.base_triplets.vals)

            with self.profile_timer("compute contact hessian"):
                n_vertex_blocks = (self.n_contacts + self.n_contacts_pt) * 16
                vertex_triplets = Triplets()
                vertex_triplets.rows = wp.zeros(n_vertex_blocks, dtype=int)
                vertex_triplets.cols = wp.zeros(n_vertex_blocks, dtype=int)
                vertex_triplets.vals = wp.zeros(n_vertex_blocks, dtype=mat33)
                vertex_force = wp.zeros(self.n_nodes, dtype=vec3)
                wp.launch(contact_hessian_ee, self.n_contacts,
                          inputs=[self.states, self.soup, self.contacts_new.list, vertex_triplets, vertex_force])
                wp.launch(contact_hessian_pt, self.n_contacts_pt,
                          inputs=[self.states, self.soup, self.contacts_pt.list, vertex_triplets,
                                  vertex_force, self.n_contacts])
                self.reduced_contact_force.zero_()
                wp.launch(_reduce_contact_force, self.n_nodes,
                          inputs=[vertex_force, self.xcs, self.body, self.abd_fixed,
                                  self.reduced_contact_force])
                wp.launch(_add_contact_gradient, self.n_bodies * 4,
                          inputs=[self.reduced_b, self.reduced_contact_force, self.h * self.h])

                reduced_nnz = n_vertex_blocks * 16
                reduced = Triplets()
                reduced.rows = wp.zeros(reduced_nnz, dtype=int)
                reduced.cols = wp.zeros(reduced_nnz, dtype=int)
                reduced.vals = wp.zeros(reduced_nnz, dtype=mat33)
                wp.launch(_reduce_contact_hessian, reduced_nnz,
                          inputs=[vertex_triplets, self.xcs, self.body, self.abd_fixed, reduced])
                self.collision_hessian = bsr_from_triplets(
                    self.n_bodies * 4, self.n_bodies * 4,
                    reduced.rows, reduced.cols, reduced.vals)
                bsr_axpy(self.collision_hessian, self.A, self.h * self.h, 1.0)
        self.b = self.reduced_b

    def compute_rhs(self):
        # compute_A assembles the reduced incremental-potential gradient.
        self.b = self.reduced_b

    def solve(self):
        with self.profile_timer("solve"):
            self.abd_states.dx.zero_()
            cg(self.A, self.reduced_b, self.abd_states.dx, tol=1.0e-7, maxiter=100,
               use_cuda_graph=False)
            wp.launch(_affine_vertex_direction, self.n_nodes,
                      inputs=[self.abd_states.dx, self.xcs, self.body, self.states.dx])

    def line_search_upper_bound(self):
        wp.launch(_affine_vertex_direction, self.n_nodes,
                  inputs=[self.abd_states.dx, self.xcs, self.body, self.states.dx])
        return self.collision_free_step(self.states.dx)

    def line_search(self):
        with self.profile_timer("line search"):
            q_start = wp.clone(self.abd_states.q)
            e0 = self.compute_inertia() + self.compute_psi() + self.compute_collision_energy()
            upper = self.line_search_upper_bound()
            alpha = upper
            e1 = np.inf
            accepted = False
            for _ in range(64):
                wp.copy(self.abd_states.q, q_start)
                wp.launch(_affine_add_step, self.n_bodies * 4,
                          inputs=[self.abd_states.q, self.abd_states.dx, alpha])
                self._sync_vertices()
                e1 = self.compute_inertia() + self.compute_psi() + self.compute_collision_energy()
                if np.isfinite(e1) and e1 < e0:
                    accepted = True
                    break
                alpha *= 0.5
                if alpha <= np.finfo(np.float64).eps * max(1.0, upper):
                    break
            if not accepted:
                wp.copy(self.abd_states.q, q_start)
                self._sync_vertices()
                alpha = 0.0
            print(f"    alpha = {alpha:1.2e}, E0 = {e0:1.2e}, E1 = {e1:1.2e}, upper bound = {upper:1.2e}")
            return alpha

    def _energies(self):
        inertia = wp.zeros(1, dtype=scalar)
        ortho = wp.zeros(1, dtype=scalar)
        wp.launch(_body_energy, self.n_bodies,
                  inputs=[self.abd_states, self.abd_mass, self.abd_volume,
                          self.abd_stiffness, self.abd_fixed, self.h * self.h,
                          inertia, ortho])
        return float(inertia.numpy()[0]), float(ortho.numpy()[0])

    def compute_inertia(self):
        return self._energies()[0]

    def compute_psi(self):
        return self._energies()[1]

    def compute_bc_energy(self):
        return 0.0

    def update_x0_xdot(self):
        wp.launch(_update_affine_velocity, self.n_bodies * 4,
                  inputs=[self.abd_states, self.h])
        self._sync_vertices()
        # Keep the inherited vertex history useful for rendering/checkpoints.
        wp.copy(self.states.x0, self.states.x)

    def compute_V(self, ret=True):
        self._sync_vertices()
        if ret:
            self.V = self.states.x.numpy()
            return self.V
        return None


SCREW_ASSET_DIR = Path(r"D:\ref_repos\warp-ipc\assets\sim_data\trimesh\screw-and-nut")


class ScrewSpin(AffineBodyDynamics):
    def __init__(self, h=0.005, angular_velocity=(1.0, 2.0, 3.0)):
        super().__init__(h, [SCREW_ASSET_DIR / "screw-big-2.obj"], gravity=(0, 0, 0))
        self.set_initial_velocity(0, angular=angular_velocity)


class ScrewAndNut(AffineBodyDynamics):
    def __init__(self, h=0.005):
        super().__init__(
            h,
            [SCREW_ASSET_DIR / "screw-big-2.obj", SCREW_ASSET_DIR / "nut-big-2.obj"],
            fixed_bodies=[1],
            gravity=(0, 0, 0),
            affine_stiffness=1.0e8,
            motors={0: (0.0, -np.pi, 0.0)},
        )
