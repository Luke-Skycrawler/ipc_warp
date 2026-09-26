"""Barrier-free augmented-Lagrangian contact for the Warp FEM solver.

Implements Algorithms 1--3 from Zheng, Luo, and Li, "Robust and Efficient
Penetration-Free Elastodynamics without Barriers". Candidate states may
intersect; contact constraints are unsigned distances linearized at the last
intersection-free state.
"""

import argparse
from pathlib import Path

import numpy as np
import warp as wp
from warp.fem.linalg import array_axpy
from warp.sparse import bsr_axpy, bsr_from_triplets

import contact as contact_module
from contact import ContactSolverBase, XConstraint, closest_point_triangle
from dynamic_contacts import (
    embed_point_edge_distance,
    edge_edge_distance_gradient_hessian,
    point_triangle_distance_gradient_hessian,
)
from fem.fem import Triplets
from fem.interface import RodComplex
from fem.params import FEMMesh, NewtonState, gravity
from geometry.static_scene import StaticScene
from ipctkwp.distance.mollifier import ee_mollifier_threshold, ee_mollifier_value
from scalar_types import mat33, scalar, vec3, vec12
from time_integrator_base import RodBCBase
from viewer import PSViewer

DECAY_FACTOR = scalar(0.9)
ACTIVE_SET_DECAY_THRESHOLD = scalar(0.01)
MU_DIAGONAL_SCALE = scalar(0.1)
TOI_FILTER_EPS = scalar(1.0e-12)
# m < 1 is exactly the region where IPC's EE mollifier is active.  The AL
# formulation uses it as a classifier and switches to a stable PE constraint.
EE_MOLLIFIER_POINT_EDGE_THRESHOLD = scalar(1.0)


@wp.struct
class ALConstraint:
    a1a2b1b2: wp.vec4i
    e0e1: wp.vec2i
    l0: scalar
    alpha: scalar
    lam: scalar
    gamma: scalar
    slack: scalar
    mollifier: scalar
    # 0: regular EE, 1/2: edge-A endpoint against B, 3/4: B against A.
    distance_type: int
    # c(x_hat) = d0 + grad^T x_hat.
    d0: scalar
    grad: vec12


@wp.struct
class Hash:
    """Compact hash table supporting lookup and contiguous traversal."""

    table_size: int
    cell_start: wp.array(dtype=int)
    cell_entries: wp.array(dtype=ALConstraint)


@wp.func
def hash_pos(hash_table: Hash, e0: int, e1: int) -> int:
    value = wp.bit_xor(e0 * 73856093, e1 * 19349663)
    if value < 0:
        value = -value
    return value % hash_table.table_size


@wp.func
def constraint_value(c: ALConstraint, x: wp.array(dtype=vec3)) -> scalar:
    value = c.d0
    for local in range(4):
        vertex = c.a1a2b1b2[local]
        value += wp.dot(
            vec3(c.grad[3 * local], c.grad[3 * local + 1], c.grad[3 * local + 2]),
            x[vertex],
        )
    return value


@wp.func
def to_al_constraint(xc: XConstraint) -> ALConstraint:
    result = ALConstraint()
    result.a1a2b1b2 = xc.a1a2b1b2
    result.e0e1 = xc.e0e1
    result.l0 = xc.l0
    result.alpha = xc.alpha
    result.lam = scalar(0.0)
    result.gamma = scalar(1.0)
    result.slack = scalar(0.0)
    result.mollifier = scalar(1.0)
    result.distance_type = 0
    result.d0 = scalar(0.0)
    result.grad = vec12()
    return result


@wp.kernel
def accumulate_vertex_toi(
    constraints: wp.array(dtype=XConstraint),
    vertex_toi: wp.array(dtype=scalar),
):
    i = wp.tid()
    c = constraints[i]
    for local in range(4):
        wp.atomic_min(vertex_toi, c.a1a2b1b2[local], c.alpha)


@wp.func
def filter_accept(c: XConstraint, vertex_toi: wp.array(dtype=scalar)) -> bool:
    for local in range(4):
        if c.alpha <= vertex_toi[c.a1a2b1b2[local]] + TOI_FILTER_EPS:
            return True
    return False


@wp.kernel
def count_constraints(
    hash_table: Hash,
    constraints: wp.array(dtype=XConstraint),
    vertex_toi: wp.array(dtype=scalar),
):
    i = wp.tid()
    c = constraints[i]
    if filter_accept(c, vertex_toi):
        key = c.e0e1
        wp.atomic_add(hash_table.cell_start, hash_pos(hash_table, key[0], key[1]), 1)


@wp.kernel
def register_constraints(
    hash_table: Hash,
    constraints: wp.array(dtype=XConstraint),
    vertex_toi: wp.array(dtype=scalar),
):
    i = wp.tid()
    c = constraints[i]
    if filter_accept(c, vertex_toi):
        key = c.e0e1
        bucket = hash_pos(hash_table, key[0], key[1])
        output = wp.atomic_add(hash_table.cell_start, bucket, -1) - 1
        hash_table.cell_entries[output] = to_al_constraint(c)


@wp.func
def keep_old(c: ALConstraint) -> bool:
    return c.gamma >= ACTIVE_SET_DECAY_THRESHOLD


@wp.func
def is_new_constraint(old: Hash, c: ALConstraint) -> bool:
    key = c.e0e1
    bucket = hash_pos(old, key[0], key[1])
    for i in range(old.cell_start[bucket], old.cell_start[bucket + 1]):
        old_key = old.cell_entries[i].e0e1
        if old_key[0] == key[0] and old_key[1] == key[1]:
            return False
    return True

@wp.kernel
def count_merge(new: Hash, old: Hash, destination: Hash, n_old: int, n_new: int):
    i = wp.tid()
    if i < n_old:
        c = old.cell_entries[i]
        if keep_old(c):
            key = c.e0e1
            wp.atomic_add(destination.cell_start, hash_pos(destination, key[0], key[1]), 1)
    else:
        c = new.cell_entries[i - n_old]
        if is_new_constraint(old, c):
            key = c.e0e1
            wp.atomic_add(destination.cell_start, hash_pos(destination, key[0], key[1]), 1)


@wp.kernel
def register_merge(new: Hash, old: Hash, destination: Hash, n_old: int, n_new: int):
    i = wp.tid()
    accepted = False
    c = ALConstraint()
    if i < n_old:
        c = old.cell_entries[i]
        accepted = keep_old(c)
    else:
        c = new.cell_entries[i - n_old]
        accepted = is_new_constraint(old, c)
    if accepted:
        key = c.e0e1
        bucket = hash_pos(destination, key[0], key[1])
        output = wp.atomic_add(destination.cell_start, bucket, -1) - 1
        destination.cell_entries[output] = c


@wp.func
def point_line_distance_squared(point: vec3, edge0: vec3, edge1: vec3):
    edge = edge1 - edge0
    alpha = wp.dot(point - edge0, edge) / wp.max(wp.dot(edge, edge), scalar(1.0e-30))
    offset = point - (edge0 + alpha * edge)
    return alpha, wp.dot(offset, offset)


@wp.func
def parallel_edge_point_edge_distance(
    x0: vec3, x1: vec3, x2: vec3, x3: vec3
):
    """Choose a deterministic PE representation for a near-parallel EE pair.

    Candidates whose projection lies on the opposite segment are preferred.
    This retains the segment distance for overlapping parallel edges.  If the
    segments do not overlap, the nearest infinite-line PE candidate is used,
    which still provides a finite, consistently oriented normal.
    """
    alpha0, distance0 = point_line_distance_squared(x0, x2, x3)
    alpha1, distance1 = point_line_distance_squared(x1, x2, x3)
    alpha2, distance2 = point_line_distance_squared(x2, x0, x1)
    alpha3, distance3 = point_line_distance_squared(x3, x0, x1)
    large = scalar(1.0e30)
    score0 = distance0
    score1 = distance1
    score2 = distance2
    score3 = distance3
    if alpha0 < scalar(0.0) or alpha0 > scalar(1.0):
        score0 = large
    if alpha1 < scalar(0.0) or alpha1 > scalar(1.0):
        score1 = large
    if alpha2 < scalar(0.0) or alpha2 > scalar(1.0):
        score2 = large
    if alpha3 < scalar(0.0) or alpha3 > scalar(1.0):
        score3 = large

    # If no endpoint projects onto the opposite segment, retain the same
    # deterministic selection rule without the segment-validity preference.
    if wp.min(wp.min(score0, score1), wp.min(score2, score3)) == large:
        score0 = distance0
        score1 = distance1
        score2 = distance2
        score3 = distance3

    distance_type = int(1)
    best = score0
    if score1 < best:
        best = score1
        distance_type = 2
    if score2 < best:
        best = score2
        distance_type = 3
    if score3 < best:
        best = score3
        distance_type = 4

    if distance_type == 1:
        grad, hess = embed_point_edge_distance(x0, x2, x3, 0, 2, 3)
        return grad, hess, distance0, distance_type
    if distance_type == 2:
        grad, hess = embed_point_edge_distance(x1, x2, x3, 1, 2, 3)
        return grad, hess, distance1, distance_type
    if distance_type == 3:
        grad, hess = embed_point_edge_distance(x2, x0, x1, 2, 0, 1)
        return grad, hess, distance2, distance_type
    grad, hess = embed_point_edge_distance(x3, x0, x1, 3, 0, 1)
    return grad, hess, distance3, distance_type


@wp.kernel
def linearize_constraints(
    hash_table: Hash,
    x_safe: wp.array(dtype=vec3),
    x_rest: wp.array(dtype=vec3),
    is_point_triangle: bool,
    point_edge_threshold: scalar,
):
    i = wp.tid()
    c = hash_table.cell_entries[i]
    ids = c.a1a2b1b2
    x0 = x_safe[ids[0]]
    x1 = x_safe[ids[1]]
    x2 = x_safe[ids[2]]
    x3 = x_safe[ids[3]]

    grad_squared = vec12()
    distance_squared = scalar(0.0)
    mollifier = scalar(1.0)
    distance_type = int(0)
    if is_point_triangle:
        grad_squared, hessian_unused = point_triangle_distance_gradient_hessian(x0, x1, x2, x3)
        closest, closest_feature = closest_point_triangle(
            wp.vec3(x0), wp.vec3(x1), wp.vec3(x2), wp.vec3(x3)
        )
        distance_squared = scalar(closest[2]) * scalar(closest[2])
    else:
        eps_x = ee_mollifier_threshold(
            x_rest[ids[0]], x_rest[ids[1]], x_rest[ids[2]], x_rest[ids[3]]
        )
        mollifier = ee_mollifier_value(x0, x1, x2, x3, eps_x)
        if mollifier < point_edge_threshold:
            grad_squared, hessian_unused, distance_squared, distance_type = (
                parallel_edge_point_edge_distance(x0, x1, x2, x3)
            )
        else:
            grad_squared, hessian_unused = edge_edge_distance_gradient_hessian(
                x0, x1, x2, x3, eps_x
            )
            closest = wp.closest_point_edge_edge(
                wp.vec3(x0), wp.vec3(x1), wp.vec3(x2), wp.vec3(x3), 1.0e-6
            )
            distance_squared = scalar(closest[2]) * scalar(closest[2])

    distance = wp.sqrt(wp.max(distance_squared, scalar(1.0e-30)))
    grad = grad_squared / (scalar(2.0) * distance)
    offset = distance - c.l0
    for local in range(4):
        vertex_grad = vec3(grad[3 * local], grad[3 * local + 1], grad[3 * local + 2])
        offset -= wp.dot(vertex_grad, x_safe[ids[local]])
    hash_table.cell_entries[i].d0 = offset
    hash_table.cell_entries[i].grad = grad
    hash_table.cell_entries[i].mollifier = mollifier
    hash_table.cell_entries[i].distance_type = distance_type


@wp.kernel
def update_slack_variables(hash_table: Hash, x_hat: wp.array(dtype=vec3), mu: scalar):
    i = wp.tid()
    c = hash_table.cell_entries[i]
    ci = constraint_value(c, x_hat)
    hash_table.cell_entries[i].slack = wp.max(scalar(0.0), ci - c.lam / mu)


@wp.kernel
def update_active_set_forces(hash_table: Hash, x_hat: wp.array(dtype=vec3), mu: scalar):
    i = wp.tid()
    c = hash_table.cell_entries[i]
    ci = constraint_value(c, x_hat)
    if c.slack == scalar(0.0):
        hash_table.cell_entries[i].lam = c.lam - mu * ci
        hash_table.cell_entries[i].gamma = scalar(1.0)
    else:
        hash_table.cell_entries[i].lam = scalar(0.0)
        hash_table.cell_entries[i].gamma = c.gamma * DECAY_FACTOR


@wp.kernel
def contact_gauss_newton_ee(
    states: NewtonState,
    constraints: wp.array(dtype=ALConstraint),
    triplets: Triplets,
    gradient: wp.array(dtype=vec3),
    mu: scalar,
):
    i = wp.tid()
    c = constraints[i]
    ci = constraint_value(c, states.x)
    residual = ci - c.lam / mu - c.slack
    scale = mu * c.gamma
    for row in range(4):
        gi = vec3(c.grad[3 * row], c.grad[3 * row + 1], c.grad[3 * row + 2])
        wp.atomic_add(gradient, c.a1a2b1b2[row], scale * residual * gi)
        for column in range(4):
            gj = vec3(c.grad[3 * column], c.grad[3 * column + 1], c.grad[3 * column + 2])
            index = i * 16 + row * 4 + column
            triplets.rows[index] = c.a1a2b1b2[row]
            triplets.cols[index] = c.a1a2b1b2[column]
            triplets.vals[index] = scale * wp.outer(gi, gj)


@wp.kernel
def contact_gauss_newton_pt(
    states: NewtonState,
    constraints: wp.array(dtype=ALConstraint),
    triplets: Triplets,
    gradient: wp.array(dtype=vec3),
    mu: scalar,
    contact_offset: int,
):
    i = wp.tid()
    c = constraints[i]
    ci = constraint_value(c, states.x)
    residual = ci - c.lam / mu - c.slack
    scale = mu * c.gamma
    for row in range(4):
        gi = vec3(c.grad[3 * row], c.grad[3 * row + 1], c.grad[3 * row + 2])
        wp.atomic_add(gradient, c.a1a2b1b2[row], scale * residual * gi)
        for column in range(4):
            gj = vec3(c.grad[3 * column], c.grad[3 * column + 1], c.grad[3 * column + 2])
            index = (contact_offset + i) * 16 + row * 4 + column
            triplets.rows[index] = c.a1a2b1b2[row]
            triplets.cols[index] = c.a1a2b1b2[column]
            triplets.vals[index] = scale * wp.outer(gi, gj)


@wp.kernel
def collision_energy_kernel(
    x_hat: wp.array(dtype=vec3),
    constraints: wp.array(dtype=ALConstraint),
    mu: scalar,
    energy: wp.array(dtype=scalar),
):
    i = wp.tid()
    c = constraints[i]
    ci = constraint_value(c, x_hat)
    equality_residual = ci - c.slack
    value = c.gamma * (
        scalar(0.5) * mu * equality_residual * equality_residual
        - c.lam * equality_residual
    )
    wp.atomic_add(energy, 0, value)


@wp.kernel
def compute_tilde_x_kernel(
    x_tilde: wp.array(dtype=vec3),
    x_t: wp.array(dtype=vec3),
    velocity: wp.array(dtype=vec3),
    h: scalar,
):
    i = wp.tid()
    x_tilde[i] = x_t[i] + h * velocity[i] + h * h * gravity


@wp.kernel
def compute_rhs_tilde(
    state: NewtonState,
    x_tilde: wp.array(dtype=vec3),
    h: scalar,
    mass: wp.array(dtype=scalar),
    gradient: wp.array(dtype=vec3),
):
    i = wp.tid()
    gradient[i] = -h * h * gradient[i] + mass[i] * (state.x[i] - x_tilde[i])


@wp.kernel
def compute_inertia_tilde(
    state: NewtonState,
    geo: FEMMesh,
    x_tilde: wp.array(dtype=vec3),
    mass: wp.array(dtype=scalar),
    energy: wp.array(dtype=scalar),
):
    i = wp.tid()
    if geo.fixed[i] == 0:
        displacement = state.x[i] - x_tilde[i]
        wp.atomic_add(energy, 0, scalar(0.5) * mass[i] * wp.length_sq(displacement))


@wp.kernel
def apply_boundary_target(state: NewtonState, geo: FEMMesh, residual: wp.array(dtype=vec3)):
    i = wp.tid()
    if geo.fixed[i] != 0:
        state.x[i] -= residual[i]


@wp.kernel
def eliminate_fixed_dofs(
    offsets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    values: wp.array(dtype=mat33),
    fixed: wp.array(dtype=int),
):
    row = wp.tid()
    for index in range(offsets[row], offsets[row + 1]):
        column = columns[index]
        if fixed[row] != 0 or fixed[column] != 0:
            if row == column and fixed[row] != 0:
                values[index] = wp.identity(3, dtype=scalar)
            else:
                values[index] = mat33()


@wp.kernel
def eliminate_fixed_gradient(gradient: wp.array(dtype=vec3), fixed: wp.array(dtype=int)):
    i = wp.tid()
    if fixed[i] != 0:
        gradient[i] = vec3()


@wp.kernel
def max_diagonal(
    offsets: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    values: wp.array(dtype=mat33),
    result: wp.array(dtype=scalar),
):
    row = wp.tid()
    for index in range(offsets[row], offsets[row + 1]):
        if columns[index] == row:
            block = values[index]
            wp.atomic_max(result, 0, wp.max(block[0, 0], wp.max(block[1, 1], block[2, 2])))


@wp.kernel
def blend_feasible_state(
    x_safe: wp.array(dtype=vec3),
    x_hat: wp.array(dtype=vec3),
    alpha: scalar,
):
    i = wp.tid()
    x_safe[i] = wp.lerp(x_safe[i], x_hat[i], alpha)


class ActiveSet:
    def __init__(self, contact_volume):
        self.hash = self.reserve(max(1, contact_volume))

    @staticmethod
    def reserve(contact_volume):
        hash_table = Hash()
        hash_table.table_size = max(2, 2 * contact_volume)
        hash_table.cell_start = wp.zeros(hash_table.table_size + 1, dtype=int)
        hash_table.cell_entries = wp.zeros(contact_volume, dtype=ALConstraint)
        return hash_table

    @property
    def capacity(self):
        return self.hash.cell_entries.shape[0]

    def count(self):
        return int(self.hash.cell_start.numpy()[-1])

    def create_hash(self, constraints, n_constraints, vertex_toi):
        required = max(1, n_constraints)
        if required > self.capacity:
            self.hash = self.reserve(1 << (required - 1).bit_length())
        self.hash.cell_start.zero_()
        self.hash.cell_entries.zero_()
        if n_constraints == 0:
            return
        wp.launch(count_constraints, n_constraints, inputs=[self.hash, constraints, vertex_toi])
        wp.utils.array_scan(self.hash.cell_start, self.hash.cell_start, inclusive=True)
        wp.launch(register_constraints, n_constraints, inputs=[self.hash, constraints, vertex_toi])

    def merge(self, new):
        n_old = self.count()
        n_new = new.count()
        destination = self.reserve(max(1, n_old + n_new))
        if n_old + n_new == 0:
            self.hash = destination
            return
        wp.launch(count_merge, n_old + n_new, inputs=[new.hash, self.hash, destination, n_old, n_new])
        wp.utils.array_scan(destination.cell_start, destination.cell_start, inclusive=True)
        wp.launch(register_merge, n_old + n_new, inputs=[new.hash, self.hash, destination, n_old, n_new])
        self.hash = destination

    def linearize(self, x_safe, x_rest, is_point_triangle, point_edge_threshold):
        n_constraints = self.count()
        if n_constraints:
            wp.launch(
                linearize_constraints,
                n_constraints,
                inputs=[
                    self.hash,
                    x_safe,
                    x_rest,
                    is_point_triangle,
                    point_edge_threshold,
                ],
            )

    def update_forces(self, x_hat, mu):
        n_constraints = self.count()
        if n_constraints:
            wp.launch(update_active_set_forces, n_constraints, inputs=[self.hash, x_hat, mu])

    def update_slack(self, x_hat, mu):
        n_constraints = self.count()
        if n_constraints:
            wp.launch(update_slack_variables, n_constraints, inputs=[self.hash, x_hat, mu])


class RodComplexAL(RodBCBase, RodComplex, ContactSolverBase):
    """Barrier-free AL time integrator using the existing Warp contact soup."""

    def __init__(self, h, meshes=None, transforms=None, static_meshes: StaticScene = None):
        self.meshes_filename = [] if meshes is None else meshes
        self.transforms = [] if transforms is None else transforms
        RodBCBase.__init__(self, h)
        self.soup.x_transformed = self.states.x
        ContactSolverBase.__init__(self)
        self.termination_tolerance = 1.0e-3
        self.minimum_outer_iterations = 2
        self.max_outer_iterations = 100
        self.max_inner_iterations = 20
        self.ee_mollifier_point_edge_threshold = float(
            EE_MOLLIFIER_POINT_EDGE_THRESHOLD
        )
        self.x_t = wp.zeros_like(self.states.x)
        self.x_safe = wp.zeros_like(self.states.x)
        self.x_tilde = wp.zeros_like(self.states.x)
        self.contact_gradient = wp.zeros_like(self.states.x)
        self.vertex_toi = wp.ones(self.n_nodes, dtype=scalar)
        initial_capacity = max(self.contacts_new.capacity, self.contacts_pt.capacity)
        self.active_set_ee = ActiveSet(initial_capacity)
        self.active_set_pt = ActiveSet(initial_capacity)
        self.detected_set_ee = ActiveSet(initial_capacity)
        self.detected_set_pt = ActiveSet(initial_capacity)
        self.mu = scalar(1.0)

    def _compute_physical_matrix(self):
        self.compute_K()
        bsr_axpy(self.M_sparse, self.K_sparse, 1.0, self.h * self.h)
        self.A = self.K_sparse

    def compute_mu(self):
        """Equation 20: mu = 0.1 max_i (nabla^2 E)_ii."""
        self._compute_physical_matrix()
        diagonal_max = wp.zeros(1, dtype=scalar)
        wp.launch(max_diagonal, self.n_nodes, inputs=[self.A.offsets, self.A.columns, self.A.values, diagonal_max])
        value = float(diagonal_max.numpy()[0])
        return scalar(max(1.0e-8, float(MU_DIAGONAL_SCALE) * value))

    def compute_A(self):
        self._compute_physical_matrix()
        n_ee = self.active_set_ee.count()
        n_pt = self.active_set_pt.count()
        n_contacts = n_ee + n_pt
        self.contact_gradient.zero_()
        self.collision_triplets = Triplets()
        self.collision_triplets.rows = wp.zeros(n_contacts * 16, dtype=int)
        self.collision_triplets.cols = wp.zeros(n_contacts * 16, dtype=int)
        self.collision_triplets.vals = wp.zeros(n_contacts * 16, dtype=mat33)
        if n_ee:
            wp.launch(contact_gauss_newton_ee, n_ee, inputs=[self.states, self.active_set_ee.hash.cell_entries, self.collision_triplets, self.contact_gradient, self.mu])
        if n_pt:
            wp.launch(contact_gauss_newton_pt, n_pt, inputs=[self.states, self.active_set_pt.hash.cell_entries, self.collision_triplets, self.contact_gradient, self.mu, n_ee])
        if n_contacts:
            collision_hessian = bsr_from_triplets(self.n_nodes, self.n_nodes, self.collision_triplets.rows, self.collision_triplets.cols, self.collision_triplets.vals)
            bsr_axpy(collision_hessian, self.A, 1.0, 1.0)
        wp.launch(eliminate_fixed_dofs, self.n_nodes, inputs=[self.A.offsets, self.A.columns, self.A.values, self.geo.fixed])

    def compute_rhs(self):
        wp.launch(compute_rhs_tilde, self.n_nodes, inputs=[self.states, self.x_tilde, self.h, self.M, self.b])
        array_axpy(self.contact_gradient, self.b, 1.0, 1.0)
        wp.launch(eliminate_fixed_gradient, self.n_nodes, inputs=[self.b, self.geo.fixed])

    def compute_collision_energy(self):
        energy = wp.zeros(1, dtype=scalar)
        n_ee = self.active_set_ee.count()
        n_pt = self.active_set_pt.count()
        if n_ee:
            wp.launch(collision_energy_kernel, n_ee, inputs=[self.states.x, self.active_set_ee.hash.cell_entries, self.mu, energy])
        if n_pt:
            wp.launch(collision_energy_kernel, n_pt, inputs=[self.states.x, self.active_set_pt.hash.cell_entries, self.mu, energy])
        return float(energy.numpy()[0])

    def compute_inertia(self):
        energy = wp.zeros(1, dtype=scalar)
        wp.launch(compute_inertia_tilde, self.n_nodes, inputs=[self.states, self.geo, self.x_tilde, self.M, energy])
        return float(energy.numpy()[0])

    def compute_bc_energy(self):
        return 0.0

    def line_search_upper_bound(self):
        return 1.0

    def compute_tilde_x(self):
        wp.copy(self.x_t, self.states.x)
        wp.copy(self.x_safe, self.states.x)
        wp.launch(compute_tilde_x_kernel, self.n_nodes, inputs=[self.x_tilde, self.x_t, self.states.xdot, self.h])

    def move_boundary(self):
        """Move x_hat's fixed vertices to their target without a penalty."""
        self.attachment_residual.zero_()
        self.b.zero_()
        self.compute_compensation()
        wp.launch(apply_boundary_target, self.n_nodes, inputs=[self.states, self.geo, self.attachment_residual])

    def solve_subproblem(self):
        """Algorithm 2: alternating slack/Newton solve and multiplier update."""
        self.active_set_ee.linearize(
            self.x_safe,
            self.soup.xcs,
            False,
            self.ee_mollifier_point_edge_threshold,
        )
        self.active_set_pt.linearize(
            self.x_safe,
            self.soup.xcs,
            True,
            self.ee_mollifier_point_edge_threshold,
        )
        for _ in range(self.max_inner_iterations):
            # Algorithm 2 line 11: update s at the current iterate, then keep
            # it fixed through this Newton direction and line search.
            self.active_set_ee.update_slack(self.states.x, self.mu)
            self.active_set_pt.update_slack(self.states.x, self.mu)
            self.compute_A()
            self.compute_rhs()
            self.solve()
            step_length = self.line_search()
            if step_length == 0.0 or step_length >= 1.0 - 1.0e-12:
                break
        # Algorithm 2 line 23 recomputes slack at the accepted x_hat before
        # the multiplier and decay update.
        self.active_set_ee.update_slack(self.states.x, self.mu)
        self.active_set_pt.update_slack(self.states.x, self.mu)
        self.active_set_ee.update_forces(self.states.x, self.mu)
        self.active_set_pt.update_forces(self.states.x, self.mu)

    def update_active_set(self):
        """Algorithm 3: add per-vertex earliest CCD pairs and decay old ones."""
        alpha = self.new_intersections(self.x_safe, self.states.x)
        self.vertex_toi.fill_(1.0)
        if self.n_contacts:
            wp.launch(accumulate_vertex_toi, self.n_contacts, inputs=[self.contacts_new.list, self.vertex_toi])
        if self.n_contacts_pt:
            wp.launch(accumulate_vertex_toi, self.n_contacts_pt, inputs=[self.contacts_pt.list, self.vertex_toi])
        self.detected_set_ee.create_hash(self.contacts_new.list, self.n_contacts, self.vertex_toi)
        self.detected_set_pt.create_hash(self.contacts_pt.list, self.n_contacts_pt, self.vertex_toi)
        self.active_set_ee.merge(self.detected_set_ee)
        self.active_set_pt.merge(self.detected_set_pt)
        return alpha

    def step(self):
        """Algorithm 1: cumulative-TOI outer iteration."""
        self.compute_tilde_x()
        self.move_boundary()
        self.mu = self.compute_mu()
        beta = 1.0
        iteration = 0
        while beta > self.termination_tolerance:
            if iteration >= self.max_outer_iterations:
                raise RuntimeError(
                    f"AL outer solve did not converge: beta={beta:.3e}, "
                    f"active EE/PT={self.active_set_ee.count()}/{self.active_set_pt.count()}"
                )
            self.solve_subproblem()
            alpha = self.update_active_set()
            wp.launch(blend_feasible_state, self.n_nodes, inputs=[self.x_safe, self.states.x, alpha])
            if iteration + 1 >= self.minimum_outer_iterations:
                beta *= 1.0 - alpha
            print(
                f"    AL iter={iteration}, alpha={alpha:.3e}, beta={beta:.3e}, "
                f"mu={float(self.mu):.3e}, active EE/PT="
                f"{self.active_set_ee.count()}/{self.active_set_pt.count()}"
            )
            iteration += 1
        wp.copy(self.states.x, self.x_safe)
        wp.copy(self.states.x0, self.x_t)
        self.update_x0_xdot()
        self.theta += self.h
        self.frame += 1


class RodsTwistAL(RodComplexAL):
    """Four-rod geometry using the barrier-free AL integrator."""

    def __init__(self, timestep=0.025, high_resolution=False):
        from rods_twist import CONTACT_THICKNESS, LOW_RES_ROD_MESH, PAPER_ROD_MESH

        self.rod_mesh = PAPER_ROD_MESH if high_resolution else LOW_RES_ROD_MESH
        if not self.rod_mesh.exists():
            raise FileNotFoundError(f"Missing {self.rod_mesh}")
        contact_module._thickness = CONTACT_THICKNESS
        contact_module.buffer = CONTACT_THICKNESS
        super().__init__(timestep, meshes=[str(self.rod_mesh)] * 4, transforms=[np.eye(4)] * 4)

    from rods_twist import RodsTwist as _RodsTwistGeometry

    get_next_object = _RodsTwistGeometry.get_next_object
    set_fixed_boundary = _RodsTwistGeometry.set_fixed_boundary
    compute_compensation = _RodsTwistGeometry.compute_compensation


def main():
    import polyscope as ps
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-resolution", action="store_true")
    parser.add_argument("--headless-steps", type=int, default=1)
    parser.add_argument("--save-checkpoint", type=Path)
    args = parser.parse_args()
    if args.headless_steps < 0:
        parser.error("--headless-steps must be non-negative")
    wp.init()
    simulation = RodsTwistAL(high_resolution=args.high_resolution)
    ps.init()
    viewer = PSViewer(simulation)
    
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(viewer.callback)
    ps.show()
    # for _ in range(args.headless_steps):
    #     simulation.step()
    # if args.save_checkpoint is not None:
    #     simulation.save_checkpoint(args.save_checkpoint)


if __name__ == "__main__":
    main()
