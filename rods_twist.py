"""Four mutually colliding rods twisted from their end caps.

This reproduces the setup of IPC's ``input/paperExamples/4_rodsTwist.txt``.
The paper-resolution mesh is used by default; pass ``--low-resolution`` for
the geometrically equivalent mesh from IPC's typical example.
"""

import argparse
import os
from pathlib import Path

import igl
import meshio
import numpy as np
import polyscope as ps
import warp as wp
from warp.fem.linalg import array_axpy
from warp.sparse import bsr_mv

import contact as contact_module
from dynamic_contacts import RodComplexBC
from fem.params import FEMMesh, NewtonState
from scalar_types import scalar, vec3
from viewer import PSViewer


REFERENCE_ROOT = Path(os.environ.get("IPC_REFERENCE_ROOT", r"D:\ref_repos\IPC"))
TET_MESH_ROOT = REFERENCE_ROOT / "input" / "tetMeshes"
PAPER_ROD_MESH = TET_MESH_ROOT / "rod300x33.msh"
LOW_RES_ROD_MESH = TET_MESH_ROOT / "rod.msh"

ROD_OFFSETS = (
    (0.0, -0.1, -0.1),
    (0.0, -0.1, 0.1),
    (0.0, 0.1, -0.1),
    (0.0, 0.1, 0.1),
)
TWIST_ANGULAR_SPEED = 0.4 * np.pi
END_CAP_TOLERANCE = 1.0e-5
CONTACT_THICKNESS = 2.0e-4


def orient_boundary(vertices, tetrahedra):
    faces = igl.boundary_facets(tetrahedra)
    faces, _ = igl.bfs_orient(faces)
    components, _ = igl.orientable_patches(faces)
    faces, _ = igl.orient_outward(vertices, faces, components)
    return faces


@wp.kernel
def set_twist_displacement(
    state: NewtonState,
    geo: FEMMesh,
    time: scalar,
    angular_speed: scalar,
    center: vec3,
    displacement: wp.array(dtype=vec3),
):
    i = wp.tid()
    displacement[i] = vec3()
    if geo.fixed[i] != 0:
        rest = geo.xcs[i]
        angle = angular_speed * time
        if rest[0] < center[0]:
            angle = -angle
        c = wp.cos(angle)
        s = wp.sin(angle)
        y = rest[1] - center[1]
        z = rest[2] - center[2]
        target = vec3(
            rest[0],
            center[1] + c * y - s * z,
            center[2] + s * y + c * z,
        )
        displacement[i] = state.x[i] - target


class RodsTwist(RodComplexBC):
    def __init__(self, timestep=0.025, high_resolution=True):
        self.rod_mesh = PAPER_ROD_MESH if high_resolution else LOW_RES_ROD_MESH
        if not self.rod_mesh.exists():
            raise FileNotFoundError(
                f"Missing {self.rod_mesh}. Set IPC_REFERENCE_ROOT to the IPC repository root."
            )

        # This is below the axial spacing of the paper mesh and avoids treating
        # nearby triangles on each undeformed surface as initial contact.
        contact_module._thickness = CONTACT_THICKNESS
        contact_module.buffer = CONTACT_THICKNESS

        super().__init__(
            timestep,
            meshes=[str(self.rod_mesh)] * 4,
            transforms=[np.eye(4)] * 4,
        )

    def get_next_object(self):
        mesh = meshio.read(self.rod_mesh)
        tetrahedra = None
        for block in mesh.cells:
            if block.type == "tetra":
                tetrahedra = np.asarray(block.data, dtype=np.int32)
                break
        if tetrahedra is None:
            raise ValueError(f"No tetrahedra found in {self.rod_mesh}")

        base_vertices = np.asarray(mesh.points[:, :3], dtype=np.float64)
        placed = [base_vertices + np.asarray(offset) for offset in ROD_OFFSETS]
        all_vertices = np.vstack(placed)
        scale = 1.0 / np.ptp(all_vertices, axis=0).max()
        scene_min = all_vertices.min(axis=0)
        placed = [(vertices - scene_min) * scale for vertices in placed]

        self.rod_vertex_count = base_vertices.shape[0]
        self.scene_center = 0.5 * (
            np.vstack(placed).min(axis=0) + np.vstack(placed).max(axis=0)
        )
        for vertices in placed:
            faces = orient_boundary(vertices, tetrahedra)
            yield vertices, igl.edges(faces), faces, tetrahedra, None

    def set_fixed_boundary(self):
        vertices = self.xcs.numpy()
        x_min = vertices[:, 0].min()
        x_max = vertices[:, 0].max()
        fixed = (
            (vertices[:, 0] < x_min + END_CAP_TOLERANCE)
            | (vertices[:, 0] > x_max - END_CAP_TOLERANCE)
        )
        self.geo.fixed.assign(fixed.astype(np.int32))
        self.fixed_indices = np.flatnonzero(fixed)
        print(
            f"four-rod twist: {self.n_nodes} vertices, {self.n_tets} tetrahedra, "
            f"{self.fixed_indices.size} end-cap vertices"
        )

    def compute_compensation(self):
        self.comp_x.zero_()
        wp.launch(
            set_twist_displacement,
            self.n_nodes,
            inputs=[
                self.states,
                self.geo,
                self.theta + self.h,
                TWIST_ANGULAR_SPEED,
                vec3(*self.scene_center),
                self.comp_x,
            ],
        )
        tmp = bsr_mv(self.A, self.comp_x)
        array_axpy(tmp, self.b, 1.0, 1.0)
        wp.copy(self.comp_x, tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--low-resolution",
        action="store_true",
        help="Use rod.msh instead of the paper's rod300x33.msh.",
    )
    args = parser.parse_args()

    wp.init()
    ps.init()
    simulation = RodsTwist(high_resolution=not args.low_resolution)
    viewer = PSViewer(simulation)

    handles = ps.register_point_cloud(
        "twisted end caps", simulation.xcs.numpy()[simulation.fixed_indices]
    )
    handles.set_radius(0.004, relative=False)

    ps.set_ground_plane_mode("none")
    ps.set_user_callback(viewer.callback)
    ps.show()


if __name__ == "__main__":
    main()
