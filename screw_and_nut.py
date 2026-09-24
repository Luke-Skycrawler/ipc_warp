"""Affine-body screw demos.

Examples
--------
Free spin about all axes::

    python screw_and_nut.py --mode spin

Threaded screw/nut contact, or a short headless verification::

    python screw_and_nut.py --mode nut
    python screw_and_nut.py --mode nut --headless-steps 80
"""

import argparse

import numpy as np
import warp as wp

from affine_body_dynamics import ScrewAndNut, ScrewSpin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("spin", "nut"), default="nut")
    parser.add_argument("--headless-steps", type=int)
    args = parser.parse_args()

    wp.init()
    simulation = ScrewSpin() if args.mode == "spin" else ScrewAndNut()

    if args.headless_steps is not None:
        q_initial = simulation.abd_states.q.numpy().copy()
        for _ in range(args.headless_steps):
            simulation.step()
        q = simulation.abd_states.q.numpy()
        A = q[1:4].T
        print(f"frames={simulation.frame}")
        print(f"body-0 translation: {q_initial[0]} -> {q[0]}")
        print(f"orthogonality error: {np.linalg.norm(A.T @ A - np.eye(3)):.6e}")
        return

    import polyscope as ps
    from viewer import PSViewer

    ps.init()
    viewer = PSViewer(simulation)
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(viewer.callback)
    ps.show()


if __name__ == "__main__":
    main()
