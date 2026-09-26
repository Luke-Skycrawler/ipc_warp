"""Warp-only reproduction of warp-ipc's ``wipc_06_wrecking_balls``.

The full scene contains 574 affine bodies plus a fixed mesh ground.  Use
``--body-limit`` for quick chain/contact regression runs.
"""

import argparse

import numpy as np
import warp as wp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-limit", type=int, default=0,
                        help="load only the first N JSON bodies; zero loads all 574")
    parser.add_argument("--solver", choices=("cg", "ldlt"), default="cg")
    parser.add_argument("--dcd", choices=("dynamic", "fixed"), default="dynamic")
    parser.add_argument("--fixed-query", choices=("scalar", "tile"), default="scalar",
                        help="fixed-BVH primitive traversal mode")
    parser.add_argument("--tile-query-threads", type=int, default=32,
                        choices=(32, 64, 128, 256))
    parser.add_argument("--pair-cull", choices=("tile", "kernel"), default="kernel")
    parser.add_argument("--headless-steps", type=int)
    args = parser.parse_args()

    wp.init()
    if args.dcd == "fixed":
        from fixed_bvh_contacts import FixedBVHWreckingBalls
        simulation = FixedBVHWreckingBalls(
            body_limit=args.body_limit, linear_solver=args.solver
        )
        simulation.fixed_query_mode = args.fixed_query
        simulation.tile_query_threads = args.tile_query_threads
        simulation.fixed_pair_cull_mode = args.pair_cull
    else:
        from affine_body_dynamics import WreckingBalls
        simulation = WreckingBalls(
            body_limit=args.body_limit, linear_solver=args.solver
        )

    q_initial = simulation.abd_states.q.numpy().copy()
    print(
        f"scene: bodies={simulation.n_bodies}, vertices={simulation.n_nodes}, "
        f"triangles={len(simulation.F)}"
    )

    if args.headless_steps is not None:
        for _ in range(args.headless_steps):
            simulation.step()
        q = simulation.abd_states.q.numpy()
        centers0 = q_initial[::4]
        centers = q[::4]
        moving = simulation.abd_fixed.numpy() == 0
        max_displacement = 0.0
        if np.any(moving):
            max_displacement = np.max(
                np.linalg.norm(centers[moving] - centers0[moving], axis=1)
            )
        fixed = ~moving
        fixed_drift = 0.0
        if np.any(fixed):
            fixed_ids = np.flatnonzero(fixed)
            fixed_drift = max(
                np.max(np.abs(q[4 * body:4 * body + 4]
                              - q_initial[4 * body:4 * body + 4]))
                for body in fixed_ids
            )
        print(f"frames={simulation.frame}")
        print(f"finite state: {np.isfinite(q).all()}")
        print(f"maximum body-center displacement: {max_displacement:.6e}")
        print(f"maximum fixed-body drift: {fixed_drift:.6e}")
        if simulation.n_bodies > 13:
            print(f"wrecking-ball center: {centers0[13]} -> {centers[13]}")
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
