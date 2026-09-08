"""Run and record a synchronized 120-step profile of the stretch example."""

import contextlib
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import warp as wp

import time_integrator_base as integrator_config
from dynamic_contacts import RodComplexBC


PROFILE_STEPS = 120
OUTPUT_DIR = Path("output")
JSON_PATH = OUTPUT_DIR / "stretch_profile_120.json"
CSV_PATH = OUTPUT_DIR / "stretch_profile_120_summary.csv"


def summarize(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(values.size),
        "total_ms": float(values.sum()),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95.0)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def main():
    wp.init()
    simulation = RodComplexBC(
        integrator_config.h,
        meshes=["assets/bar2.tobj"],
        transforms=[np.eye(4)],
    )

    # Compile every path once without contaminating the recorded timings.
    print("warming up one stretch step...")
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        simulation.step()
    simulation.reset()
    simulation.configure_profiling(
        enabled=True,
        print_timings=False,
        synchronize=True,
    )

    print(f"profiling {PROFILE_STEPS} stretch steps...")
    wall_start = time.perf_counter()
    with open(os.devnull, "w") as sink:
        for step in range(PROFILE_STEPS):
            with contextlib.redirect_stdout(sink):
                simulation.step()
            if (step + 1) % 10 == 0:
                print(f"  completed {step + 1}/{PROFILE_STEPS}")
    wp.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    raw = {name: [float(value) for value in values]
           for name, values in simulation.profile_timings.items()}
    summary = {name: summarize(values) for name, values in raw.items() if values}
    requested = [
        "compute A",
        "detect collision",
        "compute elastic hessian",
        "compute contact hessian",
        "compute rhs",
        "solve",
        "line search",
        "total newton iteration",
        "dx host transfer",
        "contact count host transfer",
        "ccd toi host transfer",
        "energy host transfer",
    ]

    result = {
        "scene": "stretch.py / assets/bar2.tobj",
        "steps": PROFILE_STEPS,
        "warmup_steps": 1,
        "synchronized_scoped_timers": True,
        "solver": integrator_config.solver_choice,
        "nodes": simulation.n_nodes,
        "tetrahedra": simulation.n_tets,
        "newton_iterations": len(raw.get("total newton iteration", [])),
        "wall_ms": wall_ms,
        "summary": summary,
        "raw_ms": raw,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["region", "count", "total_ms", "mean_ms", "median_ms",
                        "p95_ms", "min_ms", "max_ms"],
        )
        writer.writeheader()
        for name in requested:
            if name in summary:
                writer.writerow({"region": name, **summary[name]})

    print(
        f"done: {result['newton_iterations']} Newton iterations, "
        f"{wall_ms / 1000.0:.3f} s wall time"
    )
    for name in requested:
        if name in summary:
            stats = summary[name]
            print(
                f"{name:29s} {stats['mean_ms']:9.3f} ms mean  "
                f"{stats['p95_ms']:9.3f} ms p95  {stats['total_ms']:11.3f} ms total"
            )
    print(f"wrote {JSON_PATH} and {CSV_PATH}")


if __name__ == "__main__":
    main()
