# Lightweight IPC Warp Implementation

Simplified^[friction is not implemented] implementation of Incremental Potential Contact [5] and related papers [1, 2] using analytical contact hessian filtering [3].
## Prerequisites 
- [cuDSS](https://developer.nvidia.com/cudss)
- [warp-lang](https://nvidia.github.io/warp/stable/user_guide/programming_model/tiles.html)
- libigl
## Examples 

Rod twist:
```
python rods_twist.py --low-resolution
```

Nut and screw:
```
python screw_and_nut.py --solver ldlt --dcd fixed
```

Wrecking ball:
```
python wrecking_balls.py --dcd fixed --pair-cull kernel --fixed-query scalar
```
Precision is selected by `USE_FLOAT32` in `scalar_types.py`. It defaults to
`True`, so the simulation, energy reductions, and direct solver all use fp32.
Set it to `False` and restart Python to use fp64.

## References

[1] Robust and Efficient
Penetration-Free Elastodynamics without Barriers.
[2] Affine Body Dynamics: Fast, Stable and Intersection-free Simulation of
Stiff Materials.
[3] A Unified Analysis of Penalty-Based Collision Energies.
[4] [warp-ipc](https://github.com/liangqx-hku/warp-ipc/tree/main)
[5] Incremental Potential Contact: Intersection- and Inversion-free,
Large-Deformation Dynamics
