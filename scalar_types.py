import warp as wp


# Precision switch.  Change this one value and restart Python/Warp so kernels
# are compiled for the selected scalar type.
USE_FLOAT32 = True

if USE_FLOAT32:
    mat22 = wp.mat22
    mat33 = wp.mat33
    mat44 = wp.mat44
    vec4 = wp.vec4
    vec3 = wp.vec3
    vec2 = wp.vec2
    scalar = wp.float32
    quat = wp.quat
    mat6 = wp.spatial_matrix
    vec6 = wp.spatial_vector
else:
    mat22 = wp.mat22d
    mat33 = wp.mat33d
    mat44 = wp.mat44d
    vec4 = wp.vec4d
    vec3 = wp.vec3d
    vec2 = wp.vec2d
    scalar = wp.float64
    quat = wp.quatd
    mat6 = wp.spatial_matrixd
    vec6 = wp.spatial_vectord

# Energy reductions deliberately use the pipeline precision as well.
energy_scalar = scalar
scalar_epsilon = 1.1920928955078125e-7 if USE_FLOAT32 else 2.220446049250313e-16

mat34 = wp.types.matrix(shape = (3, 4), dtype = scalar)
mat23 = wp.types.matrix(shape = (2, 3), dtype = scalar)
mat24 = wp.types.matrix(shape = (2, 4), dtype = scalar)
mat12 = wp.types.matrix(shape = (12, 12), dtype = scalar)
mat99 = wp.types.matrix(shape = (9, 9), dtype = scalar)
vec12 = wp.types.vector(length = 12, dtype = scalar)
@wp.func 
def make_vec6(v: vec3, w: vec3) -> vec6:
    return vec6(v[0], v[1], v[2], w[0], w[1], w[2]) 
