import warp as wp 

mat33 = wp.mat33d
mat44 = wp.mat44d
vec4 = wp.vec4d
vec3 = wp.vec3d
scalar = wp.float64
quat = wp.quatd
mat6 = wp.spatial_matrixd
vec6 = wp.spatial_vectord
vec2 = wp.vec2d
mat22 = wp.mat22d
@wp.func
def make_vec6(v3: vec3, w3: vec3):
    return vec6(v3[0], v3[1], v3[2], w3[0], w3[1], w3[2])
