import warp as wp
import numpy as np
from .params import *
from scalar_types import *
@wp.func
def tangent_stiffness(F: mat33, dF: mat33) -> mat33:
    '''
    neo-hookean model
    '''
    F_inv_T = wp.transpose(wp.inverse(F))
    B = wp.inverse(F) @ dF
    det_F = wp.determinant(F)
    
    return mu * dF + (mu - lam * wp.log(det_F)) * F_inv_T @ wp.transpose(dF) @ F_inv_T + (lam * wp.trace(B)) * F_inv_T 

@wp.func
def PK1(F: mat33) -> mat33:
    '''
    neo-hookean
    '''
    F_inv_T = wp.transpose(wp.inverse(F))
    J = wp.determinant(F)
    return mu * (F - F_inv_T) + lam * wp.log(J) * F_inv_T

@wp.func
def psi(F: mat33) -> scalar:
    I1 = wp.trace(wp.transpose(F) @ F)
    J = wp.determinant(F)
    logJ = wp.log(J)
    return mu * scalar(0.5) * (I1 -scalar(3.0)) - mu * logJ + lam * scalar(0.5) * logJ * logJ
    