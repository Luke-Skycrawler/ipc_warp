import warp as wp
from .params import *

@wp.func
def tangent_stiffness(F: mat33, dF: mat33) -> mat33:
    '''
    dP = dF (2 mu E + lam tr(E) I) + F (2 mu dE + lam tr(dE) I)
    '''
    E = 0.5 * (wp.transpose(F) @ F - wp.identity(3, dtype = scalar))
    dE = wp.transpose(dF) @ F
    dE = 0.5 * (dE + wp.transpose(dE))
    return dF @ (2.0 * mu * E + lam * wp.trace(E) * wp.identity(3, dtype = scalar)) + F @ (2.0 * mu * dE + lam * wp.trace(dE) * wp.identity(3, dtype = scalar))

@wp.func
def psi(F: mat33) -> scalar:
    '''
    E = (F^T F - I) / 2
    psi = mu E : E + lam/2 (tr(E))^2
    '''
    E = 0.5 * (wp.transpose(F) @ F - wp.identity(3, dtype = scalar))
    # norm_E = wp.trace(wp.transpose(E) @ E)
    norm_E = wp.ddot(E, E)
    trE = wp.trace(E)
    return mu * norm_E + lam * 0.5 * trE * trE

@wp.func
def PK1(F: mat33) -> mat33: 
    '''
    P = F (2 mu E + lam tr(E) I)
    '''

    E = 0.5 * (wp.transpose(F) @ F - wp.identity(3, dtype = scalar))
    return F @ (2.0 * mu * E + lam * wp.trace(E) * wp.identity(3, dtype = scalar))
    