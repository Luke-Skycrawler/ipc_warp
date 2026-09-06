import warp as wp

from .params import *


@wp.func
def reparam(mu: scalar, lam: scalar):
    """Convert standard Lamé parameters to warp-ipc's SNH parameters."""
    snh_mu = scalar(4.0 / 3.0) * mu
    snh_lam = lam + scalar(5.0 / 6.0) * mu
    alpha = scalar(1.0) + scalar(0.75) * snh_mu / snh_lam
    return snh_mu, snh_lam, alpha


@wp.func
def psi(F: mat33) -> scalar:
    """Stable Neo-Hookean energy, defined for positive and negative J."""
    snh_mu, snh_lam, alpha = reparam(mu, lam)
    Ic = wp.ddot(F, F)
    JminusAlpha = wp.determinant(F) - alpha
    return scalar(0.5) * (
        snh_lam * JminusAlpha * JminusAlpha
        + snh_mu * (Ic - scalar(3.0))
        - snh_mu * wp.log(Ic + scalar(1.0))
    )


@wp.func
def partialJpartialF(F: mat33) -> mat33:
    FT = wp.transpose(F)
    return wp.matrix_from_cols(
        wp.cross(FT[1], FT[2]),
        wp.cross(FT[2], FT[0]),
        wp.cross(FT[0], FT[1]),
    )


@wp.func
def PK1(F: mat33) -> mat33:
    snh_mu, snh_lam, alpha = reparam(mu, lam)
    Ic = wp.ddot(F, F)
    cofactor = partialJpartialF(F)
    distortional_coeff = snh_mu * (
        scalar(1.0) - scalar(1.0) / (Ic + scalar(1.0))
    )
    volumetric_coeff = snh_lam * (wp.determinant(F) - alpha)
    return distortional_coeff * F + volumetric_coeff * cofactor


@wp.func
def tangent_stiffness(F: mat33, dF: mat33) -> mat33:
    """Directional derivative of PK1 in direction dF."""
    snh_mu, snh_lam, alpha = reparam(mu, lam)
    cofactor = partialJpartialF(F)

    Ic = wp.ddot(F, F)
    Ic_plus_one = Ic + scalar(1.0)
    dIc = scalar(2.0) * wp.ddot(F, dF)
    distortional_coeff = snh_mu * (
        scalar(1.0) - scalar(1.0) / Ic_plus_one
    )
    d_distortional_coeff = snh_mu * dIc / (Ic_plus_one * Ic_plus_one)

    dJ = wp.ddot(cofactor, dF)
    volumetric_coeff = snh_lam * (wp.determinant(F) - alpha)

    f0 = vec3(F[0, 0], F[1, 0], F[2, 0])
    f1 = vec3(F[0, 1], F[1, 1], F[2, 1])
    f2 = vec3(F[0, 2], F[1, 2], F[2, 2])
    df0 = vec3(dF[0, 0], dF[1, 0], dF[2, 0])
    df1 = vec3(dF[0, 1], dF[1, 1], dF[2, 1])
    df2 = vec3(dF[0, 2], dF[1, 2], dF[2, 2])
    dcofactor = wp.matrix_from_cols(
        wp.cross(df1, f2) + wp.cross(f1, df2),
        wp.cross(df2, f0) + wp.cross(f2, df0),
        wp.cross(df0, f1) + wp.cross(f0, df1),
    )

    return (
        distortional_coeff * dF
        + d_distortional_coeff * F
        + snh_lam * dJ * cofactor
        + volumetric_coeff * dcofactor
    )
