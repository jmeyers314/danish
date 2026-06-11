# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC. and
# Stanford University.
# All rights reserved.
# LLNL-CODE-826307

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os
from functools import lru_cache

import galsim
import numpy as np
from scipy.spatial import Delaunay

from ._danish import poly_grid_contains, pixel_frac, enclosed_circle, enclosed_strut
from .utils import hexapolar, gq_points

F2P_PREFIT_ORDER = 2
F2P_MAXITER = 10
F2P_TOL = 1e-7
F2P_STRICT = False
F2P_ACTIVE_SET_MIN = 100

# Set to a list to collect per-call stats from _focal_to_pupil; None disables (no overhead).
_f2p_stats = None


# ---------------------------------------------------------------------------
# Triangle-image accumulation helpers
# ---------------------------------------------------------------------------

def _sh_clip(poly, inside_fn, intersect_fn):
    """One-edge Sutherland-Hodgman clip."""
    if not poly:
        return poly
    result = []
    s = poly[-1]
    for p in poly:
        if inside_fn(p):
            if not inside_fn(s):
                result.append(intersect_fn(s, p))
            result.append(p)
        elif inside_fn(s):
            result.append(intersect_fn(s, p))
        s = p
    return result


def _clip_area(tri_verts, ix, iy):
    """Area of intersection of triangle with pixel [ix±0.5, iy±0.5]."""
    x0, x1 = ix - 0.5, ix + 0.5
    y0, y1 = iy - 0.5, iy + 0.5

    def lerp(s, p, val, ax):
        t = (val - s[ax]) / (p[ax] - s[ax])
        return s + t * (p - s)

    poly = list(tri_verts)
    poly = _sh_clip(poly, lambda p: p[0] >= x0, lambda s, p: lerp(s, p, x0, 0))
    poly = _sh_clip(poly, lambda p: p[0] <= x1, lambda s, p: lerp(s, p, x1, 0))
    poly = _sh_clip(poly, lambda p: p[1] >= y0, lambda s, p: lerp(s, p, y0, 1))
    poly = _sh_clip(poly, lambda p: p[1] <= y1, lambda s, p: lerp(s, p, y1, 1))

    if len(poly) < 3:
        return 0.0

    arr = np.array(poly)
    x, y = arr[:, 0], arr[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _shoelace_area(verts):
    """Signed shoelace area of a polygon (verts: (..., N, 2))."""
    x = verts[..., :, 0]
    y = verts[..., :, 1]
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=-1) - y * np.roll(x, -1, axis=-1), axis=-1)
    )


def pupil_to_focal(
    u, v, *,
    Z=None, aberrations=None, R_outer=1.0, R_inner=0.0,
    focal_length=None,
    x_offset=None, y_offset=None
):
    """Transform pupil coordinates to focal coordinates.

    Parameters
    ----------
    u, v : array of float
        Pupil coordinates in meters.
    Z : galsim.zernike.Zernike, optional
        Aberrations in meters.
    aberrations : array of float, optional
        Aberrations in meters.
    R_outer, R_inner : float, optional
        Annulus outer and inner radii in meters.
    focal_length : float
        Focal length in meters.
    x_offset, y_offset : galsim.zernike.Zernike, optional
        Additional focal plane offsets (in meters) represented as Zernike
        series.

    Returns
    -------
    x, y : array of float
        Focal coordinates in meters.
    """
    if Z is None:
        Z = galsim.zernike.Zernike(
            aberrations, R_outer=R_outer, R_inner=R_inner
        )
    if focal_length is None:
        raise ValueError("Missing focal length")

    return _pupil_to_focal(
        u, v, Z,
        focal_length=focal_length,
        x_offset=x_offset, y_offset=y_offset
    )


def _pupil_to_focal(
        u, v, Z, *,
        focal_length=None,
        x_offset=None, y_offset=None
):
    Z1 = Z * focal_length if focal_length else Z
    zx = -Z1.gradX
    zy = -Z1.gradY
    if x_offset is not None:
        zx += x_offset
    if y_offset is not None:
        zy += y_offset
    return zx(u, v), zy(u, v)


def pupil_focal_jacobian(
    u, v, *,
    Z=None, aberrations=None, R_outer=1.0, R_inner=0.0,
    focal_length=None,
    x_offset=None, y_offset=None
):
    """Compute Jacobian of the pupil-to-focal coordinate transform.

    Parameters
    ----------
    u, v : array of float
        Pupil coordinates in meters.
    Z : galsim.zernike.Zernike, optional
        Aberrations in meters.
    aberrations : array of float, optional
        Aberrations in meters.
    R_outer, R_inner : float, optional
        Annulus outer and inner radii in meters.
    focal_length : float
        Focal length in meters.
    x_offset, y_offset : galsim.zernike.Zernike, optional
        Additional focal plane offsets (in meters) represented as Zernike
        series.

    Returns
    -------
    dxdu, dxdv, dydu, dydv : array of float
        Jacobian of focal coordinates with respect to pupil coordinates.
    """
    if Z is None:
        Z = galsim.zernike.Zernike(
            aberrations, R_outer=R_outer, R_inner=R_inner
        )
    if focal_length is None:
        raise ValueError("Missing focal length")

    return _pupil_focal_jacobian(
        u, v, Z,
        focal_length=focal_length,
        x_offset=x_offset, y_offset=y_offset
    )


def _pupil_focal_jacobian(
    u, v, Z, *,
    focal_length=None,
    x_offset=None, y_offset=None
):
    Z1 = Z * focal_length if focal_length else Z
    zxx = -Z1.gradX.gradX
    zxy = -Z1.gradX.gradY
    zyx = -Z1.gradY.gradX
    zyy = -Z1.gradY.gradY

    if x_offset:
        zxx += x_offset.gradX
        zxy += x_offset.gradY
    if y_offset:
        zyx += y_offset.gradX
        zyy += y_offset.gradY

    return zxx(u, v), zxy(u, v), zyx(u, v), zyy(u, v)


def _focal_pupil_jacobian(
    u, v, Z, *,
    focal_length=None,
    x_offset=None, y_offset=None
):
    dxdu, dxdv, dydu, dydv = _pupil_focal_jacobian(
        u, v, Z,
        focal_length=focal_length,
        x_offset=x_offset, y_offset=y_offset
    )
    det = dxdu*dydv - dxdv*dydu
    dudx = dydv/det
    dudy = -dxdv/det
    dvdx = -dydu/det
    dvdy = dxdu/det
    return dudx, dudy, dvdx, dvdy


def _pixel_pupil_jacobian(
    u, v, Z, *,
    pixel_scale,
    focal_length=None,
    x_offset=None, y_offset=None,
):
    dudx, dudy, dvdx, dvdy = _focal_pupil_jacobian(
        u, v, Z,
        focal_length=focal_length,
        x_offset=x_offset, y_offset=y_offset
    )
    # Apply pixel scale to the Jacobian
    dudx *= pixel_scale
    dudy *= pixel_scale
    dvdx *= pixel_scale
    dvdy *= pixel_scale
    return dudx, dudy, dvdx, dvdy


def focal_to_pupil(
    x, y, *,
    Z=None, aberrations=None, R_outer=1.0, R_inner=0.0,
    focal_length=None,
    x_offset=None, y_offset=None,
    prefit_order=F2P_PREFIT_ORDER, maxiter=F2P_MAXITER, tol=F2P_TOL, strict=F2P_STRICT
):
    """Transform focal coordinates to pupil coordinates.

    Parameters
    ----------
    x, y : array of float
        Focal coordinates in meters.
    Z : galsim.zernike.Zernike, optional
        Aberrations in meters.
    aberrations : array of float, optional
        Aberrations in meters.
    R_outer, R_inner : float, optional
        Annulus outer and inner radii in meters.
    focal_length : float
        Focal length in meters.
    x_offset, y_offset : galsim.zernike.Zernike, optional
        Additional focal plane offsets (in meters) represented as Zernike
        series.
    prefit_order : int
        Order of prefit used to get good initial guesses for coordinate
        transformation.
    maxiter : int
        Number of Newton iterations to attempt before failing.
    tol : float
        Tolerance for successful coordinate transformation.
    strict: bool
        If True, then raise a RuntimeError if any coordinates could not be
        mapped.
        If False, then return NaN for unmappable coordinates.

    Returns
    -------
    u, v : array of float
        Pupil coordinates in meters.
    """
    if Z is None:
        Z = galsim.zernike.Zernike(
            aberrations, R_outer=R_outer, R_inner=R_inner
        )
    if focal_length is None:
        raise ValueError("Missing focal length")

    return _focal_to_pupil(
        x, y, Z,
        focal_length=focal_length,
        x_offset=x_offset, y_offset=y_offset,
        prefit_order=prefit_order,
        maxiter=maxiter,
        tol=tol, strict=strict
    )


@lru_cache(maxsize=10)
def _test_points(R_outer, R_inner):
    utest = np.linspace(-R_outer, R_outer, 10)
    utest, vtest = np.meshgrid(utest, utest)
    r2test = utest**2 + vtest**2
    w = r2test >= R_inner**2
    w &= r2test <= R_outer**2
    utest = utest[w]
    vtest = vtest[w]
    return utest, vtest


def _focal_to_pupil(
    x, y, Z, *,
    focal_length=None,
    x_offset=None, y_offset=None,
    prefit_order=F2P_PREFIT_ORDER, maxiter=F2P_MAXITER, tol=F2P_TOL,
    strict=F2P_STRICT
):
    Z1 = Z * focal_length if focal_length else Z
    utest, vtest = _test_points(Z1.R_outer, Z1.R_inner)
    xtest, ytest = _pupil_to_focal(
        utest, vtest, Z1,
        x_offset=x_offset, y_offset=y_offset
    )

    # Prefit
    jmax = (prefit_order+1)*(prefit_order+2)//2
    R_outer = np.max(np.hypot(xtest, ytest))
    a = galsim.zernike.zernikeBasis(jmax, xtest, ytest, R_outer=R_outer).T
    b = np.array([utest, vtest]).T
    r, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

    u = galsim.zernike.Zernike(r[:,0], R_outer=R_outer)(x, y)
    v = galsim.zernike.Zernike(r[:,1], R_outer=R_outer)(x, y)

    # Newton-Raphson iterations to invert pupil_to_focal
    x_current, y_current = _pupil_to_focal(
        u, v, Z1,
        x_offset=x_offset, y_offset=y_offset
    )
    dx = x_current - x
    dy = y_current - y
    dr2 = dx**2 + dy**2
    tol2 = tol * tol
    idx = np.nonzero(dr2 > tol2)[0]
    _active_sizes = [] if _f2p_stats is not None else None
    for i in range(maxiter):
        if idx.size == 0:
            break
        if _active_sizes is not None:
            _active_sizes.append(idx.size)
        ui, vi = u[idx], v[idx]
        xi, yi = x[idx], y[idx]
        dxi, dyi = dx[idx], dy[idx]
        dr2i = dr2[idx]
        dW2du2, dW2dudv, dW2dvdu, dW2dv2 = _pupil_focal_jacobian(
            ui, vi, Z1, x_offset=x_offset, y_offset=y_offset
        )
        det = (dW2du2*dW2dv2 - dW2dudv*dW2dvdu)
        dui = -(dW2dv2*dxi - dW2dvdu*dyi)/det
        dvi = -(-dW2dudv*dxi + dW2du2*dyi)/det
        # If xy miss distance increased, then decrease duv by
        # sqrt(distance ratio)
        uci = ui + dui
        vci = vi + dvi
        xci, yci = _pupil_to_focal(
            uci, vci, Z1, x_offset=x_offset, y_offset=y_offset
        )
        dxci = xci - xi
        dyci = yci - yi
        drc2i = dxci**2 + dyci**2
        w = drc2i > dr2i  # places where we're worse
        if np.any(w):
            alpha = np.maximum(0.001, (dr2i[w]/drc2i[w])**0.25)
            uci[w] = ui[w] + alpha*dui[w]
            vci[w] = vi[w] + alpha*dvi[w]
            xci[w], yci[w] = _pupil_to_focal(
                uci[w], vci[w], Z1,
                x_offset=x_offset, y_offset=y_offset
            )
            dxci[w] = xci[w] - xi[w]
            dyci[w] = yci[w] - yi[w]
            drc2i[w] = dxci[w]**2 + dyci[w]**2
        u[idx] = uci
        v[idx] = vci
        dx[idx] = dxci
        dy[idx] = dyci
        dr2[idx] = drc2i
        if idx.size > F2P_ACTIVE_SET_MIN:
            idx = np.nonzero(dr2 > tol2)[0]
    else:
        # If we failed to reach the desired tolerance, mark coordinate with a
        # NaN or if `strict`, raise a RuntimeError.
        # Diagnostic information
        intolerable = (np.abs(dx) > tol) | (np.abs(dy) > tol)
        wfail = np.nonzero(intolerable)[0]
        if strict:
            print(Z1)
            for _wi in wfail:
                print(x[_wi], y[_wi])
            raise RuntimeError("Cannot invert")
        u[wfail] = np.nan
        v[wfail] = np.nan
    if _f2p_stats is not None:
        _f2p_stats.append({
            'n_pixels':     len(x),
            'active_sizes': _active_sizes,
        })
    return u, v


def _gnomonic(u, v):
    """Transform gnomonic tangent plane coordinates to unit sphere coordinates.

    Parameters
    ----------
    u, v : array of float
        Gnomonic coordinates in radians.

    Returns
    -------
    alpha, beta, gamma : array of float
        3D coordinates on the unit sphere.
    """
    gamma = 1/np.sqrt(1.0 + u*u + v*v)
    alpha = u*gamma
    beta = v*gamma
    return alpha, beta, -gamma


def _rotxy(r, angle):
    """Rotate a 3D vector around the z-axis.

    Parameters
    ----------
    r : array of float
        3D vector to rotate.
    angle : float
        Angle in degrees to rotate the vector.

    Returns
    -------
    r_rot : array of float
        Rotated 3D vector.
    """
    sth, cth = np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))
    x = r[0]*cth + r[1]*sth
    y = -r[0]*sth + r[1]*cth
    z = r[2]
    return np.array([x, y, z])


@lru_cache
def __project_spider_vane(
    r0, v0, width, length, angle, thx, thy
):
    r0 = np.array(r0)
    v0 = np.array(v0)
    v0 /= np.linalg.norm(v0)

    # Find direction perp to v0 and (0, 0, 1)
    # This is the direction along which the width is defined.
    # Expand the cross product with (0, 0, 1) by hand for speed.
    norm = np.sqrt(v0[0]**2 + v0[1]**2)
    perp = np.array([v0[1], -v0[0], 0.0]) / norm

    # Compute the spider vane edges in 3D.
    centerline = np.array([r0 - v0*length/2, r0 + v0*length/2])
    edge1 = _rotxy(centerline.T-(perp*width/2)[:, None], angle)
    edge2 = _rotxy(centerline.T+(perp*width/2)[:, None], angle)

    # Now project the edges onto the entrance pupil (defined by z=0).
    vproj = _gnomonic(thx, thy)
    t1 = -edge1[2]/vproj[2]
    t2 = -edge2[2]/vproj[2]
    proj1 = (edge1 + np.outer(vproj, t1))
    proj2 = (edge2 + np.outer(vproj, t2))

    # Mean projected xy position.  I.e., ~center of the spider vane
    p1 = np.mean(proj1[:2], axis=1)
    p2 = np.mean(proj2[:2], axis=1)

    sth1 = proj1[1, 1] - proj1[1, 0]
    cth1 = proj1[0, 1] - proj1[0, 0]
    norm1 = np.sqrt(sth1**2 + cth1**2)
    sth1 /= norm1
    cth1 /= norm1

    sth2 = proj2[1, 1] - proj2[1, 0]
    cth2 = proj2[0, 1] - proj2[0, 0]
    norm2 = np.sqrt(sth2**2 + cth2**2)
    sth2 /= norm2
    cth2 /= norm2

    return p1[0], p1[1], sth1, cth1, p2[0], p2[1], sth2, cth2


def _project_spider_vane(
    r0, v0, width, length, angle, thx, thy
):
    """Project a 3D spider vane onto the entrance pupil.

    Parameters
    ----------
    r0 : array of float
        3D position of the spider vane center (meters).
    v0 : array of float
        3D direction of the spider vane (unitless).
    width : float
        Width of the spider vane in meters.
    length : float
        Length of the spider vane in meters.
    angle : float
        Z-axis rotation angle to apply in degrees.
    thx, thy : float
        Gnomonic tangent plane coordinates in radians along which to project the
        spider vane shadow.

    Returns
    -------
    p1x, p1y : float
        Projected ~center position of the first edge of the spider vane (meters).
    sth1, cth1 : float
        Projected sine and cosine of the angle of the first edge of the spider vane.
    p2x, p2y : float
        Projected ~center position of the second edge of the spider vane (meters).
    sth2, cth2 : float
        Projected sine and cosine of the angle of the second edge of the spider vane.
    """
    return __project_spider_vane(
        tuple(r0), tuple(v0), width, length, angle, thx, thy
    )


def strut_masked_fraction(
    x, y,
    u, v,
    length,
    p1x, p1y, sth1, cth1, # First edge point and angle
    p2x, p2y, sth2, cth2, # Second edge point and angle
    Z=None, aberrations=None, R_outer=1.0, R_inner=0.0,
    focal_length=None,
    x_offset=None, y_offset=None,
    pixel_scale=None,
):
    if Z is None:
        Z = galsim.zernike.Zernike(
            aberrations, R_outer=R_outer, R_inner=R_inner
        )
    Z1 = Z * focal_length if focal_length else Z

    if pixel_scale is None:
        raise ValueError("Missing pixel scale")

    dudx, dudy, dvdx, dvdy = _pixel_pupil_jacobian(
        u, v, Z1,
        pixel_scale=pixel_scale,
        focal_length=focal_length,
        x_offset=x_offset, y_offset=y_offset,
    )

    return _strut_masked_fraction(
        x, y, u, v, length,
        p1x, p1y, sth1, cth1, p2x, p2y, sth2, cth2,
        dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy
    )


def _pixel_frac(
    u0, v0, sth0, cth0, # Line in pupil coordinates
    u1, v1, # Pupil coordinates of pixels
    x1, y1, # Pixel coordinates of pixels
    dudx, dudy, dvdx, dvdy, # Jacobian of pupil to focal transform
):
    frac = np.empty_like(u1)
    if isinstance(u0, np.ndarray):
        pixel_frac(
            u0.ctypes.data, v0.ctypes.data,
            sth0.ctypes.data, cth0.ctypes.data,
            u1.ctypes.data, v1.ctypes.data,
            x1.ctypes.data, y1.ctypes.data,
            dudx.ctypes.data, dudy.ctypes.data,
            dvdx.ctypes.data, dvdy.ctypes.data,
            frac.ctypes.data, len(u1)
        )
    else:
        pixel_frac(
            u0, v0,
            sth0, cth0,
            u1.ctypes.data, v1.ctypes.data,
            x1.ctypes.data, y1.ctypes.data,
            dudx.ctypes.data, dudy.ctypes.data,
            dvdx.ctypes.data, dvdy.ctypes.data,
            frac.ctypes.data, len(u1)
        )
    return frac


def _strut_masked_fraction(
    x, y,
    u, v,
    length,
    u1, v1, sth1, cth1, # First edge point and angle
    u2, v2, sth2, cth2, # Second edge point and angle
    *,
    dudx, dudy, dvdx, dvdy,
):
    frac = np.empty_like(u)
    enclosed_strut(
        x.ctypes.data, y.ctypes.data,
        u.ctypes.data, v.ctypes.data,
        length,
        u1, v1, sth1, cth1,
        u2, v2, sth2, cth2,
        dudx.ctypes.data, dudy.ctypes.data, dvdx.ctypes.data, dvdy.ctypes.data,
        frac.ctypes.data, len(u)
    )
    return frac


def enclosed_fraction(
    x, y,
    u, v,
    u0, v0, radius, *,
    Z=None, aberrations=None, R_outer=1.0, R_inner=0.0,
    focal_length=None,
    x_offset=None, y_offset=None,
    pixel_scale=None,
):
    """Compute fraction of pixels enclosed by circles defined on the pupil.

    Parameters
    ----------
    x, y : array of float
        Pixel coordinates.
    u, v : array of float
        Pupil coordinates in meters.
    u0, v0 : float
        Pupil coordinates of circle center in meters.
    radius : float
        Circle radius in meters.
    Z : galsim.zernike.Zernike, optional
        Aberrations in meters.
    aberrations : array of float, optional
        Aberrations in meters.
    R_outer, R_inner : float, optional
        Annulus outer and inner radii in meters.
    focal_length : float
        Focal length in meters.
    x_offset, y_offset : galsim.zernike.Zernike, optional
        Additional focal plane offsets (in meters) represented as Zernike
        series.
    pixel_scale : float
        Pixel scale in meters.

    Returns
    -------
    enclosed : array of float, congruent to x or y
        Each pixel's enclosed fraction between 0, 1.
    """

    if Z is None:
        Z = galsim.zernike.Zernike(
            aberrations, R_outer=R_outer, R_inner=R_inner
        )
    if focal_length is None:
        raise ValueError("Missing focal length")
    if pixel_scale is None:
        raise ValueError("Missing pixel scale")

    Z1 = Z * focal_length if focal_length else Z

    dudx, dudy, dvdx, dvdy = _pixel_pupil_jacobian(
        u, v, Z1,
        pixel_scale=pixel_scale,
        x_offset=x_offset, y_offset=y_offset
    )

    return _enclosed_fraction(
        x, y, u, v, u0, v0, radius,
        dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy
    )


def _enclosed_strut_1(
    x, y, u, v,
    length,
    u1, v1, sth1, cth1,
    u2, v2, sth2, cth2,
    dudx, dudy,
    dvdx, dvdy
):
    # Center of the strut
    cu = 0.5 * (u1 + u2)
    cv = 0.5 * (v1 + v2)

    # Exclude points > length/2 from strut center
    du0 = u - cu
    dv0 = v - cv
    if (du0*du0 + dv0*dv0 >= (length/2)*(length/2)):
        return 0.0  # Outside the strut

    # Exclude points not close to either edge
    # Note this implies the strut is thin
    h1 = np.sqrt((dudx + dvdy)*(dudx + dvdy) + (dudy - dvdx)*(dudy - dvdx))
    h2 = np.sqrt((dudx - dvdy)*(dudx - dvdy) + (dudy + dvdx)*(dudy + dvdx))
    maxLinearScale = 0.5 * (h1 + h2)

    # Points close to edge1
    du1 = u - u1
    dv1 = v - v1
    d1 = np.abs(-du1*sth1 + dv1*cth1)
    wclose1 = d1 < 2*maxLinearScale

    # Points close to edge2
    du2 = u - u2
    dv2 = v - v2
    d2 = np.abs(-du2*sth2 + dv2*cth2)
    wclose2 = d2 < 2*maxLinearScale

    if not wclose1 and not wclose2:
        # Pixel is far from both edges.  Use signed perpendicular distances
        # to decide whether the pixel is fully inside (between the edges) or
        # fully outside.  The signed distances have opposite signs when the
        # pixel lies between the two edges and the same sign when outside.
        # This handles the case where the strut is wider than ~4 pixels,
        # i.e., when the "thin strut" approximation above does not hold.
        s1 = -du1*sth1 + dv1*cth1
        s2 = -du2*sth2 + dv2*cth2
        return 1.0 if s1 * s2 < 0 else 0.0

    frac = _pixel_frac_1(
        u1, v1, sth1, cth1,
        u, v,
        x, y,
        dudx, dudy,
        dvdx, dvdy
    )
    frac -= _pixel_frac_1(
        u2, v2, sth2, cth2,
        u, v,
        x, y,
        dudx, dudy,
        dvdx, dvdy
    )
    return frac


def _pixel_frac_1(
    u0, v0, sth0, cth0,
    u1, v1,
    x1, y1,
    dudx, dudy, dvdx, dvdy
):
    cph = cth0 * dvdy - sth0 * dudy
    sph = sth0 * dudx - cth0 * dvdx
    norm = np.sqrt(sph*sph + cph*cph)
    cph /= norm
    sph /= norm

    # That takes care of the initial orientation, but we need the transformed point too.
    det = dudx*dvdy - dvdx*dudy
    dxdu = dvdy/det
    dydu = -dvdx/det
    dxdv = -dudy/det
    dydv = dudx/det
    x0 = (u0-u1)*dxdu + (v0-v1)*dxdv + x1
    y0 = (u0-u1)*dydu + (v0-v1)*dydv + y1

    # express x0, y0 wrt x1, y1
    x0 = x0 - x1
    y0 = y0 - y1

    flip = False
    if cph < 0:
        cph = -cph
        x0 = -x0
        flip =  not flip
    if sph < 0:
        sph = -sph
        y0 = -y0
        flip =  not flip
    if sph > cph:
        sph, cph = cph, sph
        x0, y0 = y0, x0
        flip =  not flip

    right = (0.5 - x0) * sph/cph + y0 + 0.5  # wrt bottom
    left = (-0.5 - x0) * sph/cph + y0 + 0.5

    frac = 0.0

    if left > 1:
        frac = 1.0
    elif right >= 1:
        frac = 1.0 - 0.5 * cph / sph * (1 - left) * (1 - left)
    elif left > 0:
        frac = 0.5 * (left + right)
    elif right > 0:
        frac = 0.5 * cph / sph * right * right
    else:
        frac = 0.0

    return 1.0 - frac if flip else frac


def _enclosed_circle_1(
    x, y, u, v,
    u0, v0, radius,
    dudx, dudy, dvdx, dvdy,
):
    """
    Parameters
    ----------
    x, y : float
        Focal plane coordinates in meters.
    u, v : float
        Pupil coordinates in meters.
    u0, v0 : float
        Pupil coordinates of circle center in meters.
    radius : float
        Circle radius in meters.
    dudx, dudy, dvdx, dvdy : float
        Jacobian of pupil to focal transform in meters per pixel.
    """
    # Coords wrt circle center
    du = u - u0
    dv = v - v0

    # Determine points far from circle boundary
    drhosq = du*du + dv*dv
    h1 = np.sqrt((dudx + dvdy)*(dudx + dvdy) + (dudy - dvdx)*(dudy - dvdx))
    h2 = np.sqrt((dudx - dvdy)*(dudx - dvdy) + (dudy + dvdx)*(dudy + dvdx))
    maxLinearScale = 0.5 * (h1 + h2)
    rmin = radius - maxLinearScale
    rmax = radius + maxLinearScale
    if (drhosq < rmin**2):
        return 1.0
    if (drhosq > rmax**2):
        return 0.0

    norm = np.sqrt(drhosq)
    lineu = u0 + radius * du / norm
    linev = v0 + radius * dv / norm
    sth = -du / norm
    cth = dv / norm

    return _pixel_frac_1(
        lineu, linev, sth, cth,
        u, v, x, y,
        dudx, dudy,
        dvdx, dvdy
    )


def _enclosed_fraction(
    x, y,
    u, v,
    u0, v0, radius,
    *,
    dudx, dudy, dvdx, dvdy,
):
    frac = np.empty_like(u)
    enclosed_circle(
        x.ctypes.data, y.ctypes.data,
        u.ctypes.data, v.ctypes.data,
        u0, v0, radius,
        dudx.ctypes.data, dudy.ctypes.data,
        dvdx.ctypes.data, dvdy.ctypes.data,
        frac.ctypes.data, len(u)
    )
    return frac


class DonutTriangleFactory:
    """Build an annulus-clipped pupil triangle mesh for forward projection.

    This class intentionally focuses on mesh construction/diagnostics first.
    Projection and pixel accumulation are future milestones.
    """

    def __init__(
        self, *,
        R_outer=4.18, R_inner=2.5498,
        pupil_R_outer=None, pupil_R_inner=None,
        focal_length=10.31,
        pixel_scale=10e-6,
    ):
        self.R_outer = R_outer
        self.R_inner = R_inner
        self.pupil_R_outer = pupil_R_outer if pupil_R_outer is not None else R_outer
        self.pupil_R_inner = pupil_R_inner if pupil_R_inner is not None else R_inner * 0.9
        self.focal_length = focal_length
        self.pixel_scale = pixel_scale

    def build_annulus_mesh(
        self, *,
        nrad=18,
        naz=96,
        kfold=6,
        boundary_naz=720,
        dedup_tol=1e-12,
        debug=False,
        show_debug=True,
        plot_vertices=False,
    ):
        """Build a clipped annulus mesh using ring-stratified triangles.

        Boundary triangles are preserved instead of discarded, which keeps the
        annulus area accurate and gives a stable geometric baseline before any
        focal-plane projection work.
        """
        outer = float(self.pupil_R_outer)
        inner = float(max(0.0, self.pupil_R_inner))
        if inner >= outer:
            raise ValueError("pupil_R_inner must be smaller than pupil_R_outer")

        # Use explicit boundary rings plus smoothly interpolated interior rings.
        theta_count = int(boundary_naz)
        if theta_count < 12:
            theta_count = 12
        if theta_count % kfold:
            theta_count += kfold - (theta_count % kfold)
        thetas = np.linspace(0.0, 2.0*np.pi, theta_count, endpoint=False)

        n_layers = max(2, int(nrad))
        # Space layers approximately uniformly in area for better boundary fidelity.
        t = np.linspace(0.0, 1.0, n_layers)
        radii = np.sqrt(inner*inner + t*(outer*outer - inner*inner))

        rings = [np.column_stack([r*np.cos(thetas), r*np.sin(thetas)]) for r in radii]
        vertices = np.vstack(rings)
        triangles = []

        def vid(layer, j):
            return layer * theta_count + (j % theta_count)

        for layer in range(n_layers - 1):
            for j in range(theta_count):
                a = vid(layer, j)
                b = vid(layer + 1, j)
                c = vid(layer + 1, j + 1)
                d = vid(layer, j + 1)
                # Two triangles per quad, all oriented CCW.
                triangles.append([a, b, c])
                triangles.append([a, c, d])

        triangles = np.asarray(triangles, dtype=np.int32)
        tv = vertices[triangles]
        areas = 0.5 * np.abs(
            (tv[:, 1, 0] - tv[:, 0, 0]) * (tv[:, 2, 1] - tv[:, 0, 1])
            - (tv[:, 1, 1] - tv[:, 0, 1]) * (tv[:, 2, 0] - tv[:, 0, 0])
        )

        # Triangle categories for diagnostics: everything touching boundary rings
        # is treated as clipped; the rest are interior triangles.
        boundary_layers_mask = np.zeros(len(vertices), dtype=bool)
        boundary_layers_mask[:theta_count] = True
        boundary_layers_mask[-theta_count:] = True
        vertex_is_boundary = boundary_layers_mask[triangles]
        clipped_mask = np.any(vertex_is_boundary, axis=1)
        inside_mask = ~clipped_mask

        mesh = {
            'vertices': vertices,
            'triangles': triangles,
            'inside_triangles': int(np.sum(inside_mask)),
            'clipped_input_triangles': int(np.sum(clipped_mask)),
            'rejected_input_triangles': 0,
            'triangle_area_sum': float(np.sum(areas)),
        }

        if debug:
            fig, ax = self.plot_mesh_debug(
                mesh,
                inside_triangles=tv[inside_mask],
                clipped_input_triangles=tv[clipped_mask],
                rejected_input_triangles=[],
                show=show_debug,
                plot_vertices=plot_vertices,
            )
            mesh['debug_figure'] = fig
            mesh['debug_axes'] = ax

        return mesh

    @classmethod
    def _triangle_relation_to_circle(cls, tri, center, radius, keep_inside, tol=1e-12):
        """Classify triangle relative to circle region.

        Returns one of 'keep', 'discard', or 'partial'.
        """
        r = np.hypot(tri[:, 0] - center[0], tri[:, 1] - center[1])
        if keep_inside:
            if np.all(r <= radius + tol):
                return 'keep'
            if np.all(r >= radius - tol):
                return 'discard'
        else:
            if np.all(r >= radius - tol):
                return 'keep'
            if np.all(r <= radius + tol):
                return 'discard'
        return 'partial'

    @staticmethod
    def _circle_edge_intersections(p1, p2, center, radius):
        """Return 0, 1, or 2 intersection points of segment p1→p2 with a circle.

        Returns a list of (t, point) pairs sorted by t in [0, 1].
        """
        d = p2 - p1
        f = p1 - center
        a = float(np.dot(d, d))
        if a < 1e-30:
            return []
        b = 2.0 * float(np.dot(f, d))
        c = float(np.dot(f, f)) - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return []
        sqrt_disc = np.sqrt(max(disc, 0.0))
        results = []
        for sign in (-1.0, 1.0):
            t = (-b + sign * sqrt_disc) / (2.0 * a)
            if 0.0 <= t <= 1.0:
                results.append((t, p1 + t * d))
        results.sort(key=lambda x: x[0])
        return results

    @classmethod
    def _clip_triangle_to_circle(cls, tri, center, radius, keep_inside):
        """Exact circle-clipping of one triangle via Sutherland-Hodgman.

        The circle boundary is treated as a half-plane at each edge crossing:
        vertices are classified as inside/outside the circle, and intersection
        points are computed exactly.  The clipped polygon is then fan-
        triangulated.  This produces at most 5 output vertices (hence at most
        3 triangles) per call, regardless of triangle size.
        """
        rel = cls._triangle_relation_to_circle(tri, center, radius, keep_inside)
        if rel == 'keep':
            return [tri]
        if rel == 'discard':
            return []

        # Sutherland-Hodgman clipping against the circle.
        # "inside" means within the *kept* region.
        r = np.hypot(tri[:, 0] - center[0], tri[:, 1] - center[1])
        if keep_inside:
            inside = r <= radius
        else:
            inside = r >= radius

        poly = list(tri)  # list of 2-D points
        n = len(poly)
        out = []
        for i in range(n):
            s = poly[i]
            p = poly[(i + 1) % n]
            s_in = bool(inside[i])
            p_in = bool(inside[(i + 1) % n])

            # Compute all crossings on segment s→p
            crossings = cls._circle_edge_intersections(
                np.asarray(s, dtype=float),
                np.asarray(p, dtype=float),
                center, radius,
            )
            # Filter to actual transitions (entry/exit) based on kept region.
            # For keep_inside: crossing at t means we enter (t from outside→inside)
            # or exit (inside→outside).  Sutherland-Hodgman emits:
            #   s in  → emit s; if exit crossing, emit it
            #   s out → if entry crossing, emit it
            if s_in:
                out.append(s)
                # If p is outside, we exit the kept region: emit the first exit crossing
                if not p_in and crossings:
                    # The last crossing in direction s→p is the exit
                    out.append(crossings[-1][1])
            else:
                # s is outside; if p is inside, emit the last entry crossing
                if p_in and crossings:
                    out.append(crossings[0][1])
                # Edge can enter and re-exit the kept region entirely within segment
                elif not p_in and len(crossings) == 2:
                    out.append(crossings[0][1])
                    out.append(crossings[1][1])

        if len(out) < 3:
            return []

        # Fan-triangulate the clipped polygon from vertex 0
        tris = []
        for i in range(1, len(out) - 1):
            t = np.array([out[0], out[i], out[i + 1]], dtype=float)
            # Drop degenerate triangles
            area = 0.5 * abs(
                (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
                - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0])
            )
            if area > 1e-30:
                tris.append(t)
        return tris

    @staticmethod
    def _mesh_from_triangles(triangles, tol=1e-12):
        """Deduplicate vertices and create a mesh dictionary from triangle coordinates."""
        if len(triangles) == 0:
            return {
                'vertices': np.empty((0, 2), dtype=float),
                'triangles': np.empty((0, 3), dtype=np.int32),
                'triangle_area_sum': 0.0,
            }

        scale = 1.0 / tol
        vert_map = {}
        verts = []
        conn = []
        for tri in triangles:
            idx = []
            for p in tri:
                key = tuple(np.round(p * scale).astype(np.int64))
                j = vert_map.get(key)
                if j is None:
                    j = len(verts)
                    verts.append(np.array(p, dtype=float))
                    vert_map[key] = j
                idx.append(j)
            if len(set(idx)) == 3:
                conn.append(idx)

        verts = np.array(verts, dtype=float)
        conn = np.array(conn, dtype=np.int32)
        if len(conn):
            tv = verts[conn]
            areas = 0.5 * np.abs(
                (tv[:, 1, 0] - tv[:, 0, 0]) * (tv[:, 2, 1] - tv[:, 0, 1])
                - (tv[:, 1, 1] - tv[:, 0, 1]) * (tv[:, 2, 0] - tv[:, 0, 0])
            )
            area_sum = float(np.sum(areas))
        else:
            area_sum = 0.0

        return {
            'vertices': verts,
            'triangles': conn,
            'triangle_area_sum': area_sum,
        }

    def apply_circle_obscurations(
        self,
        mesh,
        *,
        mask_params,
        thx=0.0,
        thy=0.0,
        tol=1e-12,
        debug=False,
        show_debug=True,
        plot_vertices=False,
    ):
        """Apply Rubin-style circular obscurations to an annulus mesh.

        Spider_3D entries are ignored on purpose; this method handles only the
        circular obscuration terms in the YAML file.
        """
        thr = np.sqrt(thx*thx + thy*thy)
        thr_deg = np.rad2deg(thr)

        tri_coords = mesh['vertices'][mesh['triangles']]
        active_circles = []
        kept = tri_coords
        removed_count = 0
        clipped_count = 0

        for item, val in mask_params.items():
            if item == 'Spider_3D':
                continue
            for edge, edge_params in val.items():
                if thr_deg < edge_params['thetaMin'] or thr_deg > edge_params['thetaMax']:
                    continue

                radius = float(np.polyval(edge_params['radius'], thr_deg))
                center = float(np.polyval(edge_params['center'], thr_deg))
                cx = center * thx / thr if thr > 0 else 0.0
                cy = center * thy / thr if thr > 0 else 0.0
                keep_inside = bool(edge_params['clear'])
                active_circles.append({
                    'item': item,
                    'edge': edge,
                    'center': np.array([cx, cy], dtype=float),
                    'radius': radius,
                    'keep_inside': keep_inside,
                })

                next_tris = []
                for tri in kept:
                    rel = self._triangle_relation_to_circle(tri, np.array([cx, cy]), radius, keep_inside, tol=tol)
                    if rel == 'keep':
                        next_tris.append(tri)
                    elif rel == 'discard':
                        removed_count += 1
                    else:
                        clipped = self._clip_triangle_to_circle(
                            tri, np.array([cx, cy]), radius, keep_inside,
                        )
                        if len(clipped) == 0:
                            removed_count += 1
                        else:
                            clipped_count += 1
                            next_tris.extend(clipped)
                if len(next_tris) == 0:
                    kept = np.empty((0, 3, 2), dtype=float)
                else:
                    kept = np.array(next_tris, dtype=float)

        masked = self._mesh_from_triangles(kept, tol=tol)
        masked['kept_triangle_count'] = int(len(masked['triangles']))
        masked['removed_triangle_count'] = int(removed_count)
        masked['clipped_triangle_count'] = int(clipped_count)
        masked['active_circles'] = active_circles
        masked['field_angle_deg'] = float(thr_deg)
        masked['source_mesh'] = mesh

        if debug:
            fig, ax = self.plot_mesh_debug(
                masked,
                inside_triangles=kept,
                clipped_input_triangles=tri_coords[:0],
                rejected_input_triangles=tri_coords[:0],
                show=show_debug,
                plot_vertices=plot_vertices,
                circle_specs=active_circles,
            )
            masked['debug_figure'] = fig
            masked['debug_axes'] = ax

        return masked

    def plot_circle_obscuration_debug(
        self,
        mesh,
        masked_mesh=None,
        *,
        show=True,
        plot_vertices=False,
    ):
        """Plot the annulus mesh before/after circle obscuration clipping.

        Parameters
        ----------
        mesh : dict
            Source annulus mesh from build_annulus_mesh.
        masked_mesh : dict, optional
            Result from apply_circle_obscurations. If omitted, mesh is used.
        show : bool
            If True, call plt.show().
        plot_vertices : bool
            If True, overlay the mesh vertices.
        """
        if masked_mesh is None:
            masked_mesh = mesh

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(8, 8))
        th = np.linspace(0.0, 2.0*np.pi, 1000)
        ax.plot(
            self.pupil_R_outer*np.cos(th),
            self.pupil_R_outer*np.sin(th),
            color='k', linewidth=1.2,
        )
        if self.pupil_R_inner > 0:
            ax.plot(
                self.pupil_R_inner*np.cos(th),
                self.pupil_R_inner*np.sin(th),
                color='k', linewidth=1.0, linestyle='--',
            )

        def _add_mesh_lines(tris, color, linewidth, alpha):
            if len(tris) == 0:
                return
            tv = tris if tris.ndim == 3 else tris[mesh['triangles']]
            segs = np.concatenate([
                tv[:, [0, 1], :],
                tv[:, [1, 2], :],
                tv[:, [2, 0], :],
            ], axis=0)
            ax.add_collection(LineCollection(segs, colors=color, linewidths=linewidth, alpha=alpha))

        _add_mesh_lines(mesh['vertices'][mesh['triangles']], 'steelblue', 0.45, 0.55)
        if masked_mesh is not mesh:
            _add_mesh_lines(masked_mesh['vertices'][masked_mesh['triangles']], 'forestgreen', 0.8, 0.85)
        else:
            _add_mesh_lines(masked_mesh['vertices'][masked_mesh['triangles']], 'forestgreen', 0.8, 0.85)

        circle_specs = masked_mesh.get('active_circles', [])
        th_c = np.linspace(0.0, 2.0*np.pi, 400)
        for spec in circle_specs:
            c = spec['center']
            r = spec['radius']
            color = 'purple' if spec['keep_inside'] else 'tomato'
            linestyle = '--' if spec['keep_inside'] else ':'
            ax.plot(c[0] + r*np.cos(th_c), c[1] + r*np.sin(th_c), color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9)

        if plot_vertices:
            verts = mesh['vertices']
            if len(verts):
                ax.plot(verts[:, 0], verts[:, 1], '.', color='0.2', alpha=0.25, markersize=1.0)

        ax.set_aspect('equal')
        ax.set_xlabel('u (m)')
        ax.set_ylabel('v (m)')

        src_area = mesh.get('triangle_area_sum', 0.0)
        masked_area = masked_mesh.get('triangle_area_sum', 0.0)
        ax.set_title(
            'Circle Obscuration Debug\n'
            f"source area={src_area:.4f}, masked area={masked_area:.4f}, "
            f"active circles={len(circle_specs)}"
        )
        ax.legend(handles=[
            Line2D([0], [0], color='k', linestyle='-', linewidth=1.2),
            Line2D([0], [0], color='k', linestyle='--', linewidth=1.0),
            Line2D([0], [0], color='steelblue', linewidth=0.8),
            Line2D([0], [0], color='forestgreen', linewidth=0.8),
            Line2D([0], [0], color='purple', linestyle='--', linewidth=1.0),
            Line2D([0], [0], color='tomato', linestyle=':', linewidth=1.0),
        ], labels=[
            'Outer annulus boundary',
            'Inner annulus boundary',
            'Source annulus mesh',
            'Circle-clipped mesh',
            'Clear circle boundary',
            'Opaque circle boundary',
        ], loc='best')

        if show:
            plt.show()
        return fig, ax

    def plot_mesh_debug(
        self,
        mesh,
        *,
        inside_triangles=None,
        clipped_input_triangles=None,
        rejected_input_triangles=None,
        show=True,
        plot_vertices=False,
        circle_specs=None,
    ):
        """Plot annulus boundaries and triangle diagnostics."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        if inside_triangles is None:
            inside_triangles = []
        if clipped_input_triangles is None:
            clipped_input_triangles = []
        if rejected_input_triangles is None:
            rejected_input_triangles = []

        fig, ax = plt.subplots(figsize=(8, 8))
        th = np.linspace(0.0, 2.0 * np.pi, 1000)
        ou = self.pupil_R_outer * np.cos(th)
        ov = self.pupil_R_outer * np.sin(th)
        iu = self.pupil_R_inner * np.cos(th)
        iv = self.pupil_R_inner * np.sin(th)
        ax.plot(ou, ov, 'k-', linewidth=1.2)
        if self.pupil_R_inner > 0:
            ax.plot(iu, iv, 'k--', linewidth=1.2)

        if len(mesh['triangles']):
            tv = mesh['vertices'][mesh['triangles']]
            segs = np.concatenate([
                tv[:, [0, 1], :],
                tv[:, [1, 2], :],
                tv[:, [2, 0], :],
            ], axis=0)
            ax.add_collection(LineCollection(segs, colors='steelblue', linewidths=0.5, alpha=0.8))

        if plot_vertices:
            verts = mesh['vertices']
            if len(verts):
                ax.plot(verts[:, 0], verts[:, 1], '.', color='0.2', alpha=0.3, markersize=1.0)

        if circle_specs:
            th = np.linspace(0.0, 2.0*np.pi, 400)
            for spec in circle_specs:
                c = spec['center']
                r = spec['radius']
                color = 'purple' if spec['keep_inside'] else 'tomato'
                linestyle = '--' if spec['keep_inside'] else ':'
                ax.plot(c[0] + r*np.cos(th), c[1] + r*np.sin(th), color=color, linestyle=linestyle, linewidth=1.0, alpha=0.8)

        if len(inside_triangles):
            ins = np.array(inside_triangles)
            segs = np.concatenate([
                ins[:, [0, 1], :],
                ins[:, [1, 2], :],
                ins[:, [2, 0], :],
            ], axis=0)
            ax.add_collection(LineCollection(segs, colors='forestgreen', linewidths=0.5, alpha=0.35))

        if len(clipped_input_triangles):
            clp = np.array(clipped_input_triangles)
            segs = np.concatenate([
                clp[:, [0, 1], :],
                clp[:, [1, 2], :],
                clp[:, [2, 0], :],
            ], axis=0)
            ax.add_collection(LineCollection(segs, colors='goldenrod', linewidths=0.7, alpha=0.55))

        if len(rejected_input_triangles):
            rej = np.array(rejected_input_triangles)
            segs = np.concatenate([
                rej[:, [0, 1], :],
                rej[:, [1, 2], :],
                rej[:, [2, 0], :],
            ], axis=0)
            ax.add_collection(LineCollection(segs, colors='firebrick', linewidths=0.7, alpha=0.55))

        verts = mesh['vertices']
        if len(verts):
            ax.plot(verts[:, 0], verts[:, 1], '.', color='black', alpha=0.25, markersize=1.2)

        ax.set_aspect('equal')
        ax.set_xlabel('u (m)')
        ax.set_ylabel('v (m)')

        analytic_area = np.pi * (self.pupil_R_outer**2 - self.pupil_R_inner**2)
        rel_err = 0.0 if analytic_area == 0 else (mesh.get('triangle_area_sum', 0.0) - analytic_area) / analytic_area
        inside_count = mesh.get('inside_triangles', len(inside_triangles))
        clipped_count = mesh.get('clipped_input_triangles', len(clipped_input_triangles))
        rejected_count = mesh.get('rejected_input_triangles', len(rejected_input_triangles))
        ax.set_title(
            'Annulus Triangulation Debug\n'
            f"inside={inside_count}, clipped_in={clipped_count}, "
            f"rejected_in={rejected_count}, rel area err={rel_err:.3e}"
        )
        ax.legend(handles=[
            Line2D([0], [0], color='k', linestyle='-', linewidth=1.2),
            Line2D([0], [0], color='k', linestyle='--', linewidth=1.2),
            Line2D([0], [0], color='steelblue', linewidth=0.8),
            Line2D([0], [0], color='forestgreen', linewidth=0.8),
            Line2D([0], [0], color='goldenrod', linewidth=0.8),
            Line2D([0], [0], color='firebrick', linewidth=0.8),
            Line2D([0], [0], color='purple', linestyle='--', linewidth=1.0),
            Line2D([0], [0], color='tomato', linestyle=':', linewidth=1.0),
        ], labels=[
            'Outer annulus boundary',
            'Inner annulus boundary',
            'Output mesh triangles',
            'Inside input triangles',
            'Clipped input triangles',
            'Rejected input triangles',
            'Clear circle boundary',
            'Opaque circle boundary',
        ], loc='best')

        if show:
            plt.show()
        return fig, ax

    def image(
        self,
        mesh,
        *,
        Z=None,
        aberrations=None,
        focal_length=10.31,
        pixel_scale=10e-6,
        npix=181,
    ):
        """Accumulate mesh triangles onto pixels via forward mapping.

        Each triangle carries flux proportional to its pupil-space area.
        That flux is apportioned into output pixels according to the overlap
        area between the projected triangle and each pixel square.

        Parameters
        ----------
        mesh : dict
            Output from build_annulus_mesh or apply_circle_obscurations.
        Z : galsim.zernike.Zernike, optional
            Aberrations in meters.
        aberrations : array of float, optional
            Aberrations in meters.
        focal_length : float
            Focal length in meters. Default 10.31 (LSST).
        pixel_scale : float
            Pixel scale in meters. Default 10e-6 (10 µm).
        npix : int
            Number of pixels on each side. Must be odd.

        Returns
        -------
        image : ndarray, shape (npix, npix)
            Accumulated flux image.
        """
        if npix % 2 == 0:
            raise ValueError(f"Argument npix={npix} must be odd.")

        if Z is None:
            Z = galsim.zernike.Zernike(
                aberrations, R_outer=self.pupil_R_outer, R_inner=self.pupil_R_inner
            )

        no2 = (npix - 1) // 2

        verts_uv = mesh["vertices"]   # (N, 2) in meters
        tris     = mesh["triangles"]  # (M, 3) indices
        tri_uv   = verts_uv[tris]     # (M, 3, 2) in meters

        # Pupil-space areas (flux per triangle)
        pupil_areas = _shoelace_area(tri_uv)  # m²

        # Forward-map all vertices to centred pixel coordinates
        u_all = verts_uv[:, 0]
        v_all = verts_uv[:, 1]

        Z1 = Z * focal_length
        xf = -Z1.gradX(u_all, v_all)
        yf = -Z1.gradY(u_all, v_all)

        xp = xf / pixel_scale
        yp = yf / pixel_scale

        verts_px = np.column_stack([xp, yp])  # (N, 2)
        tri_px   = verts_px[tris]             # (M, 3, 2)

        # Projected areas in pixel²
        proj_areas = _shoelace_area(tri_px)

        # Guard against degenerate projected triangles
        valid = proj_areas > 0.0

        # Accumulate onto image
        image = np.zeros((npix, npix), dtype=np.float64)

        for k in np.where(valid)[0]:
            tri_v   = tri_px[k]        # (3, 2) in centred pixel coords
            flux    = pupil_areas[k]
            aproj   = proj_areas[k]

            # Bounding box in centred pixel integer indices
            xmin = int(np.floor(tri_v[:, 0].min() + 0.5))
            xmax = int(np.floor(tri_v[:, 0].max() + 0.5))
            ymin = int(np.floor(tri_v[:, 1].min() + 0.5))
            ymax = int(np.floor(tri_v[:, 1].max() + 0.5))

            # Clip to image bounds
            xmin = max(xmin, -no2)
            xmax = min(xmax,  no2)
            ymin = max(ymin, -no2)
            ymax = min(ymax,  no2)

            for iy in range(ymin, ymax + 1):
                for ix in range(xmin, xmax + 1):
                    area = _clip_area(tri_v, ix, iy)
                    if area > 0.0:
                        image[iy + no2, ix + no2] += flux * area / aproj

        return image


class DonutFactory:
    """Build and render geometric donut (and spot) images for a given telescope.

    Parameters
    ----------
    R_outer : float
        Zernike normalization radius in meters.
    R_inner : float
        Zernike normalization inner radius in meters.
    pupil_R_outer : float, optional
        Physical entrance pupil outer radius in meters.  Used for pixel
        selection and primary mirror clip.  Defaults to R_outer.
    pupil_R_inner : float, optional
        Physical entrance pupil inner radius in meters.  Used for inner
        obscuration early exclusion and flux normalization.
        Defaults to ``R_inner * 0.9``.
    mask_params : dict
        Nested dictionary containing the mask model. See the notes below
        for details on the format.
    spider_angle: float, optional
        Additional rotation for spider struts around optic axis in degrees.  If None,
        then don't model the spider shadows.
    focal_length : float
        Focal length in meters.
    pixel_scale : float
        Pixel scale in meters.
    bandpass_filter : string, optional
        Bandpass filter name for AOI-dependent throughput correction.
        Choose from: ['u','g','r','i','z','y']. If None (default), no
        throughput correction is applied.
    stellar_Tbb : float, optional
        Blackbody temperature in Kelvin used to select the throughput
        lookup. Must be in range 4000-10000 in steps of 200. Default 6000.
    airmass : float, optional
        Airmass used to select the throughput lookup. Must be in range
        1.0-2.5 in steps of 0.1. Default 1.5.

    Notes
    -----
    The mask_params dictionary is a nested dictionary that specifies the
    mask model. Each top-level item in the dictionary (except for
    `Spider_3D`, see below) can have any number of edges (usually "outer"
    and/or "inner"). Each edge is modeled as a circle in pupil space. For
    each of these edges, there is a minimum and maximum field angle where
    the edge needs to be computed, as well as polynomial coefficients for
    calculating the center and radius of the circle. These coefficients
    are meant to be used with np.polyval. Each edge also has a "clear"
    bool which indicates whether the interior of the circle is clear or
    opaque.

    Spider struts are modeled as 2D rectangles situated in 3D space.
    The Spider_3d item is a list of dictionaries, each containing the
    following keys:
      - 'r0': [float, float, float]
        3D position of the spider vane center in meters.  The coordinate
        system is such that the Z-axis is the optic axis, and the origin
        is center of the entrance pupil.
      - 'v0': [float, float, float]
        3D direction of the spider vane in meters.
      - 'width': float
        Width of the spider vane in meters.  The width is measured
        perpendicular to both the optic axis and the spider vane
        direction.
      - 'length': float
        Approximate length of the spider vane in meters.  We assume that
        the ends of the spider struts are obscured by other components so
        detailed modeling is not necessary.
      - 'angle': float
        Additional Z-axis rotation angle to apply in degrees.

    An obscuration dictionary containing both circular and spider strut
    components would look something like:

    {
        item1:
            edge:
                clear: bool
                thetaMin: float (degrees)
                thetaMax: float (degrees)
                center: [float,] (meters)
                radius: [float,] (meters)
        item2:
            edge2:
                clear: bool
                thetaMin: float (degrees)
                thetaMax: float (degrees)
                center: [float,] (meters)
                radius: [float,] (meters)
        Spider_3D:
            -
                r0: [float, float, float]  (meters)
                v0: [float, float, float]  (meters)
                width: float  (meters)
                length: float  (meters)
                angle: float  (degrees)
            ...
    }
    """
    def __init__(
        self, *,
        R_outer=4.18, R_inner=2.5498,
        pupil_R_outer=None, pupil_R_inner=None,
        mask_params=None,
        spider_angle=None,
        focal_length=10.31,
        pixel_scale=10e-6,
        bandpass_filter=None,
        stellar_Tbb=6000,
        airmass=1.5,
    ):
        self.R_outer = R_outer
        self.R_inner = R_inner
        self.pupil_R_outer = pupil_R_outer if pupil_R_outer is not None else R_outer
        self.pupil_R_inner = pupil_R_inner if pupil_R_inner is not None else R_inner * 0.9
        self.mask_params = mask_params
        self.spider_angle = spider_angle
        self.focal_length = focal_length
        self.pixel_scale = pixel_scale
        self.bandpass_filter = bandpass_filter
        self.stellar_Tbb = stellar_Tbb
        self.airmass = airmass
        if self.bandpass_filter is not None:
            self.thruput_by_aoi = self._load_thruput_by_aoi(
                bandpass_filter, stellar_Tbb, airmass
            )

    def _load_thruput_by_aoi(self, bandpass_filter, stellar_Tbb, airmass):
        """Load and bilinearly interpolate the AOI-dependent throughput curve.

        The JSON table is sampled on a discrete (Tbb, airmass) grid.  Input
        values are clamped to the available range and then bilinearly
        interpolated across the four surrounding grid points so that
        off-grid values (e.g. airmass=1.49999) work without error.
        """
        aoi_dep_file = os.path.join(
            os.path.dirname(__file__), 'data', 'Tbb_airmass_aoi_dep_integrals.json'
        )
        with open(aoi_dep_file, mode='r') as f:
            catalog = json.load(f)[bandpass_filter]

        # Build float→JSON-string key maps (airmass keys are not uniform:
        # e.g. '1' and '2' rather than '1.0' and '2.0').
        tbb_key_map = {float(k): k for k in catalog.keys()}
        tbb_keys    = sorted(tbb_key_map.keys())
        am_key_map  = {float(k): k for k in catalog[tbb_key_map[tbb_keys[0]]].keys()}
        am_keys     = sorted(am_key_map.keys())

        # Clamp inputs to the available grid range.
        tbb_val = float(np.clip(stellar_Tbb, tbb_keys[0], tbb_keys[-1]))
        am_val  = float(np.clip(airmass,     am_keys[0],  am_keys[-1]))

        def _bounds(val, keys):
            """Return (lo, hi, t) bracketing val; t is the fractional weight toward hi."""
            if val <= keys[0]:  return keys[0], keys[0], 0.0
            if val >= keys[-1]: return keys[-1], keys[-1], 0.0
            for lo, hi in zip(keys, keys[1:]):
                if lo <= val <= hi:
                    return lo, hi, (val - lo) / (hi - lo)
            return keys[-1], keys[-1], 0.0

        tbb_lo, tbb_hi, tbb_t = _bounds(tbb_val, tbb_keys)
        am_lo,  am_hi,  am_t  = _bounds(am_val,  am_keys)

        def _load(tbb, am):
            """Return (aoi_array, throughput_array) for one grid point."""
            data = catalog[tbb_key_map[tbb]][am_key_map[am]]
            lut = sorted(
                ({'aoi': float(k), 'thruput': v} for k, v in data.items() if k != '_comment'),
                key=lambda x: x['aoi']
            )
            return (
                np.array([e['aoi']     for e in lut]),
                np.array([e['thruput'] for e in lut]),
            )

        # Bilinear interpolation across the four surrounding grid points.
        aoi_grid, t00 = _load(tbb_lo, am_lo)
        _,        t01 = _load(tbb_lo, am_hi)
        _,        t10 = _load(tbb_hi, am_lo)
        _,        t11 = _load(tbb_hi, am_hi)

        return {
            'aoi':   aoi_grid,
            'value': (
                (1-tbb_t)*(1-am_t)*t00 +
                (1-tbb_t)*am_t    *t01 +
                tbb_t    *(1-am_t)*t10 +
                tbb_t    *am_t    *t11
            ),
        }

    def image(
        self, *,
        Z=None, aberrations=None,
        x_offset=None, y_offset=None,
        thx=0, thy=0,
        npix=181,
        prefit_order=F2P_PREFIT_ORDER, maxiter=F2P_MAXITER, tol=F2P_TOL, strict=F2P_STRICT,
        debug=False
    ):
        """Compute aberrated donut image.

        Parameters
        ----------
        Z : galsim.zernike.Zernike, optional
            Aberrations in meters.
        aberrations : array of float, optional
            Aberrations in meters.
        x_offset, y_offset : galsim.zernike.Zernike, optional
            Additional focal plane offsets (in meters) represented as Zernike
            series.
        thx, thy : float
            Field angles in radians.
        npix : int
            Number of pixels along image edge.  Must be odd.
        prefit_order : int
            Order of prefit used to get good initial guesses for focal-to-pupil
            coordinate transformation.
        maxiter : int
            Number of Newton iterations to attempt for focal-to-pupil
            coordinate transformation before failing.
        tol : float
            Tolerance for successful focal-to-pupil coordinate transformation.
        strict: bool
            If True, then raise a RuntimeError if any failed focal-to-pupil
            transformations occurred.
            If False, then set image to zero at failed coordinates.
        debug : bool
            If True, show a diagnostic plot of the projected pupil grid and
            pixel classification (valid pixels in blue, caustic/failed pixels
            in red).

        Returns
        -------
        img : array of float
            Donut image.
        """
        if npix%2 == 0:
            raise ValueError(f"Argument npix={npix} must be odd.")
        no2 = (npix-1)//2
        if Z is None:
            Z = galsim.zernike.Zernike(
                aberrations, R_outer=self.R_outer, R_inner=self.R_inner
            )
        Z1 = Z*self.focal_length

        # Get good pixels by projecting entrance pupil polygon onto pixels.
        ph = np.linspace(0, 2*np.pi, 1000, endpoint=True)
        u, v = self.pupil_R_outer*np.cos(ph), self.pupil_R_outer*np.sin(ph)
        x, y = _pupil_to_focal(
            u, v, Z1, x_offset=x_offset, y_offset=y_offset
        )

        xp = x/self.pixel_scale
        yp = y/self.pixel_scale

        xgrid = np.arange(-no2-0.5, no2+1.5)  # pixel corners
        corners = np.empty((len(xgrid), len(xgrid)), dtype=bool)
        poly_grid_contains(
            xp.ctypes.data, yp.ctypes.data, len(xp),
            xgrid.ctypes.data, xgrid.ctypes.data, corners.ctypes.data,
            len(xgrid), len(xgrid)
        )
        contained = np.array(corners[1:,1:]) # Be sure to make a copy!
        contained |= corners[:-1,1:]
        contained |= corners[1:,:-1]
        contained |= corners[:-1,:-1]

        # Exclude pixels fully inside the inner obscuration before calling
        # _focal_to_pupil (saves Newton iterations, no accuracy loss).
        if self.pupil_R_inner > 0:
            _R_inner_eff = self.pupil_R_inner - 0.02
            u_inner = _R_inner_eff*np.cos(ph)
            v_inner = _R_inner_eff*np.sin(ph)
            x_inner, y_inner = _pupil_to_focal(
                u_inner, v_inner, Z1, x_offset=x_offset, y_offset=y_offset
            )
            xp_inner = x_inner/self.pixel_scale
            yp_inner = y_inner/self.pixel_scale
            corners_in = np.empty((len(xgrid), len(xgrid)), dtype=bool)
            poly_grid_contains(
                xp_inner.ctypes.data, yp_inner.ctypes.data, len(xp_inner),
                xgrid.ctypes.data, xgrid.ctypes.data, corners_in.ctypes.data,
                len(xgrid), len(xgrid)
            )
            fully_obscured = (corners_in[1:,1:] & corners_in[:-1,1:] &
                              corners_in[1:,:-1] & corners_in[:-1,:-1])
            contained &= ~fully_obscured

        ypix, xpix = np.nonzero(contained)
        x = (xpix.astype(float) - no2)*self.pixel_scale # meters
        y = (ypix.astype(float) - no2)*self.pixel_scale

        # Now invert to get pixel centers projected on pupil
        u, v = _focal_to_pupil(
            x, y, Z1,
            x_offset=x_offset, y_offset=y_offset,
            prefit_order=prefit_order, maxiter=maxiter, tol=tol, strict=strict
        )

        wfail = np.where(np.isnan(u))[0]

        if debug:
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Rectangle
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(xp, yp, 'k-')
            _x_ib, _y_ib = _pupil_to_focal(
                self.pupil_R_inner*np.cos(ph), self.pupil_R_inner*np.sin(ph), Z1,
                x_offset=x_offset, y_offset=y_offset
            )
            ax.plot(_x_ib/self.pixel_scale, _y_ib/self.pixel_scale, 'k--')
            _nrings, _nspokes, _npts = 80, 160, 2000
            _margin = 0.1 * self.pupil_R_outer
            _ring_radii = np.linspace(self.pupil_R_inner - _margin, self.pupil_R_outer + _margin, _nrings + 2)[1:-1]
            _th = np.linspace(0, 2*np.pi, _npts)
            for _r in _ring_radii:
                if _r < self.pupil_R_inner:
                    _color = 'mediumpurple'
                elif _r <= self.pupil_R_outer:
                    _color = 'steelblue'
                else:
                    _color = 'goldenrod'
                _xr, _yr = _pupil_to_focal(
                    _r*np.cos(_th), _r*np.sin(_th), Z1,
                    x_offset=x_offset, y_offset=y_offset
                )
                ax.plot(_xr/self.pixel_scale, _yr/self.pixel_scale,
                        color=_color, alpha=0.5, linewidth=0.6)
            _r_hole = np.linspace(self.pupil_R_inner - _margin, self.pupil_R_inner, 10)
            _r_ann  = np.linspace(self.pupil_R_inner, self.pupil_R_outer, 40)
            _r_out  = np.linspace(self.pupil_R_outer, self.pupil_R_outer + _margin, 10)
            for _angle in np.linspace(0, 2*np.pi, _nspokes, endpoint=False):
                _ca, _sa = np.cos(_angle), np.sin(_angle)
                for _rseg, _col in [(_r_hole, 'mediumpurple'), (_r_ann, 'steelblue'), (_r_out, 'goldenrod')]:
                    _xs, _ys = _pupil_to_focal(
                        _rseg*_ca, _rseg*_sa, Z1,
                        x_offset=x_offset, y_offset=y_offset
                    )
                    ax.plot(_xs/self.pixel_scale, _ys/self.pixel_scale,
                            color=_col, alpha=0.5, linewidth=0.6)
            _is_fail = np.zeros(len(x), dtype=bool)
            _is_fail[wfail] = True
            _xpc = x / self.pixel_scale
            _ypc = y / self.pixel_scale
            from matplotlib.path import Path as MplPath
            _inner_path = MplPath(np.column_stack([_x_ib/self.pixel_scale, _y_ib/self.pixel_scale]))
            _in_inner = _inner_path.contains_points(np.column_stack([_xpc, _ypc]))
            _is_good = ~_is_fail & ~_in_inner
            _good_patches = [Rectangle((xi-0.5, yi-0.5), 1, 1) for xi, yi, g in zip(_xpc, _ypc, _is_good) if g]
            _fail_patches = [Rectangle((xi-0.5, yi-0.5), 1, 1) for xi, yi, f in zip(_xpc, _ypc, _is_fail) if f]
            ax.add_collection(PatchCollection(
                _good_patches, facecolor='lightblue', alpha=0.6, edgecolor='steelblue', linewidth=0.5
            ))
            if _fail_patches:
                ax.add_collection(PatchCollection(
                    _fail_patches, facecolor='red', alpha=0.8, edgecolor='darkred', linewidth=0.5
                ))
            ax.set_xlabel('x (pix)')
            ax.set_ylabel('y (pix)')
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0], [0], color='k',            linewidth=1.0),
                Line2D([0], [0], color='k',            linewidth=1.0, linestyle='--'),
                Line2D([0], [0], color='mediumpurple', linewidth=0.6),
                Line2D([0], [0], color='steelblue',    linewidth=0.6),
                Line2D([0], [0], color='goldenrod',    linewidth=0.6),
                plt.Rectangle((0,0), 1, 1, fc='lightblue', ec='steelblue'),
                plt.Rectangle((0,0), 1, 1, fc='red',       ec='darkred'),
            ], labels=[
                'Outer pupil boundary',
                'Inner pupil boundary',
                'Grid (inside inner)',
                'Grid (annulus)',
                'Grid (outside outer)',
                'Pixels',
                'Caustic pixels',
            ])
            ax.set_aspect('equal')
            ax.set_title('Debug plot: pixel centers projected onto pupil')
            plt.show()

        # Any pixels where we failed to find the pupil coordinate we'll just
        # leave as zero.
        wgood = ~np.isnan(u)
        u = u[wgood]
        v = v[wgood]
        x = x[wgood]
        y = y[wgood]
        xpix = xpix[wgood]
        ypix = ypix[wgood]

        img = np.zeros((npix, npix))

        # Compute jacobian just once
        dudx, dudy, dvdx, dvdy = _pixel_pupil_jacobian(
            u, v, Z1,
            pixel_scale=self.pixel_scale,
            x_offset=x_offset, y_offset=y_offset
        )

        # Always clip out the primary mirror outer diameter
        f = _enclosed_fraction(
            x, y, u, v,
            0.0, 0.0, self.pupil_R_outer,
            dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy
        )

        # Clip out other obscurations as requested
        w = np.nonzero(f)[0]
        if self.mask_params is not None:
            thr = np.sqrt(thx*thx + thy*thy)
            thr_deg = np.rad2deg(thr)
            for item, val in self.mask_params.items():
                if item == "Spider_3D":
                    if self.spider_angle is None:
                        continue
                    for vane in val:
                        p1x, p1y, sth1, cth1, p2x, p2y, sth2, cth2 = _project_spider_vane(
                            vane["r0"], vane["v0"],
                            vane["width"], vane["length"],
                            vane["angle"]+self.spider_angle, thx, thy
                        )
                        enc = _strut_masked_fraction(
                            x[w], y[w],
                            u[w], v[w],
                            vane["length"],
                            p1x, p1y, sth1, cth1,
                            p2x, p2y, sth2, cth2,
                            dudx=dudx[w], dudy=dudy[w],
                            dvdx=dvdx[w], dvdy=dvdy[w]
                        )
                        f[w] = np.minimum(f[w], 1-enc)
                else:
                    for edge, edge_params in val.items():
                        if not np.any(w):
                            break
                        if thr_deg < edge_params["thetaMin"] or thr_deg > edge_params["thetaMax"]:
                            continue

                        radius = np.polyval(edge_params["radius"], thr_deg)
                        center = np.polyval(edge_params["center"], thr_deg)
                        cx = center*thx/thr if thr > 0 else 0
                        cy = center*thy/thr if thr > 0 else 0

                        enc = _enclosed_fraction(
                            x[w], y[w], u[w], v[w],
                            cx, cy, radius,
                            dudx=dudx[w], dudy=dudy[w], dvdx=dvdx[w], dvdy=dvdy[w]
                        )
                        if edge_params["clear"]:
                            f[w] = np.minimum(f[w], enc)
                        else:
                            f[w] = np.minimum(f[w], 1-enc)

                        w = np.nonzero(f)[0]
                if not np.any(w):
                    break

        # pixel pupil-to-focal area ratio
        # Negative hessian almost certainly means there's a caustic, but we'll
        # leave that analysis to a separate function.  Using the absolute value
        # of the Hessian means at least one ray path to an affected pixel gets
        # to contribute to the illumination, which is the behavior we want when
        # we're being sloppy.
        Fx = -Z1.gradX
        Fy = -Z1.gradY
        if x_offset:
            Fx += x_offset
        if y_offset:
            Fy += y_offset

        # # The Zernike math directly below is more elegant, but it turns out that
        # # forming the products and sums _after_ evaluating is usually more efficient.
        # inv_sb = Fx.gradX * Fy.gradY - Fx.gradY * Fy.gradX
        # f[w] /= np.abs(inv_sb(u[w], v[w]))

        uw = u[w]
        vw = v[w]
        inv_sb = Fx.gradX(uw, vw)*Fy.gradY(uw, vw) - Fx.gradY(uw, vw)*Fy.gradX(uw, vw)
        f[w] /= np.abs(inv_sb)

        # this is where any surface brigthness modification imparted by bandpass shifts should be
        # placed, if specified:
        if self.bandpass_filter is not None:
            aoi_proxy = np.rad2deg(np.atan(np.sqrt(uw**2+vw**2)/self.focal_length))
            # now interpolate using self.thruput_by_aoi
            tput = np.interp(aoi_proxy,self.thruput_by_aoi['aoi'],self.thruput_by_aoi['value'])
            f[w] *= tput

        img[ypix, xpix] = f
        img /= np.pi * (self.pupil_R_outer**2 - self.pupil_R_inner**2) / self.pixel_scale**2
        return img

    def is_caustic(
        self, *,
        Z=None, aberrations=None,
        x_offset=None, y_offset=None,
        nrad=50, naz=100
    ):
        """Check if given aberration introduces a caustic.

        This method is approximate.  It checks for the presence of a caustic by
        projecting concentric circles from the pupil to focal plane and then
        looking for intersections of the circles.  That ought to be sufficient
        in the limit of infinite sample radii and azimuths, but will be somewhat
        imperfect for finite values.  It also checks the entire annular pupil,
        including any bits that are vignetted.

        Parameters
        ----------
        Z : galsim.zernike.Zernike, optional
            Aberrations in meters.
        aberrations : array of float, optional
            Aberrations in meters.
        x_offset, y_offset : galsim.zernike.Zernike, optional
            Additional focal plane offsets (in meters) represented as Zernike
            series.
        nrad : int
            Number of radii to check between R_inner and R_outer.
        naz : int
            Number of points around each test circle.

        Returns
        -------
        is_caustic : bool
            True if any projected circles intersect.
        """
        from batoid import ObscPolygon
        if Z is None:
            Z = galsim.zernike.Zernike(
                aberrations, R_outer=self.R_outer, R_inner=self.R_inner
            )
        Z1 = Z*self.focal_length

        # Project concentric circles from pupil to focal, and then see if any
        # of them intersect.  Outer radii are more likely to have a caustic, so
        # start with them and short-circuit if a caustic is found.
        radii = np.linspace(self.R_outer, self.R_inner, nrad)
        th = np.linspace(0, 2*np.pi, naz)
        uu, vv = np.cos(th), np.sin(th)

        u0 = uu * radii[0]
        v0 = vv * radii[0]
        x0, y0 = _pupil_to_focal(
            u0, v0, Z1,
            x_offset=x_offset, y_offset=y_offset
        )

        for radius in radii[1:]:
            u1 = uu * radius
            v1 = vv * radius
            x1, y1 = _pupil_to_focal(
                u1, v1, Z1,
                x_offset=x_offset, y_offset=y_offset
            )
            # Check that inner circle is contained in outer circle
            circle = ObscPolygon(x0, y0)
            if np.any(~circle.contains(x1, y1)):
                return True
            x0, y0 = x1, y1
        else:
            return False

    def spots(
        self, *,
        Z=None, aberrations=None,
        x_offset=None, y_offset=None,
        thx=0, thy=0,
        nrad=5, naz=None,
    ):
        """Compute aberrated spot diagram.

        Parameters
        ----------
        Z : galsim.zernike.Zernike, optional
            Aberrations in meters.
        aberrations : array of float, optional
            Aberrations in meters.
        x_offset, y_offset : galsim.zernike.Zernike, optional
            Additional focal plane offsets (in meters) represented as Zernike
            series.
        thx, thy : float
            Field angles in radians.
        nrad : int
            Number of pupil radii to use between R_inner and R_outer.
        naz : int, optional
            Approximate number of azimuthal angles to use along the outer most radius.
            See hexapolar for details.

        Returns
        -------
        x_spots, y_spots, w_spots : array of float
            Focal plane coordinates of spots and weights.
        """
        if Z is None:
            Z = galsim.zernike.Zernike(
                aberrations, R_outer=self.R_outer, R_inner=self.R_inner
            )
        Z1 = Z*self.focal_length

        u, v = hexapolar(
            outer=self.R_outer, inner=self.R_inner, nrad=nrad, naz=naz)
        w = np.ones_like(u, dtype=bool)

        if self.mask_params is not None:
            thr = np.sqrt(thx*thx + thy*thy)
            thr_deg = np.rad2deg(thr)
            for item, val in self.mask_params.items():
                if item == "Spider_3D":
                    if self.spider_angle is None:
                        continue
                    for vane in val:
                        p1x, p1y, sth1, cth1, p2x, p2y, sth2, cth2 = _project_spider_vane(
                            vane["r0"], vane["v0"],
                            vane["width"], vane["length"],
                            vane["angle"]+self.spider_angle, thx, thy
                        )
                        cu = 0.5 * (p1x + p2x)
                        cv = 0.5 * (p1y + p2y)
                        half_len = vane["length"] / 2
                        du = u - cu
                        dv = v - cv
                        near = du*du + dv*dv < half_len*half_len
                        left1 = cth1*(v - p1y) - sth1*(u - p1x) > 0
                        left2 = cth2*(v - p2y) - sth2*(u - p2x) > 0
                        w[near & ~left1 & left2] = False
                else:
                    for edge, edge_params in val.items():
                        if thr_deg < edge_params["thetaMin"] or thr_deg > edge_params["thetaMax"]:
                            continue

                        radius = np.polyval(edge_params["radius"], thr_deg)
                        center = np.polyval(edge_params["center"], thr_deg)
                        cx = center*thx/thr if thr > 0 else 0
                        cy = center*thy/thr if thr > 0 else 0

                        r = np.hypot(u-cx, v-cy)
                        if edge_params["clear"]:
                            w[r > radius] = False
                        else:
                            w[r < radius] = False
        x, y = _pupil_to_focal(
            u, v, Z1,
            x_offset=x_offset, y_offset=y_offset
        )
        return x, y, w

    def spot_image(
        self, *,
        Z=None, aberrations=None,
        x_offset=None, y_offset=None,
        thx=0, thy=0,
        nrad=5, naz=None,
        npix=15,
        gq_kwargs=None,
    ):
        """Compute image from spots.

        Parameters
        ----------
        Z : galsim.zernike.Zernike, optional
            Aberrations in meters.
        aberrations : array of float, optional
            Aberrations in meters.
        x_offset, y_offset : galsim.zernike.Zernike, optional
            Additional focal plane offsets (in meters) represented as Zernike
            series.
        thx, thy : float
            Field angles in radians.
        nrad : int
            Number of pupil radii to use between R_inner and R_outer.
        naz : int, optional
            Approximate number of azimuthal angles to use along the outer most radius.
            See hexapolar for details.
        npix : int
            Number of pixels along image edge.  Must be odd.
        gq_kwargs : dict, optional
            Additional keyword arguments to pass to gq_points.

        Returns
        -------
        img : array of float
            Spot diagram image.
        x, y : array of float
            Focal plane coordinates of convolved spots.
        w : array of float
            Weights of convolved spots.
        """
        sx, sy, sw = self.spots(
            Z=Z, aberrations=aberrations,
            x_offset=x_offset, y_offset=y_offset,
            thx=thx, thy=thy,
            nrad=nrad, naz=naz
        )
        gx, gy, gw = gq_points(
            **(gq_kwargs or {})
        )
        x = np.add.outer(sx, gx)
        y = np.add.outer(sy, gy)
        w = np.multiply.outer(sw, gw)

        img = np.zeros((npix, npix))
        no2 = (npix-1)//2
        bds = np.linspace(-no2-0.5, no2+0.5, npix+1)*self.pixel_scale
        H, *_ = np.histogram2d(y.ravel(), x.ravel(), bins=[bds, bds], weights=w.ravel(), density=False)
        return H, x, y, w
