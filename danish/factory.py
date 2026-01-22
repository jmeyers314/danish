# Copyright (c) 2021-2026, Lawrence Livermore National Laboratory, LLC.
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

import numpy as np
import galsim
from functools import lru_cache
from ._danish import poly_grid_contains, pixel_frac, enclosed_circle, enclosed_strut


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
    prefit_order=2, maxiter=20, tol=1e-5, strict=False
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


def _focal_to_pupil(
    x, y, Z, *,
    focal_length=None,
    x_offset=None, y_offset=None,
    prefit_order=2, maxiter=20, tol=1e-5,
    strict=False
):
    Z1 = Z * focal_length if focal_length else Z
    utest = np.linspace(-Z1.R_outer, Z1.R_outer, 10)
    utest, vtest = np.meshgrid(utest, utest)
    r2test = utest**2 + vtest**2
    w = r2test >= Z1.R_inner**2
    w &= r2test <= Z1.R_outer**2
    utest = utest[w]
    vtest = vtest[w]
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
    for i in range(maxiter):
        if i >= 1:
            if np.max(np.abs(dx)) < tol and np.max(np.abs(dy)) < tol:
                break
        dW2du2, dW2dudv, dW2dvdu, dW2dv2 = _pupil_focal_jacobian(
            u, v, Z1, x_offset=x_offset, y_offset=y_offset
        )
        det = (dW2du2*dW2dv2 - dW2dudv*dW2dvdu)
        # du = -(dW2dv2*dx - dW2dudv*dy)/det
        # dv = -(-dW2dvdu*dx + dW2du2*dy)/det
        du = -(dW2dv2*dx - dW2dvdu*dy)/det
        dv = -(-dW2dudv*dx + dW2du2*dy)/det
        # If xy miss distance increased, then decrease duv by
        # sqrt(distance ratio)
        uc = u + du
        vc = v + dv
        xc, yc = _pupil_to_focal(
            uc, vc, Z1, x_offset=x_offset, y_offset=y_offset
        )
        dxc = xc - x
        dyc = yc - y
        drc2 = dxc**2 + dyc**2
        w = drc2 > dr2  # places where we're worse
        if np.any(w):
            alpha = np.maximum(0.001, (dr2[w]/drc2[w])**0.25)
            uc[w] = u[w] + alpha*du[w]
            vc[w] = v[w] + alpha*dv[w]
            xc[w], yc[w] = _pupil_to_focal(
                uc[w], vc[w], Z1,
                x_offset=x_offset, y_offset=y_offset
            )
            dxc[w] = xc[w] - x[w]
            dyc[w] = yc[w] - y[w]
            drc2[w] = dxc[w]**2 + dyc[w]**2
        u, v, dr2 = uc, vc, drc2
        x_current, y_current = xc, yc
        dx, dy = dxc, dyc
    else:
        # If we failed to reach the desired tolerance, mark coordinate with a
        # NaN or if `strict`, raise a RuntimeError.
        # Diagnostic information
        intolerable = (np.abs(dx) > tol) | (np.abs(dy) > tol)
        wfail = np.nonzero(intolerable)
        if strict:
            print(Z1)
            for idx in wfail:
                print(x[idx], y[idx])
            raise RuntimeError("Cannot invert")
        u[wfail] = np.nan
        v[wfail] = np.nan
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
        return 0.0  # Outside the strut

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


class DonutFactory:
    """
    Parameters
    ----------
    R_outer : float
        Entrance pupil radius in meters.
        Also assumed to be Zernike normalization radius.
    R_inner : float
        Entrance pupil inner radius.  Used for defining annular Zernikes.
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
        mask_params=None,
        spider_angle=None,
        focal_length=10.31, pixel_scale=10e-6
    ):
        self.R_outer = R_outer
        self.R_inner = R_inner
        self.mask_params = mask_params
        self.spider_angle = spider_angle
        self.focal_length = focal_length
        self.pixel_scale = pixel_scale

    def image(
        self, *,
        Z=None, aberrations=None,
        x_offset=None, y_offset=None,
        thx=0, thy=0,
        npix=181,
        prefit_order=2, maxiter=20, tol=1e-5, strict=False
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
        u, v = self.R_outer*np.cos(ph), self.R_outer*np.sin(ph)
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
        ypix, xpix = np.nonzero(contained)
        x = (xpix.astype(float) - no2)*self.pixel_scale # meters
        y = (ypix.astype(float) - no2)*self.pixel_scale

        # Now invert to get pixel centers projected on pupil
        u, v = _focal_to_pupil(
            x, y, Z1,
            x_offset=x_offset, y_offset=y_offset,
            prefit_order=prefit_order, maxiter=maxiter, tol=tol, strict=strict
        )

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
            0.0, 0.0, self.R_outer,
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

        f[w] /= np.max(f[w])
        img[ypix, xpix] = f
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
