# Copyright (c) 2021, Lawrence Livermore National Laboratory, LLC.
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
from ._danish import poly_grid_contains


def pupil_to_focal(
    u, v, *,
    Z=None, aberrations=None, R_outer=1.0, R_inner=0.0,
    focal_length=None
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
    """
    if Z is None:
        Z = galsim.zernike.Zernike(
            aberrations, R_outer=R_outer, R_inner=R_inner
        )
    if focal_length is None:
        raise ValueError("Missing focal length")

    return _pupil_to_focal(u, v, Z, focal_length=focal_length)


def _pupil_to_focal(u, v, Z, *, focal_length=None):
    Z1 = Z * focal_length if focal_length else Z
    return (
        Z1.gradX(u, v),
        Z1.gradY(u, v)
    )


def pupil_focal_jacobian(
    u, v, *,
    Z=None, aberrations=None, R_outer=1.0, R_inner=0.0,
    focal_length=None
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
    """
    if Z is None:
        Z = galsim.zernike.Zernike(
            aberrations, R_outer=R_outer, R_inner=R_inner
        )
    if focal_length is None:
        raise ValueError("Missing focal length")

    return _pupil_focal_jacobian(u, v, Z, focal_length)


def _pupil_focal_jacobian(u, v, Z, *, focal_length=None):
    Z1 = Z * focal_length if focal_length else Z
    dxdu = Z1.gradX.gradX(u, v)
    dxdv = Z1.gradX.gradY(u, v)
    dydu = dxdv
    # dydu = Z1.gradY.gradX(u, v)
    dydv = Z1.gradY.gradY(u, v)
    return dxdu, dxdv, dydu, dydv


def focal_to_pupil(
    x, y, *,
    Z=None, aberrations=None, R_outer=1.0, R_inner=0.0,
    focal_length=None,
    prefit_order=2, maxiter=20, tol=1e-5
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
        prefit_order=prefit_order,
        focal_length=focal_length,
        maxiter=maxiter,
        tol=tol
    )


def _focal_to_pupil(x, y, Z, *, focal_length=None, prefit_order=2, maxiter=20, tol=1e-5):
    Z1 = Z * focal_length if focal_length else Z
    utest = np.linspace(-Z1.R_outer, Z1.R_outer, 10)
    utest, vtest = np.meshgrid(utest, utest)
    r2test = utest**2 + vtest**2
    w = r2test >= Z1.R_inner**2
    w &= r2test <= Z1.R_outer**2
    utest = utest[w]
    vtest = vtest[w]
    xtest, ytest = _pupil_to_focal(utest, vtest, Z1)

    # Prefit
    jmax = (prefit_order+1)*(prefit_order+2)//2
    R_outer = np.max(np.hypot(xtest, ytest))
    a = galsim.zernike.zernikeBasis(jmax, xtest, ytest, R_outer=R_outer).T
    b = np.array([utest, vtest]).T
    r, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

    u = galsim.zernike.Zernike(r[:,0], R_outer=R_outer)(x, y)
    v = galsim.zernike.Zernike(r[:,1], R_outer=R_outer)(x, y)

    # Newton-Raphson iterations to invert pupil_to_focal
    x_current, y_current = _pupil_to_focal(u, v, Z1)
    dx = x_current - x
    dy = y_current - y
    dr2 = dx**2 + dy**2
    for i in range(maxiter):
        if i >= 1:
            if np.max(np.abs(dx)) < tol and np.max(np.abs(dy)) < tol:
                break
        dW2du2 = Z1.gradX.gradX(u, v)
        dW2dudv = Z1.gradX.gradY(u, v)
        dW2dv2 = Z1.gradY.gradY(u, v)
        det = (dW2du2*dW2dv2 - dW2dudv**2)
        du = -(dW2dv2*dx - dW2dudv*dy)/det
        dv = -(-dW2dudv*dx + dW2du2*dy)/det
        # If xy miss distance increased, then decrease duv by sqrt(distance ratio)
        uc = u + du
        vc = v + dv
        xc, yc = _pupil_to_focal(uc, vc, Z1)
        dxc = xc - x
        dyc = yc - y
        drc2 = dxc**2 + dyc**2
        w = drc2 > dr2  # places where we're worse
        if np.any(w):
            alpha = np.maximum(0.001, (dr2[w]/drc2[w])**0.25)
            uc[w] = u[w] + alpha*du[w]
            vc[w] = v[w] + alpha*dv[w]
            xc[w], yc[w] = _pupil_to_focal(uc[w], vc[w], Z1)
            dxc[w] = xc[w] - x[w]
            dyc[w] = yc[w] - y[w]
            drc2[w] = dxc[w]**2 + dyc[w]**2
        u, v, dr2 = uc, vc, drc2
        x_current, y_current = xc, yc
        dx, dy = dxc, dyc
    else:
        # Diagnostic information
        wfail = np.nonzero((np.abs(dx) > tol) | (np.abs(dy) > tol))[0]
        print(Z1)
        for idx in wfail:
            print(x[idx], y[idx])
        raise RuntimeError("Cannot invert")
    return u, v


def enclosed_fraction(
    x, y,
    u, v,
    u0, v0, radius, *,
    Z=None, aberrations=None, R_outer=1.0, R_inner=0.0,
    focal_length=None, pixel_scale=None,
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

    return _enclosed_fraction(
        x, y, u, v, u0, v0, radius, Z, focal_length, pixel_scale,
    )


# @profile
def _enclosed_fraction(
    x, y,
    u, v,
    u0, v0, radius,
    Z, *,
    focal_length=None,
    pixel_scale=1.0,
    _jac=None,
):
    Z1 = Z * focal_length if focal_length else Z
    out = np.zeros_like(x)  # the enclosed fraction
    du = u - u0  # pupil displacement from circle center
    dv = v - v0

    # First determine "obvious" points either far inside or far outside circle
    # of interest.
    if _jac is None:
        dxdu, dxdv, dydu, dydv = _pupil_focal_jacobian(u, v, Z1)
        det = dxdu*dydv - dxdv*dydu
        dudx = dydv/det
        dudy = -dxdv/det
        dvdx = -dydu/det
        dvdy = dxdu/det
    else:
        dxdu, dxdv, dydu, dydv, dudx, dudy, dvdx, dvdy = _jac

    drho = np.hypot(du, dv)
    h1 = np.sqrt((dudx + dvdy)**2 + (dudy - dvdx)**2)
    h2 = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2)
    maxLinearScale = 0.5 * (h1 + h2) * pixel_scale
    winside = drho < radius - maxLinearScale
    woutside = drho > radius + maxLinearScale
    wunknown = ~winside & ~woutside
    wx = np.nonzero(wunknown)[0]

    out[winside] = 1.0
    out[woutside] = 0.0

    if not np.any(wunknown):
        return out

    # restrict to unknown points
    u = u[wunknown]
    v = v[wunknown]
    du = du[wunknown]
    dv = dv[wunknown]
    x = x[wunknown]
    y = y[wunknown]
    dxdu = dxdu[wunknown]
    dxdv = dxdv[wunknown]
    dydu = dydu[wunknown]
    dydv = dydv[wunknown]
    dudx = dudx[wunknown]
    dudy = dudy[wunknown]
    dvdx = dvdx[wunknown]
    dvdy = dvdy[wunknown]

    # Calculate nearby slope/intercept of circle in pupil coords
    # See Janish (A.2)
    mp = -du/dv
    bp = np.hypot(du, dv)/dv*radius
    # Adjust for circle center
    bp += v0 - mp*u0

    # Transform slope/intercept to focal coords
    alpha = dvdy - mp*dudy
    beta = mp*dudx - dvdx
    gamma = mp*u + bp - v
    m = beta/alpha
    b = (-beta*x + gamma)/alpha + y

    # Use local linear approx to transform u0, v0 -> x0, y0
    x0 = x + (u0-u)*dxdu + (v0-v)*dxdv
    y0 = y + (u0-u)*dydu + (v0-v)*dydv

    # Center coords around x0, y0
    x -= x0
    y -= y0
    b += m*x0-y0

    # Normalize to m < 0
    w = (m > 0)
    x[w] = -x[w]
    m[w] = -m[w]

    # Normalize to -1 < m
    w = (m < -1)
    x[w], y[w] = y[w], x[w]
    m[w], b[w] = 1/m[w], -b[w]/m[w]

    # Convert meters -> pixels
    x /= pixel_scale
    y /= pixel_scale
    b /= pixel_scale

    # Distance b/n top of pixel and circle intersection point.
    # Janish (A.3) and (A.4)
    gamma = y + 0.5 - (m*(x + 0.5) + b)

    # pixel is fully inside circle
    w = gamma < 0
    out[wx[w]] = 1.0

    # pixel is fully outside circle
    w = gamma > (1-m)
    out[wx[w]] = 0.0

    # line crosses left and bottom
    w = (1 < gamma) & (gamma < (1-m))
    mw = m[w]
    out[wx[w]] = -0.5/mw*(1 - (gamma[w]+mw))**2

    # crosses left and right
    w = (-m < gamma) & (gamma < 1)
    out[wx[w]] = 1 - gamma[w] - m[w]/2

    # crosses top and right
    w = (0 < gamma) & (gamma < -m)
    out[wx[w]] = 1 + 0.5*gamma[w]**2/m[w]

    w = y<0
    out[wx[w]] = 1 - out[wx[w]]

    return out


def _enclosed_fraction_debug(
    x, y,
    u, v,
    u0, v0, radius,
    Z, *,
    focal_length=None,
    pixel_scale=1.0,
    axes=None
):  # pragma: no cover
    if axes:
        ax0, ax1 = axes

    print(f"(x, y) = ({x:.6f}, {y:.6f})")
    print(f"(xp, yp) = ({x/pixel_scale:.1f}, {y/pixel_scale:.1f})")
    print(f"(u, v) = ({u:.4f}, {v:.4f})")
    print(f"(u0, v0) = ({u0:.4f}, {v0:.4f})")
    Z1 = Z * focal_length if focal_length else Z
    du = u - u0  # pupil displacement from circle center
    dv = v - v0

    # Transform slope/intercept into focal coords
    dxdu, dxdv, dydu, dydv = _pupil_focal_jacobian(u, v, Z1)
    det = dxdu*dydv - dxdv*dydu
    dudx = dydv/det
    dudy = -dxdv/det
    dvdx = -dydu/det
    dvdy = dxdu/det

    drho = np.hypot(du, dv)
    h1 = np.sqrt((dudx + dvdy)**2 + (dudy - dvdx)**2)
    h2 = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2)
    maxLinearScale = 0.5 * (h1 + h2) * pixel_scale
    print(f"maxLinearScale = {maxLinearScale:.4f}")

    if drho < radius - maxLinearScale:
        print("quick inside")
        return 1.0
    elif drho > radius + maxLinearScale:
        print("quick outside")
        return 0.0
    else:
        print("quick unknown")

    # Calculate nearby slope/intercept of circle in pupil coords
    mp = -du/dv
    bp = np.hypot(du, dv)/dv*radius
    # Adjust for circle center
    bp += v0 - mp*u0

    if ax1:
        xs = np.linspace(-4.18, 4.18)
        ys = xs*mp + bp
        ax1.plot(xs, ys, c='r')

    # alpha = dvdy - mp*dudy
    # beta = mp*dudx - dvdx
    # gamma = mp*du + bp - dv
    # m = beta/alpha
    # b = (-beta*x + gamma)/alpha + y
    alpha = dvdy - mp*dudy
    beta = mp*dudx - dvdx
    gamma = mp*u + bp - v
    m = beta/alpha
    b = (-beta*x + gamma)/alpha + y

    if ax0:
        xs = np.linspace(-90, 90)
        ys = xs*m + b/pixel_scale
        ax0.plot(xs, ys, c='m')

    # Use local linear approx to transform u0, v0 -> x0, y0
    x0 = x + (u0-u)*dxdu + (v0-v)*dxdv
    y0 = y + (u0-u)*dydu + (v0-v)*dydv

    # Center coords around x0, y0
    x -= x0
    y -= y0
    b += m*x0-y0

    print(f"initial m = {m:.4f}")
    print(f"initial b = {b:.4f}")

    # Normalize to m < 0
    if m > 0:
        print("normalizing to m < 0")
        x, m = -x, -m

    # Normalize to -1 < m
    if m < -1:
        print("normalizing to -1 < m")
        x, y = y, x
        m, b = 1/m, -b/m

    # Convert meters -> pixels
    x /= pixel_scale
    y /= pixel_scale
    b /= pixel_scale

    gamma = y + 0.5 - (m*(x + 0.5) + b)
    print(f"gamma = {gamma:.4f}")
    print(f"m = {m:.4f}")

    # pixel is fully inside circle
    if gamma < 0:
        print("slow inside")
        print(f"  gamma < 0  :  {gamma:.4f} < 0")
        out = 1.0

    # pixel is fully outside circle
    if gamma > (1-m):
        print("slow outside")
        print(f"  gamma > (1-m)  :  {gamma:.4f} > {1-m:.4f}")
        out = 0.0

    # line crosses left and bottom
    if (1 < gamma) & (gamma < (1-m)):
        print("slow LB")
        print(f"  1 < gamma  :  1 < {gamma:.4f}")
        print(f"  gamma < (1-m)  :  {gamma:.4f} < {1-m:.4f}")
        out = -0.5/m*(1 - (gamma+m))**2

    # crosses left and right
    if (-m < gamma) & (gamma < 1):
        print("slow LR")
        print(f"  -m < gamma  :  {-m:.4f} < {gamma:.4f}")
        print(f"  gamma < 1  :  {gamma:.4f} < 1")
        out = 1 - gamma - m/2

    # crosses top and right
    if (0 < gamma) & (gamma < -m):
        print("slow TR")
        print(f"  0 < gamma  :  0 < {gamma:.4f}")
        print(f"  gamma < -m  :  {gamma:.4f} < {-m:.4f}")
        out = 1 + 0.5*gamma**2/m

    if y<0:
        print("flip")
        print(f"  y < 0  :  {y:.4f} < 0")
        out = 1 - out

    print(out)
    return out


class DonutFactory:
    """
    Parameters
    ----------
    R_outer : float
        Entrance pupil radius in meters.
        Also assumed to be Zernike normalization radius.
    R_inner : float
        Entrance pupil inner radius.  Used for defining annular Zernikes.
    obsc_radii : array of float
        Obscuration radii projected onto pupil in meters.
    obsc_motion : array of float
        Obscuration motion with field angle in meters/radian.
    focal_length : float
        Focal length in meters.
    pixel_scale : float
        Pixel scale in meters.
    """
    def __init__(
        self, *,
        R_outer=4.18, R_inner=2.5498,
        obsc_radii=None, obsc_motion=None,
        focal_length=10.31, pixel_scale=10e-6
    ):
        self.R_outer = R_outer
        self.R_inner = R_inner

        # TODO: Allow focal_length, obsc_radii, obsc_motion to be functions
        #       of field angle.
        self.obsc_radii = obsc_radii
        self.obsc_motion = obsc_motion
        self.focal_length = focal_length
        self.pixel_scale = pixel_scale

    # @profile
    def image(
        self, *,
        Z=None, aberrations=None,
        thx=0, thy=0, npix=181,
    ):
        """Compute aberrated donut image.

        Parameters
        ----------
        Z : galsim.zernike.Zernike, optional
            Aberrations in meters.
        aberrations : array of float, optional
            Aberrations in meters.
        thx, thy : float
            Field angles in radians.
        npix : int
            Number of pixels along image edge.  Must be odd.

        Returns
        -------
        img : array of float
            Donut image.
        """
        if npix%2 == 0:
            raise ValueError(f"Argument {npix=} must be odd.")
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
            u, v, Z1
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

        contained = corners[1:,1:]
        contained |= corners[:-1,1:]
        contained |= corners[1:,:-1]
        contained |= corners[:-1,:-1]
        ypix, xpix = np.nonzero(contained)
        x = (xpix.astype(float) - no2)*self.pixel_scale # meters
        y = (ypix.astype(float) - no2)*self.pixel_scale

        # Now invert to get pixel centers projected on pupil
        u, v = _focal_to_pupil(x, y, Z1)

        img = np.zeros((npix, npix))

        # Compute jacobian just once
        dxdu, dxdv, dydu, dydv = _pupil_focal_jacobian(u, v, Z1)
        det = dxdu*dydv - dxdv*dydu
        dudx = dydv/det
        dudy = -dxdv/det
        dvdx = -dydu/det
        dvdy = dxdu/det
        jac = np.array([dxdu, dxdv, dydu, dydv, dudx, dudy, dvdx, dvdy])

        f = _enclosed_fraction(
            x, y, u, v,
            0.0, 0.0, self.R_outer,
            Z=Z1,
            pixel_scale=self.pixel_scale,
            _jac=jac,
        )

        w = np.nonzero(f)[0]
        if self.obsc_radii is not None:
            for k in self.obsc_radii:
                if not np.any(w):
                    break
                enc = _enclosed_fraction(
                    x[w], y[w], u[w], v[w],
                    -self.obsc_motion[k]*thx,
                    -self.obsc_motion[k]*thy,
                    self.obsc_radii[k],
                    Z=Z1,
                    pixel_scale=self.pixel_scale,
                    _jac=jac[:, w],
                )
                if '_inner' in k:
                    f[w] = np.minimum(f[w], 1-enc)
                else:
                    f[w] = np.minimum(f[w], enc)
                w = np.nonzero(f)[0]

        # pixel pupil-to-focal area ratio
        f[w] /= Z1.hessian(u[w], v[w])
        f[w] /= np.max(f[w])

        img[ypix, xpix] = f
        return img
