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

from pathlib import Path
import yaml

import numpy as np
from numpy.typing import NDArray


def hexapolar(outer=1.0, inner=0.0, nrad=5, naz=None, kfold=6, rth=False):
    """Generate hexapolar grid of points.

    Parameters
    ----------
    outer : float, optional
        Outer radius of hexapolar grid.
    inner : float, optional
        Inner radius of hexapolar grid.
    nrad : int, optional
        Number of radii on which create points.
    naz : int, optional
        Approximate number of azimuthal angles uniformly spaced along the
        outermost ring.  Each ring is constrained to have a multiple of kfold
        azimuths, so the realized value may be slightly different than the
        input value here.  Inner rings will have fewer azimuths in proportion
        to their radius, but will still be constrained to a multiple of kfold.
        (If the innermost ring has radius 0, then exactly 1 point, with azimuth
        undefined, will be used on that "ring".)  Default: None, which means to
        scale the number of azimuths to the number of radii such that radii and
        azimuths are approximately equally spaced.
    kfold : int, optional
        Each ring will have a multiple of this many azimuths.  Default: 6.
    rth : bool, optional
        If True, return r, theta instead of x, y.

    Returns
    -------
    x, y : ndarray
        Hexapolar grid.
    """
    nphis = []
    rhos = np.linspace(outer, inner, nrad)
    if naz is None:
        naz = int(2*np.pi*nrad*(1-inner/outer))
    for rho in rhos:
        nphi = int((naz*rho/outer)//kfold)*kfold
        if nphi == 0:
            nphi = kfold
        nphis.append(nphi)
    if inner == 0.0:
        nphis[-1] = 1
    n = np.sum(nphis)
    th = np.empty(n)
    rr = np.empty(n)
    idx = 0
    for rho, nphi in zip(rhos, nphis):
        rr[idx:idx+nphi] = rho
        th[idx:idx+nphi] = np.linspace(0, 2*np.pi, nphi, endpoint=False)
        idx += nphi
    if inner == 0.0:
        rr[-1] = 0.0
        th[-1] = 0.0
    if rth:
        return rr, th
    else:
        return rr*np.cos(th), rr*np.sin(th)


def gq_points(
    nrad: int = 2,
    naz: int | None = None,
    kfold: int = 6,
    cov: NDArray | None = None,
    center: bool = False,
    rmax: float | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Deterministic weighted point set whose moments match a 2D Gaussian.

    Builds concentric rings of equally-spaced points whose radii and
    per-ring weights are set by quadrature on the radial variable.

    When *rmax* is ``None`` (the default), Gauss-Laguerre quadrature is
    used on `s = r²/2`, placing nodes over the full ``[0, ∞)`` range.
    All moments ``E[x^a y^b]`` of the target Gaussian are reproduced
    exactly through total degree

        D = min(naz - 1, 4 * nrad - 2)

    For the default ``nrad=2, naz=6`` (12 points) this gives D = 5,
    i.e. all moments through 5th order are exact.

    When *rmax* is given, Gauss-Legendre quadrature is used on the
    radial CDF interval ``[0, u_max]`` where ``u_max = 1 - exp(-rmax²/2)``,
    so that all rings lie within *rmax* sigma.  The small amount of
    probability mass beyond *rmax* is discarded and the weights are
    renormalized to sum to 1.  This places nodes more densely in the
    interior and is better suited for UKF-style applications where
    many rings are desired without reaching far into the tails.

    Parameters
    ----------
    nrad : int
        Number of concentric rings (default 2).
    naz : int
        Approximate number of azimuthal angles uniformly spaced along the
        outermost ring.  Each ring is constrained to have a multiple of kfold
        azimuths, so the realized value may be slightly different than the
        input value here.  Inner rings will have fewer azimuths in proportion
        to their radius, but will still be constrained to a multiple of kfold.
    kfold : int, optional
        Each ring will have a multiple of this many azimuths.  Default: 6.
    cov : (2, 2) array_like or None
        Target covariance matrix.  Defaults to the 2x2 identity
        (standard normal).  For a general covariance the standard-normal
        points are linearly transformed via the Cholesky factor.
    center : bool
        If *True*, prepend a point at the origin with weight zero.
        (The Gauss-Laguerre rule always assigns zero weight to r = 0
        because the radial density r·exp(-r²/2) vanishes there, but a
        centre point can still be useful as a structural reference.)
    rmax : float or None
        Maximum radius (in units of sigma) for the outermost ring.  When
        set, Gauss-Legendre quadrature on the radial CDF is used instead
        of Gauss-Laguerre, keeping all points within *rmax*.

    Returns
    -------
    x, y : ndarray, shape ``(N,)``
        Sample positions, where ``N = nrad * naz (+ 1 if center)``.
    w : ndarray, shape ``(N,)``
        Non-negative weights summing to 1.
    """
    if rmax is not None:
        # Gauss-Legendre on the radial CDF interval [0, u_max].
        # CDF substitution: u = 1 - exp(-r²/2), r = √(-2 ln(1 - u))
        u_max = 1.0 - np.exp(-0.5 * rmax**2)
        xi, omega_leg = np.polynomial.legendre.leggauss(nrad)
        # Map Legendre nodes from [-1, 1] to [0, u_max]
        u = u_max * 0.5 * (xi + 1.0)
        # Radial weights, renormalized so they sum to 1
        omega = 0.5 * omega_leg * u_max / u_max  # = omega_leg / 2
        # (The u_max factors from the change-of-variable and the
        #  renormalization cancel, leaving omega_leg / 2.)
        omega = 0.5 * omega_leg
        # Invert CDF to get radii
        rhos = np.sqrt(-2.0 * np.log(1.0 - u))
    else:
        # Gauss–Laguerre nodes t_j and weights omega_j for
        # ∫ f(t) exp(−t) dt on [0, ∞).
        t, omega = np.polynomial.laguerre.laggauss(nrad)
        # Map to radii: s = r²/2 = t  ⟹  r = √(2t)
        rhos = np.sqrt(2.0 * t)

    if naz is None:
        naz = int(2*np.pi*nrad)

    nphis = []
    outer = np.max(rhos)
    for rho in rhos:
        nphi = int((naz*rho/outer)//kfold)*kfold
        if nphi == 0:
            nphi = kfold
        nphis.append(nphi)
    n = np.sum(nphis)
    r = np.empty(n)
    th = np.empty(n)
    w = np.empty(n)
    idx = 0
    for rho, nphi, om in zip(rhos, nphis, omega):
        r[idx:idx+nphi] = rho
        th[idx:idx+nphi] = np.linspace(0, 2*np.pi, nphi, endpoint=False)
        w[idx:idx+nphi] = om / nphi
        idx += nphi

    x = r * np.cos(th)
    y = r * np.sin(th)
    if center:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[0.0], y])
        w = np.concatenate([[0.0], w])

    # Apply covariance transformation  x' = L @ x  where Σ = L Lᵀ
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        L = np.linalg.cholesky(cov)
        xy = L @ np.stack([x, y])
        x, y = xy[0], xy[1]

    return x, y, w


def load_mask_params(filename) -> dict:
    """ Load mask parameters from a YAML file in the datadir.

    Parameters
    ----------
    filename : str
        Name of the YAML file containing the mask parameters.

    Returns
    -------
    dict
        Dictionary containing the mask parameters.
    """
    from . import datadir
    path = Path(datadir) / filename
    with open(path, "r") as f:
        mask = yaml.safe_load(f)
    return mask
