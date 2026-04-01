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
    nrings: int = 2,
    nphi: int = 6,
    cov: NDArray | None = None,
    center: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """Deterministic weighted point set whose moments match a 2D Gaussian.

    Builds concentric rings of equally-spaced points whose radii and
    per-ring weights are set by Gauss-Laguerre quadrature on the radial
    variable `s = r²/2`.  All moments ``E[x^a y^b]`` of the target
    Gaussian are reproduced exactly through total degree

        D = min(nphi - 1, 4 * nrings - 2)

    For the default ``nrings=2, nphi=6`` (12 points) this gives D = 5,
    i.e. all moments through 5th order are exact.

    Parameters
    ----------
    nrings : int
        Number of concentric rings (default 2).
    nphi : int
        Points per ring (default 6).  Even values give point symmetry
        ``(x, y) ↔ (-x, -y)``.
    cov : (2, 2) array_like or None
        Target covariance matrix.  Defaults to the 2x2 identity
        (standard normal).  For a general covariance the standard-normal
        points are linearly transformed via the Cholesky factor.
    center : bool
        If *True*, prepend a point at the origin with weight zero.
        (The Gauss-Laguerre rule always assigns zero weight to r = 0
        because the radial density r·exp(-r²/2) vanishes there, but a
        centre point can still be useful as a structural reference.)

    Returns
    -------
    x, y : ndarray, shape ``(N,)``
        Sample positions, where ``N = nrings * nphi (+ 1 if center)``.
    w : ndarray, shape ``(N,)``
        Non-negative weights summing to 1.
    """
    # Gauss–Laguerre nodes t_j and weights omega_j for  ∫ f(t) exp(−t) dt
    # on [0, ∞).  An n-point rule integrates polynomials of degree ≤ 2n−1.
    t, omega = np.polynomial.laguerre.laggauss(nrings)

    # Map to radii: s = r²/2 = t  ⟹  r = √(2t)
    r = np.sqrt(2.0 * t)

    # Split each ring weight equally among its azimuthal points.
    w_per_point = omega / nphi

    # Uniform azimuthal grid on each ring
    theta = np.linspace(0, 2 * np.pi, nphi, endpoint=False)

    # (nrings, nphi) outer product → flattened
    x = (r[:, None] * np.cos(theta[None, :])).ravel()
    y = (r[:, None] * np.sin(theta[None, :])).ravel()
    w = np.repeat(w_per_point, nphi)

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
