"""test_cpp_sync.py

Verify that the scalar Python reference implementations in factory.py stay
in sync with their C++ counterparts in _danish.so.

Each test generates a batch of inputs, evaluates the Python _1 scalar
function for each point, then evaluates the C++ vectorised wrapper on the
whole batch and compares.  test_enclosed_strut_wide additionally checks the
wide-strut code path (pixels far from both edges) that was previously buggy.
"""

import numpy as np

from danish.factory import (
    _pixel_frac_1,
    _enclosed_circle_1,
    _enclosed_strut_1,
    _pixel_frac,
    _enclosed_fraction,
    _strut_masked_fraction,
)
from danish_test_helpers import timer, runtests

RNG = np.random.default_rng(57721)


def _jacobian(n, step=0.05):
    """Simple axis-aligned Jacobian arrays (no shear)."""
    return (
        np.full(n, step),  # dudx
        np.zeros(n),       # dudy
        np.zeros(n),       # dvdx
        np.full(n, step),  # dvdy
    )


@timer
def test_pixel_frac_sync():
    """_pixel_frac_1 (Python scalar) must match _pixel_frac (C++) for random inputs."""
    n = 400
    u0, v0 = 1.2, -0.8
    angle = 0.7
    sth0, cth0 = float(np.sin(angle)), float(np.cos(angle))

    u1 = RNG.uniform(-4, 4, n).astype(np.float64)
    v1 = RNG.uniform(-4, 4, n).astype(np.float64)
    x1 = RNG.uniform(-0.05, 0.05, n).astype(np.float64)
    y1 = RNG.uniform(-0.05, 0.05, n).astype(np.float64)
    dudx, dudy, dvdx, dvdy = _jacobian(n)

    py = np.array([
        _pixel_frac_1(u0, v0, sth0, cth0,
                      u1[i], v1[i], x1[i], y1[i],
                      dudx[i], dudy[i], dvdx[i], dvdy[i])
        for i in range(n)
    ])
    cpp = _pixel_frac(u0, v0, sth0, cth0,
                      u1, v1, x1, y1,
                      dudx, dudy, dvdx, dvdy)
    np.testing.assert_allclose(py, cpp, rtol=1e-12, atol=1e-14)


@timer
def test_enclosed_circle_sync():
    """_enclosed_circle_1 (Python scalar) must match _enclosed_fraction (C++) for random inputs."""
    n = 400
    u0, v0, radius = 0.5, -0.3, 2.0

    u = RNG.uniform(-4, 4, n).astype(np.float64)
    v = RNG.uniform(-4, 4, n).astype(np.float64)
    x = RNG.uniform(-0.05, 0.05, n).astype(np.float64)
    y = RNG.uniform(-0.05, 0.05, n).astype(np.float64)
    dudx, dudy, dvdx, dvdy = _jacobian(n)

    py = np.array([
        _enclosed_circle_1(x[i], y[i], u[i], v[i],
                           u0, v0, radius,
                           dudx[i], dudy[i], dvdx[i], dvdy[i])
        for i in range(n)
    ])
    cpp = _enclosed_fraction(x, y, u, v, u0, v0, radius,
                             dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy)
    np.testing.assert_allclose(py, cpp, rtol=1e-12, atol=1e-14)


@timer
def test_enclosed_strut_sync():
    """_enclosed_strut_1 (Python scalar) must match _strut_masked_fraction (C++) for random inputs."""
    n = 400
    # Horizontal strut centred at origin, edges at v = ±0.3
    # Direction along u: sth=0, cth=1
    u1, v1, sth1, cth1 =  0.0,  0.3, 0.0, 1.0
    u2, v2, sth2, cth2 =  0.0, -0.3, 0.0, 1.0
    length = 4.0

    u = RNG.uniform(-4, 4, n).astype(np.float64)
    v = RNG.uniform(-4, 4, n).astype(np.float64)
    x = RNG.uniform(-0.05, 0.05, n).astype(np.float64)
    y = RNG.uniform(-0.05, 0.05, n).astype(np.float64)
    dudx, dudy, dvdx, dvdy = _jacobian(n)

    py = np.array([
        _enclosed_strut_1(x[i], y[i], u[i], v[i], length,
                          u1, v1, sth1, cth1,
                          u2, v2, sth2, cth2,
                          dudx[i], dudy[i], dvdx[i], dvdy[i])
        for i in range(n)
    ])
    cpp = _strut_masked_fraction(x, y, u, v, length,
                                 u1, v1, sth1, cth1,
                                 u2, v2, sth2, cth2,
                                 dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy)
    np.testing.assert_allclose(py, cpp, rtol=1e-12, atol=1e-14)


@timer
def test_enclosed_strut_wide():
    """Wide-strut code path: pixels far from both edges must return 1.0 (inside)
    or 0.0 (outside).

    Uses a tiny pixel scale so maxLinearScale << half_width, forcing the
    wide-strut branch for all interior and exterior test pixels.

    Geometry: two horizontal edges at v = ±half_w running along u.
      sth = 0, cth = 1 → perpendicular distance = |v - v_edge|
      signed distance s = (v - v_edge)
      Interior pixel (v = 0): s1 = -half_w, s2 = +half_w  → s1*s2 < 0 → 1.0
      Exterior pixel (v = 2*half_w): s1 = s2 > 0           → s1*s2 > 0 → 0.0
    """
    step = 1e-4    # tiny → maxLinearScale ≈ step, 2*maxLinearScale ≈ 2e-4
    half_w = 0.5   # >> 2*maxLinearScale: all test pixels take the wide-strut branch
    length = 10.0

    u1, v1, sth1, cth1 = 0.0,  half_w, 0.0, 1.0
    u2, v2, sth2, cth2 = 0.0, -half_w, 0.0, 1.0

    n = 20
    u_pts = np.linspace(-0.4 * length / 2, 0.4 * length / 2, n)
    x = np.zeros(n)
    y = np.zeros(n)
    dudx, dudy, dvdx, dvdy = _jacobian(n, step=step)

    def _py(v_pts):
        return np.array([
            _enclosed_strut_1(x[i], y[i], u_pts[i], v_pts[i], length,
                              u1, v1, sth1, cth1,
                              u2, v2, sth2, cth2,
                              dudx[i], dudy[i], dvdx[i], dvdy[i])
            for i in range(n)
        ])

    def _cpp(v_pts):
        return _strut_masked_fraction(x, y, u_pts, v_pts, length,
                                      u1, v1, sth1, cth1,
                                      u2, v2, sth2, cth2,
                                      dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy)

    # Interior pixels (v = 0, between the two edges)
    v_in = np.zeros(n)
    py_in  = _py(v_in)
    cpp_in = _cpp(v_in)
    np.testing.assert_array_equal(py_in,  1.0)
    np.testing.assert_array_equal(cpp_in, 1.0)

    # Exterior pixels (v = 2*half_w, outside both edges)
    v_out = np.full(n, 2 * half_w)
    py_out  = _py(v_out)
    cpp_out = _cpp(v_out)
    np.testing.assert_array_equal(py_out,  0.0)
    np.testing.assert_array_equal(cpp_out, 0.0)


if __name__ == "__main__":
    runtests(__file__)
