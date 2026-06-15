"""test_cpp_sync.py

Verify that the scalar Python reference implementations in factory.py stay
in sync with their C++ counterparts in _danish.so.

Each test generates a batch of inputs, evaluates the Python _1 scalar
function for each point, then evaluates the C++ vectorised wrapper on the
whole batch and compares.  test_enclosed_strut_wide additionally checks the
wide-strut code path (pixels far from both edges) that was previously buggy.

The triangle-clipping tests (test_clip_triangle_*) verify the pure-Python
_clip_triangle_to_circle and _triangle_relation_to_circle helpers against the
C++ _clip_triangles_to_circle_cpp path exercised by apply_circle_obscurations.
"""

import numpy as np

from danish.factory import (
    _pixel_frac_1,
    _enclosed_circle_1,
    _enclosed_strut_1,
    _pixel_frac,
    _enclosed_fraction,
    _strut_masked_fraction,
    DonutTriangleFactory,
)
from danish._danish import clip_triangles_to_circle as _clip_triangles_to_circle_cpp
from danish_test_helpers import timer, runtests

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
    rng = np.random.default_rng(57721)
    n = 400
    u0, v0 = 1.2, -0.8
    angle = 0.7
    sth0, cth0 = float(np.sin(angle)), float(np.cos(angle))

    u1 = rng.uniform(-4, 4, n).astype(np.float64)
    v1 = rng.uniform(-4, 4, n).astype(np.float64)
    x1 = rng.uniform(-0.05, 0.05, n).astype(np.float64)
    y1 = rng.uniform(-0.05, 0.05, n).astype(np.float64)
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
    rng = np.random.default_rng(31416)
    n = 400
    u0, v0, radius = 0.5, -0.3, 2.0

    u = rng.uniform(-4, 4, n).astype(np.float64)
    v = rng.uniform(-4, 4, n).astype(np.float64)
    x = rng.uniform(-0.05, 0.05, n).astype(np.float64)
    y = rng.uniform(-0.05, 0.05, n).astype(np.float64)
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
    rng = np.random.default_rng(27183)
    n = 400
    # Horizontal strut centred at origin, edges at v = ±0.3
    # Direction along u: sth=0, cth=1
    u1, v1, sth1, cth1 =  0.0,  0.3, 0.0, 1.0
    u2, v2, sth2, cth2 =  0.0, -0.3, 0.0, 1.0
    length = 4.0

    u = rng.uniform(-4, 4, n).astype(np.float64)
    v = rng.uniform(-4, 4, n).astype(np.float64)
    x = rng.uniform(-0.05, 0.05, n).astype(np.float64)
    y = rng.uniform(-0.05, 0.05, n).astype(np.float64)
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


@timer
def test_clip_triangle_to_circle_keep_inside():
    """_clip_triangle_to_circle (Python) must match the C++ path for keep_inside=True."""
    rng = np.random.default_rng(11235)
    center = np.array([0.0, 0.0])
    radius = 2.0
    tol = 1e-12

    py_areas = []
    cpp_areas = []

    for _ in range(200):
        tri = rng.uniform(-3.0, 3.0, (3, 2))

        # Python scalar path
        clipped = DonutTriangleFactory._clip_triangle_to_circle(
            tri, center, radius, keep_inside=True
        )
        py_area = sum(
            0.5 * abs(
                (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
                - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0])
            )
            for t in clipped
        )
        py_areas.append(py_area)

        # C++ path via the batch wrapper (one triangle at a time)
        tri_batch = np.ascontiguousarray(tri[None], dtype=np.float64)
        out_buf = np.empty((3, 3, 2), dtype=np.float64)
        n_rem = np.zeros(1, dtype=np.int32)
        n_clip = np.zeros(1, dtype=np.int32)
        ntri_out = _clip_triangles_to_circle_cpp(
            tri_batch.ctypes.data, 1,
            float(center[0]), float(center[1]), float(radius),
            1,          # keep_inside
            float(tol),
            out_buf.ctypes.data,
            n_rem.ctypes.data,
            n_clip.ctypes.data,
        )
        cpp_area = sum(
            0.5 * abs(
                (out_buf[k, 1, 0] - out_buf[k, 0, 0]) * (out_buf[k, 2, 1] - out_buf[k, 0, 1])
                - (out_buf[k, 1, 1] - out_buf[k, 0, 1]) * (out_buf[k, 2, 0] - out_buf[k, 0, 0])
            )
            for k in range(ntri_out)
        )
        cpp_areas.append(cpp_area)

    np.testing.assert_allclose(py_areas, cpp_areas, rtol=1e-10, atol=1e-14)


@timer
def test_clip_triangle_to_circle_keep_outside():
    """_clip_triangle_to_circle (Python) must match the C++ path for keep_inside=False."""
    rng = np.random.default_rng(31415)
    center = np.array([0.5, -0.3])
    radius = 1.5
    tol = 1e-12

    py_areas = []
    cpp_areas = []

    for _ in range(200):
        tri = rng.uniform(-3.0, 3.0, (3, 2))

        clipped = DonutTriangleFactory._clip_triangle_to_circle(
            tri, center, radius, keep_inside=False
        )
        py_area = sum(
            0.5 * abs(
                (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
                - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0])
            )
            for t in clipped
        )
        py_areas.append(py_area)

        tri_batch = np.ascontiguousarray(tri[None], dtype=np.float64)
        out_buf = np.empty((3, 3, 2), dtype=np.float64)
        n_rem = np.zeros(1, dtype=np.int32)
        n_clip = np.zeros(1, dtype=np.int32)
        ntri_out = _clip_triangles_to_circle_cpp(
            tri_batch.ctypes.data, 1,
            float(center[0]), float(center[1]), float(radius),
            0,          # keep_inside=False
            float(tol),
            out_buf.ctypes.data,
            n_rem.ctypes.data,
            n_clip.ctypes.data,
        )
        cpp_area = sum(
            0.5 * abs(
                (out_buf[k, 1, 0] - out_buf[k, 0, 0]) * (out_buf[k, 2, 1] - out_buf[k, 0, 1])
                - (out_buf[k, 1, 1] - out_buf[k, 0, 1]) * (out_buf[k, 2, 0] - out_buf[k, 0, 0])
            )
            for k in range(ntri_out)
        )
        cpp_areas.append(cpp_area)

    np.testing.assert_allclose(py_areas, cpp_areas, rtol=1e-10, atol=1e-14)


@timer
def test_triangle_relation_to_circle():
    """_triangle_relation_to_circle returns 'keep', 'discard', or 'partial' correctly."""
    center = np.array([0.0, 0.0])
    radius = 1.0

    # Triangle fully inside (all vertices within radius)
    tri_in = np.array([[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0]])
    assert DonutTriangleFactory._triangle_relation_to_circle(
        tri_in, center, radius, keep_inside=True) == 'keep'
    assert DonutTriangleFactory._triangle_relation_to_circle(
        tri_in, center, radius, keep_inside=False) == 'discard'

    # Triangle fully outside (all vertices beyond radius)
    tri_out = np.array([[2.0, 0.0], [3.0, 0.0], [2.5, 1.0]])
    assert DonutTriangleFactory._triangle_relation_to_circle(
        tri_out, center, radius, keep_inside=True) == 'discard'
    assert DonutTriangleFactory._triangle_relation_to_circle(
        tri_out, center, radius, keep_inside=False) == 'keep'

    # Triangle straddling the boundary
    tri_partial = np.array([[0.5, 0.0], [1.5, 0.0], [1.0, 1.0]])
    assert DonutTriangleFactory._triangle_relation_to_circle(
        tri_partial, center, radius, keep_inside=True) == 'partial'
    assert DonutTriangleFactory._triangle_relation_to_circle(
        tri_partial, center, radius, keep_inside=False) == 'partial'


@timer
def test_clip_triangle_output_vertices_in_correct_region():
    """All output vertices of _clip_triangle_to_circle must lie in the kept region.

    Note: area(inside) + area(outside) is NOT guaranteed to equal area(original)
    because the algorithm uses chords (straight lines) rather than arcs at the
    circle boundary.  Instead we verify the geometric region membership of every
    output vertex, which is the property the algorithm actually guarantees.
    """
    rng = np.random.default_rng(57721)
    center = np.array([0.2, -0.1])
    radius = 1.8
    tol = 1e-9  # boundary tolerance

    def _area(tris):
        return sum(
            0.5 * abs(
                (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
                - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0])
            )
            for t in tris
        )

    for _ in range(300):
        tri = rng.uniform(-3.0, 3.0, (3, 2))
        orig_area = 0.5 * abs(
            (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
            - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0])
        )

        inside_tris  = DonutTriangleFactory._clip_triangle_to_circle(tri, center, radius, keep_inside=True)
        outside_tris = DonutTriangleFactory._clip_triangle_to_circle(tri, center, radius, keep_inside=False)

        # Clipped output area must not exceed the original
        assert _area(inside_tris)  <= orig_area + 1e-12
        assert _area(outside_tris) <= orig_area + 1e-12

        # Every vertex of every inside triangle must be within radius + tol
        for t in inside_tris:
            r = np.hypot(t[:, 0] - center[0], t[:, 1] - center[1])
            assert np.all(r <= radius + tol), f"inside vertex outside circle: r={r.max():.6f} > {radius}"

        # Every vertex of every outside triangle must be outside radius - tol
        for t in outside_tris:
            r = np.hypot(t[:, 0] - center[0], t[:, 1] - center[1])
            assert np.all(r >= radius - tol), f"outside vertex inside circle: r={r.min():.6f} < {radius}"


if __name__ == "__main__":
    runtests(__file__)
