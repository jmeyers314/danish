import danish
import numpy as np
from numpy.typing import NDArray
from danish_test_helpers import timer, runtests
from galsim import Moffat, Kolmogorov


def moments(x: NDArray, y: NDArray, w: NDArray, a: int, b: int) -> float:
    return np.sum(w * x**a * y**b)


@timer
def test_gq_points_moments():
    """Test that the gq_points reproduce Gaussian moments."""
    x, y, w = danish.utils.gq_points(nrad=3, naz=8, kfold=8)

    # 0th order
    np.testing.assert_allclose(moments(x, y, w, 0, 0), 1.0, atol=1e-15, rtol=0)
    # 1st order
    np.testing.assert_allclose(moments(x, y, w, 1, 0), 0.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 0, 1), 0.0, atol=1e-15, rtol=0)
    # 2nd order
    np.testing.assert_allclose(moments(x, y, w, 2, 0), 1.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 0, 2), 1.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 1, 1), 0.0, atol=1e-14, rtol=0)
    # 3rd order
    np.testing.assert_allclose(moments(x, y, w, 3, 0), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 0, 3), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 2, 1), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 1, 2), 0.0, atol=1e-14, rtol=0)
    # 4th order
    np.testing.assert_allclose(moments(x, y, w, 4, 0), 3.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 0, 4), 3.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 3, 1), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 1, 3), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 2, 2), 1.0, atol=1e-14, rtol=0)
    # 5th order
    np.testing.assert_allclose(moments(x, y, w, 5, 0), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 0, 5), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 4, 1), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 1, 4), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 3, 2), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 2, 3), 0.0, atol=1e-14, rtol=0)
    # 6th order
    np.testing.assert_allclose(moments(x, y, w, 6, 0), 15.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 0, 6), 15.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 5, 1), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 1, 5), 0.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 4, 2), 3.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 2, 4), 3.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 3, 3), 0.0, atol=1e-14, rtol=0)

    # Check a covariance other than identity
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    x, y, w = danish.utils.gq_points(nrad=3, naz=8, cov=cov, kfold=8)
    # 0th order
    np.testing.assert_allclose(moments(x, y, w, 0, 0), 1.0, atol=1e-15, rtol=0)
    # 1st order
    np.testing.assert_allclose(moments(x, y, w, 1, 0), 0.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 0, 1), 0.0, atol=1e-15, rtol=0)
    # 2nd order
    np.testing.assert_allclose(moments(x, y, w, 2, 0), cov[0, 0], atol=1e-15, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 0, 2), cov[1, 1], atol=1e-15, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 1, 1), cov[0, 1], atol=1e-15, rtol=0)
    # 3rd order
    np.testing.assert_allclose(moments(x, y, w, 3, 0), 0.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 0, 3), 0.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 2, 1), 0.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(moments(x, y, w, 1, 2), 0.0, atol=1e-15, rtol=0)


if __name__ == "__main__":
    runtests(__file__)
