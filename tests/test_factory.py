from danish.factory import focal_to_pupil
import numpy as np

import danish


def test_coord_roundtrip():
    """Check that pupil -> focal -> pupil -> focal round trips.
    """
    rng = np.random.default_rng(123)
    for _ in range(10):
        R_outer = rng.uniform(2.1, 4.2)
        eps = rng.uniform(0.3, 0.6)
        focal_length = rng.uniform(10.0, 20.0)
        R_inner = R_outer * eps
        for __ in range(10):
            aberrations = np.zeros(22)
            aberrations[4] = rng.uniform(20.0, 30.0)
            aberrations[5:] = rng.uniform(-0.2, 0.2, size=17)
            r = np.sqrt(rng.uniform(R_inner**2, R_outer**2, size=1000))
            ph = rng.uniform(0, 2*np.pi, size=1000)
            u, v = r*np.cos(ph), r*np.sin(ph)
            x, y = danish.pupil_to_focal(
                u, v,
                aberrations=aberrations,
                R_outer=R_outer,
                R_inner=R_inner,
                focal_length=focal_length
            )
            u1, v1 = danish.focal_to_pupil(
                x, y,
                aberrations=aberrations,
                R_outer=R_outer,
                R_inner=R_inner,
                focal_length=focal_length,
                tol=1e-12
            )
            np.testing.assert_allclose(u, u1, rtol=0, atol=1e-12)
            np.testing.assert_allclose(v, v1, rtol=0, atol=1e-12)

            x1, y1 = danish.pupil_to_focal(
                u1, v1,
                aberrations=aberrations,
                R_outer=R_outer,
                R_inner=R_inner,
                focal_length=focal_length,
            )
            np.testing.assert_allclose(x, x1, rtol=0, atol=1e-12)
            np.testing.assert_allclose(y, y1, rtol=0, atol=1e-12)


if __name__ == "__main__":
    test_coord_roundtrip()
