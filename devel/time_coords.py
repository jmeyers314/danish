import time

import numpy as np
import galsim


R_outer = 4.18
R_inner = 0.61*R_outer
focal_length = 10.31


def time_p2f():
    from danish.factory import _pupil_to_focal

    rng = np.random.default_rng(1234)
    rho = np.sqrt(rng.uniform(R_inner**2, R_outer**2, 10000))
    phi = rng.uniform(0, 2*np.pi, 10000)
    u = rho*np.cos(phi)
    v = rho*np.sin(phi)

    # Prime the caches
    zarr = np.zeros(67)
    zarr[4] = 31e-6
    zarr[5:] = 1e-8

    Z = galsim.zernike.Zernike(
        zarr,
        R_outer=R_outer, R_inner=R_inner
    )
    _ = _pupil_to_focal(
        u, v,
        Z = Z,
        focal_length=focal_length
    )

    N = 2000
    t0 = time.time()
    for _ in range(N):
        zarr[4] = rng.uniform(30e-6, 32e-6)
        zarr[5:] = rng.uniform(-1e-8, 1e-8, 62)
        Z = galsim.zernike.Zernike(
            zarr,
            R_outer=R_outer, R_inner=R_inner
        )
        Z1 = Z * focal_length
        _ = _pupil_to_focal(
            u, v,
            Z = Z1,
        )
    t1 = time.time()
    print(f"Time for _pupil_to_focal(): {(t1-t0)/N*1e3:.2f} ms")


def time_f2p():
    from danish.factory import _pupil_to_focal, _focal_to_pupil

    rng = np.random.default_rng(234)
    rho = np.sqrt(rng.uniform(R_inner**2, R_outer**2, 10000))
    phi = rng.uniform(0, 2*np.pi, 10000)
    u = rho*np.cos(phi)
    v = rho*np.sin(phi)

    # Prime the caches
    zarr = np.zeros(67)
    zarr[4] = 31e-6
    zarr[5:] = 1e-8

    Z = galsim.zernike.Zernike(
        zarr,
        R_outer=R_outer, R_inner=R_inner
    )
    x, y = _pupil_to_focal(
        u, v,
        Z = Z,
        focal_length=focal_length
    )
    # Shrink to make inversion easier
    x *= 0.5
    y *= 0.5
    _focal_to_pupil(x, y, Z=Z, focal_length=focal_length)

    N = 500
    t0 = time.time()
    for _ in range(N):
        zarr[4] = rng.uniform(30e-6, 32e-6)
        zarr[5:] = rng.uniform(-1e-8, 1e-8, 62)
        Z = galsim.zernike.Zernike(
            zarr,
            R_outer=R_outer, R_inner=R_inner
        )
        Z1 = Z * focal_length
        _ = _focal_to_pupil(x, y, Z=Z1)
    t1 = time.time()
    print(f"Time for _focal_to_pupil(): {(t1-t0)/N*1e3:.2f} ms")


def time_p2f_jac():
    from danish.factory import _pupil_focal_jacobian

    rng = np.random.default_rng(3456)
    rho = np.sqrt(rng.uniform(R_inner**2, R_outer**2, 10000))
    phi = rng.uniform(0, 2*np.pi, 10000)
    u = rho*np.cos(phi)
    v = rho*np.sin(phi)

    # Prime the caches
    zarr = np.zeros(67)
    zarr[4] = 31e-6
    zarr[5:] = 1e-8

    Z = galsim.zernike.Zernike(
        zarr,
        R_outer=R_outer, R_inner=R_inner
    )
    _ = _pupil_focal_jacobian(
        u, v,
        Z = Z,
        focal_length=focal_length
    )

    N = 2000
    t0 = time.time()
    for _ in range(N):
        zarr[4] = rng.uniform(30e-6, 32e-6)
        zarr[5:] = rng.uniform(-1e-8, 1e-8, 62)
        Z = galsim.zernike.Zernike(
            zarr,
            R_outer=R_outer, R_inner=R_inner
        )
        Z1 = Z * focal_length
        _ = _pupil_focal_jacobian(
            u, v,
            Z = Z1,
        )
    t1 = time.time()
    print(f"Time for _pupil_focal_jacobian(): {(t1-t0)/N*1e3:.2f} ms")


if __name__ == "__main__":
    time_p2f()
    time_f2p()
    time_p2f_jac()
