import numpy as np

import danish
from test_helpers import timer


LSST_obsc_radii = {
    'M1_inner': 2.5580033095346875,
    'M2_outer': 4.502721059044802,
    'M2_inner': 2.3698531889709487,
    'M3_outer': 5.4353949343626216,
    'M3_inner': 1.1919725733251365,
    'L1_entrance': 7.692939426566589,
    'L1_exit': 8.103064894823262,
    'L2_entrance': 10.746925431763076,
    'L2_exit': 11.548732622162085,
    'Filter_entrance': 28.06952057721957,
    'Filter_exit': 30.895257933242576,
    'L3_entrance': 54.5631834759912,
    'L3_exit': 114.76715786850136
}

LSST_obsc_motion = {
    'M1_inner': 0.1517605552388959,
    'M2_outer': 16.818667026561727,
    'M2_inner': 16.818667026561727,
    'M3_outer': 53.2113063872138,
    'M3_inner': 53.2113063872138,
    'L1_entrance': 131.69949884635324,
    'L1_exit': 137.51151184228345,
    'L2_entrance': 225.63931108752732,
    'L2_exit': 236.8641351903567,
    'Filter_entrance': 801.6598843836333,
    'Filter_exit': 879.4647343264201,
    'L3_entrance': 1594.7432961792515,
    'L3_exit': 3328.637595923783
}


@timer
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


@timer
def test_LSST():
    """ Check that we can draw donuts for fiducial LSST optics.

    Note: This just tests that the code runs for now.  There's no check for
          accuracy.
    """
    import batoid
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0,0,0.0015])

    wavelength = 750e-9

    rng = np.random.default_rng(234)
    for _ in range(10):
        thr = np.sqrt(rng.uniform(0, 1.8**2))
        ph = rng.uniform(0, 2*np.pi)
        thx, thy = thr*np.cos(ph), thr*np.sin(ph)
        zref = batoid.analysis.zernikeTransverseAberration(
            telescope,
            np.deg2rad(thx),
            np.deg2rad(thy),
            wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            obsc_radii=LSST_obsc_radii, obsc_motion=LSST_obsc_motion,
            focal_length=10.31, pixel_scale=10e-6
        )

        img = factory.image(
            aberrations=zref*wavelength,
            thx=np.deg2rad(thx), thy=np.deg2rad(thy)
        )

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()


@timer
def test_LSST_aberrated():
    """ Check that we can draw donuts for fiducial LSST optics + additional
    Zernike aberrations.

    Note: This just tests that the code runs for now.  There's no check for
          accuracy.
    """
    import batoid
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0,0,0.0015])

    wavelength = 750e-9

    rng = np.random.default_rng(234)
    for _ in range(10):
        thr = np.sqrt(rng.uniform(0, 1.8**2))
        ph = rng.uniform(0, 2*np.pi)
        thx, thy = thr*np.cos(ph), thr*np.sin(ph)
        zref = batoid.analysis.zernikeTransverseAberration(
            telescope,
            np.deg2rad(thx),
            np.deg2rad(thy),
            wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )

        z = np.array(zref)
        z[4:22] += rng.uniform(-0.2, 0.2, size=18)
        z *= wavelength

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            obsc_radii=LSST_obsc_radii, obsc_motion=LSST_obsc_motion,
            focal_length=10.31, pixel_scale=10e-6
        )

        img = factory.image(
            aberrations=z,
            thx=np.deg2rad(thx), thy=np.deg2rad(thy)
        )

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()


if __name__ == "__main__":
    test_coord_roundtrip()
    test_LSST()
    test_LSST_aberrated()
