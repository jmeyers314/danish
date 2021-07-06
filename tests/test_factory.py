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


def test_LSST():
    """ Check that we can draw donuts for fiducial LSST optics.

    Note: This just tests that the code runs for now.  There's no check for
          accuracy.
    """
    import batoid
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0,0,0.0015])

    obsc_radii = {
        'M1_inner': 2.5498,
        'M2_inner': 2.3698999752679404,
        'M2_outer': 4.502809953009087,
        'M3_inner': 1.1922312943631603,
        'M3_outer': 5.436574702296011,
        'L1_entrance': 7.697441260764198,
        'L1_exit': 8.106852624652701,
        'L2_entrance': 10.748915941599885,
        'L2_exit': 11.5564127895276,
        'Filter_entrance': 28.082220873785978,
        'Filter_exit': 30.91023954045243,
        'L3_entrance': 54.67312185149621,
        'L3_exit': 114.58705556485711
    }
    obsc_motion = {
        'M1_inner': 0.0,
        'M2_inner': 16.8188788239707,
        'M2_outer': 16.8188788239707,
        'M3_inner': 53.22000661238318,
        'M3_outer': 53.22000661238318,
        'L1_entrance': 131.76650078100135,
        'L1_exit': 137.57031952814913,
        'L2_entrance': 225.6949885074127,
        'L2_exit': 237.01739037674315,
        'Filter_entrance': 802.0137451419788,
        'Filter_exit': 879.8810309773828,
        'L3_entrance': 1597.8959863335774,
        'L3_exit': 3323.60145194633
    }

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
            obsc_radii=obsc_radii, obsc_motion=obsc_motion,
            focal_length=10.31, pixel_scale=10e-6
        )

        img = factory.image(
            aberrations=zref*wavelength,
            thx=np.deg2rad(thx), thy=np.deg2rad(thy)
        )

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()


def test_LSST_aberrated():
    """ Check that we can draw donuts for fiducial LSST optics + additional
    Zernike aberrations.

    Note: This just tests that the code runs for now.  There's no check for
          accuracy.
    """
    import batoid
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0,0,0.0015])

    obsc_radii = {
        'M1_inner': 2.5498,
        'M2_inner': 2.3698999752679404,
        'M2_outer': 4.502809953009087,
        'M3_inner': 1.1922312943631603,
        'M3_outer': 5.436574702296011,
        'L1_entrance': 7.697441260764198,
        'L1_exit': 8.106852624652701,
        'L2_entrance': 10.748915941599885,
        'L2_exit': 11.5564127895276,
        'Filter_entrance': 28.082220873785978,
        'Filter_exit': 30.91023954045243,
        'L3_entrance': 54.67312185149621,
        'L3_exit': 114.58705556485711
    }
    obsc_motion = {
        'M1_inner': 0.0,
        'M2_inner': 16.8188788239707,
        'M2_outer': 16.8188788239707,
        'M3_inner': 53.22000661238318,
        'M3_outer': 53.22000661238318,
        'L1_entrance': 131.76650078100135,
        'L1_exit': 137.57031952814913,
        'L2_entrance': 225.6949885074127,
        'L2_exit': 237.01739037674315,
        'Filter_entrance': 802.0137451419788,
        'Filter_exit': 879.8810309773828,
        'L3_entrance': 1597.8959863335774,
        'L3_exit': 3323.60145194633
    }

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

        z22_aberration = np.zeros(22)
        z22_aberration[4:] = rng.uniform(-0.1, 0.1, size=18)

        z = np.array(zref)
        z[4:22] += rng.uniform(-0.2, 0.2, size=18)
        z *= wavelength

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            obsc_radii=obsc_radii, obsc_motion=obsc_motion,
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
