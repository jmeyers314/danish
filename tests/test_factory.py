import os
import yaml
import numpy as np

import danish
from galsim.zernike import Zernike
from test_helpers import timer

Rubin_obsc = yaml.safe_load(open(os.path.join(danish.datadir, 'RubinObsc.yaml')))


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
        zref = batoid.zernikeTA(
            telescope,
            np.deg2rad(thx),
            np.deg2rad(thy),
            wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            obsc_radii=Rubin_obsc['radii'],
            obsc_centers=Rubin_obsc['centers'],
            obsc_th_mins=Rubin_obsc['th_mins'],
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
        zref = batoid.zernikeTA(
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
            obsc_radii=Rubin_obsc['radii'],
            obsc_centers=Rubin_obsc['centers'],
            obsc_th_mins=Rubin_obsc['th_mins'],
            focal_length=10.31, pixel_scale=10e-6
        )

        img = factory.image(
            aberrations=z,
            thx=np.deg2rad(thx), thy=np.deg2rad(thy)
        )

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()


@timer
def test_factory_offsets():
    rng = np.random.default_rng(192837465)
    for _ in range(10):
        R_outer = rng.uniform(2.1, 4.2)
        eps = rng.uniform(0.3, 0.6)
        focal_length = rng.uniform(10.0, 20.0)
        R_inner = R_outer * eps

        factory = danish.DonutFactory(
            R_outer=R_outer, R_inner=R_inner,
            focal_length=focal_length,
            pixel_scale=10e-6
        )

        for __ in range(10):
            aberrations = np.zeros(22)
            aberrations[4] = rng.uniform(20.0, 30.0)
            aberrations[5:] = rng.uniform(-0.2, 0.2, size=17)
            Z = Zernike(
                aberrations,
                R_outer=R_outer,
                R_inner=R_inner,
            )

            r = np.sqrt(rng.uniform(R_inner**2, R_outer**2, size=1000))
            ph = rng.uniform(0, 2*np.pi, size=1000)
            u, v = r*np.cos(ph), r*np.sin(ph)

            x, y = danish.pupil_to_focal(u, v, Z=Z, focal_length=focal_length)
            x1, y1 = danish.pupil_to_focal(
                u, v, Z=Z*0,
                focal_length=focal_length,
                x_offset=-Z.gradX*focal_length, y_offset=-Z.gradY*focal_length
            )
            np.testing.assert_allclose(x, x1, atol=1e-12, rtol=1e-14)
            np.testing.assert_allclose(y, y1, atol=1e-12, rtol=1e-14)

            u1, v1 = danish.focal_to_pupil(
                x, y, Z=Z, focal_length=focal_length,
                tol=1e-12
            )
            u2, v2 = danish.focal_to_pupil(
                x1, y1, Z=Z*0,
                focal_length=focal_length,
                x_offset=-Z.gradX*focal_length, y_offset=-Z.gradY*focal_length,
                tol=1e-12
            )
            np.testing.assert_allclose(u, u1, atol=1e-12, rtol=1e-14)
            np.testing.assert_allclose(v, v1, atol=1e-12, rtol=1e-14)
            np.testing.assert_allclose(u, u2, atol=1e-12, rtol=1e-14)
            np.testing.assert_allclose(v, v2, atol=1e-12, rtol=1e-14)

            # Test images too
            img = factory.image(
                Z=Z,
            )
            img2 = factory.image(
                Z=Z*0,
                x_offset=-Z.gradX*focal_length, y_offset=-Z.gradY*focal_length
            )
            np.testing.assert_allclose(img, img2, atol=1e-12, rtol=1e-14)


@timer
def test_curly_offsets():
    """ The distorted transformation from pupil to focal for Rubin isn't
    actually curl-free, so we make a small error when modeling as a gradient
    of a scalar function.  This test uses an alternate API with out the scalar
    gradient assumption and checks that the donut images are similar.  (The
    error is known to be small).
    """

    import batoid
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0,0,-0.0015])

    wavelength = 750e-9

    rng = np.random.default_rng(234)
    for _ in range(10):
        thr = np.sqrt(rng.uniform(0, 1.8**2))
        ph = rng.uniform(0, 2*np.pi)
        thx, thy = thr*np.cos(ph), thr*np.sin(ph)

        zref = batoid.zernikeTA(
            telescope,
            np.deg2rad(thx),
            np.deg2rad(thy),
            wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )

        dz = np.zeros(22)
        dz[4:22] = rng.uniform(-0.2, 0.2, size=18)
        z = np.array(zref)
        z[4:22] += dz[4:22]
        z *= wavelength

        x_offset, y_offset = batoid.zernikeXYAberrations(
            telescope,
            np.deg2rad(thx),
            np.deg2rad(thy),
            wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.612
        )
        zx = Zernike(
            x_offset,
            R_outer=4.18, R_inner=4.18*0.61,
        )
        zy = Zernike(
            y_offset,
            R_outer=4.18, R_inner=4.18*0.61,
        )

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            obsc_radii=Rubin_obsc['radii'],
            obsc_centers=Rubin_obsc['centers'],
            obsc_th_mins=Rubin_obsc['th_mins'],
            focal_length=10.31, pixel_scale=10e-6
        )

        img = factory.image(
            aberrations=z,
            thx=np.deg2rad(thx), thy=np.deg2rad(thy)
        )
        img2 = factory.image(
            aberrations=dz*wavelength,
            thx=np.deg2rad(thx), thy=np.deg2rad(thy),
            x_offset=zx, y_offset=zy
        )


        img[img == 0] = np.nan
        img2[img2 == 0] = np.nan

        # import matplotlib.pyplot as plt
        # def colorbar(mappable):
        #     from mpl_toolkits.axes_grid1 import make_axes_locatable
        #     import matplotlib.pyplot as plt
        #     last_axes = plt.gca()
        #     ax = mappable.axes
        #     fig = ax.figure
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     cbar = fig.colorbar(mappable, cax=cax)
        #     plt.sca(last_axes)
        #     return cbar
        # fig, axes = plt.subplots(ncols=3)
        # colorbar(axes[0].imshow(img, origin='lower'))
        # colorbar(axes[1].imshow(img2, origin='lower'))
        # colorbar(axes[2].imshow(img-img2, origin='lower', vmin=-0.01, vmax=0.01, cmap='RdBu'))

        # fig.tight_layout()
        # plt.show()

        # 90% of pixels (non-edge pixels basically) are within 0.01 of each other
        assert np.nanquantile(np.abs(img-img2), 0.9) < 0.01


if __name__ == "__main__":
    test_coord_roundtrip()
    test_LSST()
    test_LSST_aberrated()
    test_factory_offsets()
    test_curly_offsets()
