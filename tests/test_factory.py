import batoid
from contextlib import ExitStack
import os
import yaml
import numpy as np

import danish
from galsim.zernike import Zernike
from danish_test_helpers import timer, runtests

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
            mask_params=Rubin_obsc,
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
            mask_params=Rubin_obsc,
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
    of a scalar function.  This test uses an alternate API without the scalar
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
            mask_params=Rubin_obsc,
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

        # 90% of pixels (non-edge pixels basically) are within 0.02 of each other
        assert np.nanquantile(np.abs(img-img2), 0.9) < 0.02


@timer
def test_focal_plane_hits_fiducial():
    """Test that polynomial fit of focal plane ray hit locations (wrt the chief ray)
    matches the true focal plane ray hit locations."""

    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    # We mostly care about the transformation for donuts, so shift the camera
    shifted = fiducial.withGloballyShiftedOptic("LSSTCamera", [0,0,0.0015])
    wavelength = 622e-9
    focal_length = 10.31
    eps = 0.621

    rng = np.random.default_rng(987)

    for _ in range(10):
        thr = np.deg2rad(np.sqrt(rng.uniform(0, 1.8**2)))
        ph = rng.uniform(0, 2*np.pi)
        thx, thy = thr*np.cos(ph), thr*np.sin(ph)

        rays = batoid.RayVector.asPolar(
            optic=shifted,
            theta_x=thx, theta_y=thy,
            wavelength=wavelength,
            nrad=20, naz=120,
        )

        epRays = shifted.stopSurface.interact(rays.copy())
        u = epRays.x
        v = epRays.y
        focal = shifted.trace(rays.copy())

        chief = batoid.RayVector.fromStop(
            0, 0, shifted, wavelength=wavelength,
            theta_x=thx, theta_y=thy,
        )

        shifted.trace(chief)
        dx = focal.x - chief.x
        dy = focal.y - chief.y

        for order, tol_xy, tol_ta in [
            (11, 7e-2, 6e-1),
            (12, 2e-2, 6e-2),
            (13, 2e-3, 2e-2),
            (14, 6e-4, 9e-3),
        ]:
            x_offset, y_offset = batoid.zernikeXYAberrations(
                shifted,
                thx, thy,
                wavelength,
                nrad=20, naz=120, reference='chief',
                # nrad=80, naz=480, reference='chief',
                jmax=np.sum(np.arange(order)), eps=eps,
                include_vignetted=False
            )
            zx = Zernike(
                x_offset,
                R_outer=4.18, R_inner=4.18*eps,
            )
            zy = Zernike(
                y_offset,
                R_outer=4.18, R_inner=4.18*eps,
            )

            w = ~focal.vignetted
            dx1 = zx(u, v)
            dy1 = zy(u, v)

            rms_xy = np.sqrt(
                np.mean(
                    (dx-dx1)**2 + (dy-dy1)**2
                )
            )/10e-6
            # print(rms_xy)

            ddr1 = np.hypot(dx-dx1, dy-dy1)/10e-6  # pixels
            np.testing.assert_array_less(ddr1, tol_xy)

            zTA = batoid.zernikeTA(
                shifted,
                thx, thy,
                wavelength,
                nrad=20, naz=120, reference='chief',
                jmax=np.sum(np.arange(order)), eps=eps,
                focal_length=focal_length,
            ) * wavelength

            zz = Zernike(
                zTA,
                R_outer=4.18, R_inner=4.18*eps,
            )
            zzx = -zz.gradX*focal_length
            zzy = -zz.gradY*focal_length

            dx2 = zzx(u, v)
            dy2 = zzy(u, v)

            rms_ta =np.sqrt(
                np.mean(
                    (dx-dx2)**2 + (dy-dy2)**2
                )
            )/10e-6
            # print(rms_ta)

            ddr2 = np.hypot(dx-dx2, dy-dy2)/10e-6  # pixels
            np.testing.assert_array_less(ddr2, tol_ta)

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

            # fig, axes = plt.subplots(ncols=2, figsize=(10, 4.5))
            # for ax, val, title in zip(
            #     axes,
            #     [dx-dx1, dy-dy1],
            #     ["dx", "dy"]
            # ):
            #     colorbar(
            #         ax.scatter(
            #             u[w], v[w], c=val[w]/10e-6, cmap='bwr', vmin=-0.01, vmax=0.01, s=5
            #         )
            #     )
            #     ax.set_aspect('equal')
            #     ax.set_title(title)
            # fig.suptitle(
            #     f"focal plane hit residuals (pixels)\n\n"
            #     f"Using zernikeXYAberrations order={order}, j={np.sum(np.arange(order))}"
            # )
            # fig.tight_layout()
            # plt.show()

            # fig, axes = plt.subplots(ncols=2, figsize=(10, 4.5))
            # for ax, val, title in zip(
            #     axes,
            #     [dx-dx2, dy-dy2],
            #     ["dx", "dy"]
            # ):
            #     colorbar(
            #         ax.scatter(
            #             u[w], v[w], c=val[w]/10e-6, cmap='bwr', vmin=-0.01, vmax=0.01, s=5
            #         )
            #     )
            #     ax.set_aspect('equal')
            #     ax.set_title(title)
            # fig.suptitle(
            #     f"focal plane hit residuals (pixels)\n\n"
            #     f"Using zernikeTA order={order}, j={np.sum(np.arange(order))}"
            # )
            # fig.tight_layout()
            # plt.show()


@timer
def test_focal_plane_hits_perturbed(run_slow):
    """Test that polynomial model for ray aberrations produces the correct
    ray hit locations on the focal plane.
    """
    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    # We mostly care about the transformation for donuts, so shift the camera
    shifted = fiducial.withGloballyShiftedOptic("LSSTCamera", [0,0,0.0015])
    wavelength = 622e-9
    focal_length = 10.31
    eps = 0.621

    rng = np.random.default_rng(987)

    # Loop over a few perturbations
    # Use a phase screen as the perturbation
    with ExitStack() as stack:
        if run_slow:
            from tqdm import tqdm
            pbar = stack.enter_context(tqdm(total=400))
        else:
            pbar = None
        for _ in range(10):
            amplitude = 100e-9  # ~100 nm RMS perturbations
            jmax = 22
            coefs = rng.uniform(-1, 1, size=jmax+1)*amplitude/np.sqrt(jmax+1)
            coefs[:4] = 0.0  # No PTT
            # Perturb both the fiducial optics and the shifted optics.
            # Use the perturbed+shifted optics as the "truth" to match.
            # Use the perturbed+fiducial optics to get the perturbation
            # to Zernike coefficients.
            perturbed_fiducial = fiducial.withInsertedOptic(
                before="M1",
                item=batoid.OPDScreen(
                    name='Screen',
                    surface=batoid.Plane(),
                    screen=batoid.Zernike(
                        coefs,
                        R_outer=4.18,
                        R_inner=4.18*eps,
                    ),
                    coordSys=fiducial.stopSurface.coordSys,
                    obscuration=fiducial['M1'].obscuration,
                )
            )
            perturbed_shifted = shifted.withInsertedOptic(
                before="M1",
                item=batoid.OPDScreen(
                    name='Screen',
                    surface=batoid.Plane(),
                    screen=batoid.Zernike(
                        coefs,
                        R_outer=4.18,
                        R_inner=4.18*eps,
                    ),
                    coordSys=shifted.stopSurface.coordSys,
                    obscuration=shifted['M1'].obscuration,
                )
            )

            # Now loop over some field angles
            for __ in range(10):
                thr = np.deg2rad(np.sqrt(rng.uniform(0, 1.8**2)))
                ph = rng.uniform(0, 2*np.pi)
                thx, thy = thr*np.cos(ph), thr*np.sin(ph)

                rays = batoid.RayVector.asPolar(
                    optic=shifted,
                    theta_x=thx, theta_y=thy,
                    wavelength=wavelength,
                    nrad=20, naz=120,
                )

                epRays = shifted.stopSurface.interact(rays.copy())
                u = epRays.x
                v = epRays.y
                focal = perturbed_shifted.trace(rays.copy())

                chief = batoid.RayVector.fromStop(
                    0, 0, shifted, wavelength=wavelength,
                    theta_x=thx, theta_y=thy,
                )

                perturbed_shifted.trace(chief)
                dx = focal.x - chief.x
                dy = focal.y - chief.y

                for order, tol_xy, tol_ta in [
                    (11, 6e-2, 5e-1),
                    (12, 2e-2, 7e-2),
                    (13, 5e-3, 6e-2),
                    (14, 4e-3, 6e-2),
                ]:
                    # Get "intrinsic" zernikes from the unperturbed optics
                    x_offset, y_offset = batoid.zernikeXYAberrations(
                        shifted,
                        thx, thy,
                        wavelength,
                        nrad=20, naz=120, reference='chief',
                        # nrad=80, naz=480, reference='chief',
                        jmax=np.sum(np.arange(order)), eps=eps,
                        include_vignetted=False
                    )
                    zx = Zernike(
                        x_offset,
                        R_outer=4.18, R_inner=4.18*eps,
                    )
                    zy = Zernike(
                        y_offset,
                        R_outer=4.18, R_inner=4.18*eps,
                    )

                    # Get the perturbation Zernikes from perturbed in-focus optics.
                    # Use reference sphere Zernikes.
                    zfiducial = batoid.zernike(
                        fiducial,
                        thx, thy,
                        wavelength,
                        nx=256, reference='chief',
                        jmax=np.sum(np.arange(order)), eps=eps,
                    )*wavelength
                    zperturbed = batoid.zernike(
                        perturbed_fiducial,
                        thx, thy,
                        wavelength,
                        nx=256, reference='chief',
                        jmax=np.sum(np.arange(order)), eps=eps,
                    )*wavelength
                    dz = zperturbed - zfiducial
                    zperturbation = Zernike(
                        dz,
                        R_outer=4.18, R_inner=4.18*eps,
                    )

                    w = ~focal.vignetted
                    dx1 = (zx - zperturbation.gradX*focal_length)(u, v)
                    dy1 = (zy - zperturbation.gradY*focal_length)(u, v)

                    rms_xy = np.sqrt(
                        np.mean(
                            (dx-dx1)**2 + (dy-dy1)**2
                        )
                    )/10e-6
                    # print(rms_xy)

                    ddr1 = np.hypot(dx-dx1, dy-dy1)/10e-6  # pixels
                    np.testing.assert_array_less(ddr1, tol_xy)

                    # Now try the TA method
                    zTA = batoid.zernikeTA(
                        shifted,
                        thx, thy,
                        wavelength,
                        nrad=20, naz=120, reference='chief',
                        jmax=np.sum(np.arange(order)), eps=eps,
                        focal_length=focal_length,
                    ) * wavelength

                    zz = Zernike(
                        zTA+dz,
                        R_outer=4.18, R_inner=4.18*eps,
                    )
                    zzx = -zz.gradX*focal_length
                    zzy = -zz.gradY*focal_length

                    dx2 = zzx(u, v)
                    dy2 = zzy(u, v)

                    rms_ta =np.sqrt(
                        np.mean(
                            (dx-dx2)**2 + (dy-dy2)**2
                        )
                    )/10e-6
                    # print(rms_ta)

                    ddr2 = np.hypot(dx-dx2, dy-dy2)/10e-6  # pixels
                    np.testing.assert_array_less(ddr2, tol_ta)

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

                    # fig, axes = plt.subplots(ncols=2, figsize=(10, 4.5))
                    # for ax, val, title in zip(
                    #     axes,
                    #     [dx-dx1, dy-dy1],
                    #     ["dx", "dy"]
                    # ):
                    #     colorbar(
                    #         ax.scatter(
                    #             u[w], v[w], c=val[w]/10e-6, cmap='bwr', vmin=-0.01, vmax=0.01, s=5
                    #         )
                    #     )
                    #     ax.set_aspect('equal')
                    #     ax.set_title(title)
                    # fig.suptitle(
                    #     f"focal plane hit residuals (pixels)\n\n"
                    #     f"Using zernikeXYAberrations order={order}, j={np.sum(np.arange(order))}"
                    # )
                    # fig.tight_layout()
                    # plt.show()

                    # fig, axes = plt.subplots(ncols=2, figsize=(10, 4.5))
                    # for ax, val, title in zip(
                    #     axes,
                    #     [dx-dx2, dy-dy2],
                    #     ["dx", "dy"]
                    # ):
                    #     colorbar(
                    #         ax.scatter(
                    #             u[w], v[w], c=val[w]/10e-6, cmap='bwr', vmin=-0.01, vmax=0.01, s=5
                    #         )
                    #     )
                    #     ax.set_aspect('equal')
                    #     ax.set_title(title)
                    # fig.suptitle(
                    #     f"focal plane hit residuals (pixels)\n\n"
                    #     f"Using zernikeTA order={order}, j={np.sum(np.arange(order))}"
                    # )
                    # fig.tight_layout()
                    # plt.show()

                    if pbar:
                        pbar.update()


if __name__ == "__main__":
    runtests(__file__)
