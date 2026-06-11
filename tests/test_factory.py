import batoid
from contextlib import ExitStack
import os
import yaml
import numpy as np

import danish
from galsim.zernike import Zernike
from danish_test_helpers import timer, runtests

Rubin_obsc = danish.load_mask_params("RubinObsc.yaml")


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
def test_bandpass_filter():
    """Check that bandpass_filter='r' runs and produces a different image than
    no filter.
    """
    import batoid
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0,0,0.0015])

    wavelength = 750e-9
    thx, thy = np.deg2rad(0.5), np.deg2rad(0.3)

    zref = batoid.zernikeTA(
        telescope,
        thx, thy,
        wavelength,
        nrad=20, naz=120, reference='chief',
        jmax=66, eps=0.61
    )
    aberrations = zref * wavelength

    factory_no_filter = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6
    )
    factory_r = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
        bandpass_filter='r'
    )

    img_no_filter = factory_no_filter.image(aberrations=aberrations, thx=thx, thy=thy)
    img_r = factory_r.image(aberrations=aberrations, thx=thx, thy=thy)

    # Both images should be non-trivial
    assert np.any(img_no_filter != 0)
    assert np.any(img_r != 0)
    # AOI correction should change the result
    assert not np.allclose(img_no_filter, img_r)


@timer
def test_thruput_interpolation():
    """Test _load_thruput_by_aoi clamping and interpolation without drawing images."""
    # Minimal factory — no batoid needed.
    f = danish.DonutFactory(bandpass_filter='r')

    def load(tbb=6000, am=1.5):
        return f._load_thruput_by_aoi('r', tbb, am)['value']

    # Off-grid airmass: 1.49999 should be indistinguishable from 1.5.
    assert np.allclose(load(am=1.5), load(am=1.49999), atol=1e-4)

    # Clamping below range: airmass=0.9 → same as airmass=1.0 (grid minimum).
    assert np.allclose(load(am=0.9), load(am=1.0))

    # Clamping above range: airmass=3.0 → same as airmass=2.5 (grid maximum).
    assert np.allclose(load(am=3.0), load(am=2.5))

    # Clamping Tbb below range: 3000 K → same as 4000 K (grid minimum).
    assert np.allclose(load(tbb=3000), load(tbb=4000))

    # Clamping Tbb above range: 12000 K → same as 10000 K (grid maximum).
    assert np.allclose(load(tbb=12000), load(tbb=10000))

    # Off-grid Tbb interpolation: result at midpoint should lie between endpoints.
    t_lo = load(tbb=7600)
    t_hi = load(tbb=7800)
    t_mid = load(tbb=7700)
    assert np.all((np.minimum(t_lo, t_hi) <= t_mid + 1e-12) &
                  (t_mid <= np.maximum(t_lo, t_hi) + 1e-12))


@timer
def test_pupil_R_inner():
    """Check that pupil_R_inner early exclusion doesn't change the donut image.

    pupil_R_inner controls both early exclusion and flux normalization.  Two
    factories with different pupil_R_inner values will produce images that
    differ only by a known normalization factor.  After undoing the
    normalization, the raw surface-brightness arrays must be identical: the
    early exclusion is a pure optimization that skips pixels the mask would
    zero out anyway.
    """
    rng = np.random.default_rng(57291)
    R_outer = 4.18
    R_inner = 2.5498
    pixel_scale = 10e-6
    aberrations = np.zeros(22)
    aberrations[4] = 25e-6  # large defocus
    aberrations[5:] = rng.uniform(-0.2e-6, 0.2e-6, size=17)

    # Default: pupil_R_inner = R_inner * 0.9
    factory_default = danish.DonutFactory(
        R_outer=R_outer, R_inner=R_inner,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=pixel_scale
    )
    # Explicit: pupil_R_inner = R_inner (tighter exclusion, different normalization)
    factory_with_inner = danish.DonutFactory(
        R_outer=R_outer, R_inner=R_inner,
        pupil_R_inner=R_inner,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=pixel_scale
    )

    img_default = factory_default.image(aberrations=aberrations)
    img_with_inner = factory_with_inner.image(aberrations=aberrations)

    # Undo the per-factory normalization to recover raw surface brightness,
    # then verify the two paths give the same result.
    denom_default = np.pi * (R_outer**2 - (R_inner*0.9)**2) / pixel_scale**2
    denom_with_inner = np.pi * (R_outer**2 - R_inner**2) / pixel_scale**2
    np.testing.assert_allclose(
        img_default * denom_default,
        img_with_inner * denom_with_inner,
        rtol=1e-12, atol=0
    )


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
        for _ in range(10 if run_slow else 1):
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
            for __ in range(10 if run_slow else 1):
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


@timer
def test_spots(run_slow):
    fiducial = batoid.Optic.fromYaml("Rubin_v3.14_r.yaml")
    fiducial = fiducial.withGloballyShiftedOptic("LSSTCamera", [0,0,30e-6])
    wavelength = 622e-9
    focal_length = 10.31
    eps = 0.621

    rng = np.random.default_rng(31415)
    for _ in range(30 if run_slow else 5):
        amplitude = 1000e-9  # ~1000 nm RMS perturbations
        jmax = 28
        coefs = rng.uniform(-1, 1, size=jmax+1)*amplitude/np.sqrt(jmax+1)
        coefs[:4] = 0.0  # No PTT
        perturbed = fiducial.withInsertedOptic(
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

        for __ in range(30 if run_slow else 5):
            thr = np.deg2rad(np.sqrt(rng.uniform(0, 1.8**2)))
            ph = rng.uniform(0, 2*np.pi)
            thx, thy = thr*np.cos(ph), thr*np.sin(ph)

            nrad = 40
            u, v = danish.hexapolar(outer=4.18, inner=4.18*eps, nrad=nrad)
            rays = batoid.RayVector.fromStop(
                u, v, optic=perturbed, wavelength=wavelength,
                theta_x=thx, theta_y=thy,
            )
            rays = perturbed.trace(rays)
            rx = rays.x
            ry = rays.y
            zTA = batoid.zernikeTA(
                perturbed,
                thx, thy,
                wavelength,
                nrad=20, naz=120, reference='mean',
                jmax=66, eps=eps,
                focal_length=focal_length,
            ) * wavelength
            zTA[:4] = 0.0

            factory = danish.DonutFactory(
                R_outer=4.18, R_inner=4.18*eps,
                mask_params=Rubin_obsc,
                focal_length=focal_length, pixel_scale=10e-6
            )
            sx, sy, sw = factory.spots(
                aberrations=zTA,
                thx=thx, thy=thy,
                nrad=nrad
            )

            # Align the means
            sx -= np.mean(sx[sw])
            sy -= np.mean(sy[sw])
            rx -= np.mean(rx[sw])
            ry -= np.mean(ry[sw])

            # No points worse than a pixel off
            np.testing.assert_allclose(sx[sw], rx[sw], atol=3e-7, rtol=0)
            np.testing.assert_allclose(sy[sw], ry[sw], atol=3e-7, rtol=0)
            # and most points much better than a pixel off
            assert np.nanquantile(np.hypot(sx[sw]-rx[sw], sy[sw]-ry[sw]), 0.9) < 5e-8

            # if np.nanquantile(np.hypot(sx[sw]-rx[sw], sy[sw]-ry[sw]), 0.9) >= 5e-8:
            #     import matplotlib.pyplot as plt
            #     fig, axs = plt.subplots(ncols=3, figsize=(12, 4), constrained_layout=True)
            #     axs[0].scatter(rx, ry, s=0.02, c='k')
            #     axs[0].scatter(rx[rays.vignetted], ry[rays.vignetted], s=0.02, c='r')
            #     axs[1].scatter(sx, sy, s=0.02, c='k')
            #     axs[1].scatter(sx[~sw], sy[~sw], s=0.02, c='r')
            #     Q = axs[2].quiver(
            #         sx, sy, rx-sx, ry-sy,
            #         angles='xy', scale_units='xy', scale=1, width=0.002, headwidth=3, headlength=4, headaxislength=3
            #     )
            #     axs[2].quiverkey(Q, 0.7, 0.9, 1e-6, "1 micron", labelpos='E')

            #     axs[0].set_title("Batoid ray hits")
            #     axs[1].set_title("Factory spot locations")
            #     axs[2].set_title("Spot errors")
            #     for ax in axs:
            #         ax.set_aspect('equal')
            #         ax.axhline(0, color='k', ls='--')
            #         ax.axvline(0, color='k', ls='--')
            #         ax.grid(True, which='major', c='gray', ls='-', alpha=0.2)
            #         ax.set_xticks([i*10e-6 for i in range(-10, 11)])
            #         ax.set_yticks([i*10e-6 for i in range(-10, 11)])
            #         ax.set_xticklabels([])
            #         ax.set_yticklabels([])
            #     plt.show()


@timer
def test_spot_image():
    fiducial = batoid.Optic.fromYaml("Rubin_v3.14_r.yaml")
    fiducial = fiducial.withGloballyShiftedOptic("LSSTCamera", [0,0,30e-6])
    wavelength = 622e-9
    focal_length = 10.31
    eps = 0.621

    rng = np.random.default_rng(31415)
    for _ in range(10):
        # amplitude = 100e-9  # ~100 nm RMS perturbations
        amplitude = 1500e-9  # ~1000 nm RMS perturbations
        jmax = 15
        coefs = rng.uniform(-1, 1, size=jmax+1)*amplitude/np.sqrt(jmax+1)
        coefs[:4] = 0.0  # No PTT
        perturbed = fiducial.withInsertedOptic(
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

        for __ in range(1):
            thr = np.deg2rad(np.sqrt(rng.uniform(0, 1.8**2)))
            ph = rng.uniform(0, 2*np.pi)
            thx, thy = thr*np.cos(ph), thr*np.sin(ph)
            nrad = 40
            zTA = batoid.zernikeTA(
                perturbed,
                thx, thy,
                wavelength,
                nrad=20, naz=120, reference='mean',
                jmax=66, eps=eps,
                focal_length=focal_length,
            ) * wavelength
            zTA[:4] = 0.0

            # Random atm ellipticity and size
            # size = rng.uniform(0.5, 1.5)/2.35
            size = 0.5/2.35
            e = np.sqrt(rng.uniform(0, 0.2**2))
            phi = rng.uniform(0, 2*np.pi)
            e1 = e*np.cos(phi)
            e2 = e*np.sin(phi)
            sigma = size*5*10e-6
            s = sigma**2
            den = np.sqrt(1-e1**2-e2**2)
            Ixx = s * (1 + e1) / den
            Iyy = s * (1 - e1) / den
            Ixy = s * e2 / den
            cov = np.array([[Ixx, Ixy], [Ixy, Iyy]])

            factory = danish.DonutFactory(
                R_outer=4.18, R_inner=4.18*eps,
                mask_params=Rubin_obsc,
                focal_length=focal_length, pixel_scale=10e-6
            )
            sx, sy, sw = factory.spots(
                aberrations=zTA,
                thx=thx, thy=thy,
                nrad=nrad
            )
            simg, sx1, sy1, sw1 = factory.spot_image(
                aberrations=zTA,
                thx=thx, thy=thy,
                nrad=nrad,
                gq_kwargs=dict(cov=cov, rmax=3.5)
            )
            simg2, sx2, sy2, sw2 = factory.spot_image(
                aberrations=zTA,
                thx=thx, thy=thy,
                nrad=nrad,
                gq_kwargs=dict(cov=cov, nrad=10, rmax=3.5)
            )

            import galsim
            from galsim.hsm import FindAdaptiveMom
            mom = FindAdaptiveMom(galsim.Image(simg))
            mom2 = FindAdaptiveMom(galsim.Image(simg2))

            s1 = (mom.moments_sigma)**2
            den_1 = np.sqrt(1 - mom.observed_shape.e1**2 - mom.observed_shape.e2**2)
            Ixx_1 = s1 * (1 + mom.observed_shape.e1) / den_1
            Iyy_1 = s1 * (1 - mom.observed_shape.e1) / den_1
            Ixy_1 = s1 * mom.observed_shape.e2 / den_1

            s2 = (mom2.moments_sigma)**2
            den_2 = np.sqrt(1 - mom2.observed_shape.e1**2 - mom2.observed_shape.e2**2)
            Ixx_2 = s2 * (1 + mom2.observed_shape.e1) / den_2
            Iyy_2 = s2 * (1 - mom2.observed_shape.e1) / den_2
            Ixy_2 = s2 * mom2.observed_shape.e2 / den_2

            # Need to account for the atmosphere and the pixel for the spots.
            Ixx_s0 = (np.var(sx[sw]) + cov[0,0]) / (10e-6)**2 + 1/12
            Iyy_s0 = (np.var(sy[sw]) + cov[1,1]) / (10e-6)**2 + 1/12
            Ixy_s0 = (np.mean(sx[sw]*sy[sw]) + cov[0,1]) / (10e-6)**2

            Ix_s1 = np.sum(sx1*sw1)/np.sum(sw1)
            Iy_s1 = np.sum(sy1*sw1)/np.sum(sw1)
            Ixx_s1 = np.sum((sx1-Ix_s1)**2*sw1)/np.sum(sw1) / (10e-6)**2 + 1/12
            Iyy_s1 = np.sum((sy1-Iy_s1)**2*sw1)/np.sum(sw1) / (10e-6)**2 + 1/12
            Ixy_s1 = np.sum((sx1-Ix_s1)*(sy1-Iy_s1)*sw1)/np.sum(sw1) / (10e-6)**2

            Ix_s2 = np.sum(sx2*sw2)/np.sum(sw2)
            Iy_s2 = np.sum(sy2*sw2)/np.sum(sw2)
            Ixx_s2 = np.sum((sx2-Ix_s2)**2*sw2)/np.sum(sw2) / (10e-6)**2 + 1/12
            Iyy_s2 = np.sum((sy2-Iy_s2)**2*sw2)/np.sum(sw2) / (10e-6)**2 + 1/12
            Ixy_s2 = np.sum((sx2-Ix_s2)*(sy2-Iy_s2)*sw2)/np.sum(sw2) / (10e-6)**2

            # TODO: Add an actual test here.

            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(ncols=5, figsize=(20, 4), constrained_layout=True)
            # axs[0].scatter(sx, sy, s=0.02, c='k')
            # axs[0].scatter(sx[~sw], sy[~sw], s=0.02, c='r')
            # axs[1].scatter(sx1.ravel(), sy1.ravel(), s=0.02, alpha=sw1.ravel())
            # axs[1].scatter(sx1[0], sy1[0], s=0.2, c='r')
            # axs[2].scatter(sx2.ravel(), sy2.ravel(), s=0.02, alpha=sw2.ravel()*3)
            # axs[2].scatter(sx2[0], sy2[0], s=0.2, c='r')

            # npix = 45
            # extent = [-npix*10e-6, npix*10e-6, -npix*10e-6, npix*10e-6]
            # im = axs[3].imshow(
            #     simg, origin='lower', vmin=0, vmax=np.nanmax(simg),
            #     extent=extent,
            # )
            # fig.colorbar(im, ax=axs[3], label='Spot intensity (arbitrary units)')
            # im2 = axs[4].imshow(
            #     simg2, origin='lower', vmin=0, vmax=np.nanmax(simg2),
            #     extent=extent,
            # )
            # fig.colorbar(im2, ax=axs[4], label='Spot intensity (arbitrary units)')
            # axs[0].set_title("Factory spot locations")
            # axs[1].set_title("Convolved spot locations")
            # axs[2].set_title("Convolved spot locations (high density)")
            # axs[3].set_title("Factory spot image")
            # axs[4].set_title("Factory spot image (high density)")
            # axs[0].text(
            #     0.03, 0.95, f"Ixx={Ixx_s0:.2f} Iyy={Iyy_s0:.2f} Ixy={Ixy_s0:.2f}",
            #     transform=axs[0].transAxes, fontsize=8, verticalalignment='top'
            # )
            # axs[1].text(
            #     0.03, 0.95, f"Ixx={Ixx_s1:.2f} Iyy={Iyy_s1:.2f} Ixy={Ixy_s1:.2f}",
            #     transform=axs[1].transAxes, fontsize=8, verticalalignment='top'
            # )
            # axs[2].text(
            #     0.03, 0.95, f"Ixx={Ixx_s2:.2f} Iyy={Iyy_s2:.2f} Ixy={Ixy_s2:.2f}",
            #     transform=axs[2].transAxes, fontsize=8, verticalalignment='top'
            # )
            # axs[3].text(
            #     0.03, 0.95, f"Ixx={Ixx_1:.2f} Iyy={Iyy_1:.2f} Ixy={Ixy_1:.2f}",
            #     transform=axs[3].transAxes, fontsize=8, verticalalignment='top',
            #     c="w"

            # )
            # axs[4].text(
            #     0.03, 0.95, f"Ixx={Ixx_2:.2f} Iyy={Iyy_2:.2f} Ixy={Ixy_2:.2f}",
            #     transform=axs[4].transAxes, fontsize=8, verticalalignment='top',
            #     c="w"
            # )

            # for ax in axs:
            #     lim = npix*10e-6/2
            #     ax.set_xlim(-lim, lim)
            #     ax.set_ylim(-lim, lim)
            #     ax.set_aspect('equal')
            #     ax.axhline(0, color='k', ls='--')
            #     ax.axvline(0, color='k', ls='--')
            #     ax.grid(True, which='major', c='gray', ls='-', alpha=0.2)
            #     ax.set_xticks([5e-6+i*10e-6 for i in range(-npix, npix)])
            #     ax.set_yticks([5e-6+i*10e-6 for i in range(-npix, npix)])
            #     ax.set_xticklabels([])
            #     ax.set_yticklabels([])
            # plt.show()



@timer
def test_triangle_factory_annulus_mesh_area():
    factory = danish.DonutTriangleFactory(
        R_outer=4.18,
        R_inner=2.5498,
    )
    mesh = factory.build_annulus_mesh(
        nrad=16,
        naz=84,
        boundary_naz=360,
        debug=False,
    )

    assert mesh['vertices'].shape[1] == 2
    assert mesh['triangles'].shape[1] == 3
    assert mesh['triangles'].shape[0] > 0

    analytic = np.pi * (factory.pupil_R_outer**2 - factory.pupil_R_inner**2)
    rel_err = abs(mesh['triangle_area_sum'] - analytic) / analytic
    assert rel_err < 2e-2


@timer
def test_triangle_factory_debug_plot_smoke():
    factory = danish.DonutTriangleFactory(
        R_outer=4.18,
        R_inner=2.5498,
    )
    mesh = factory.build_annulus_mesh(
        nrad=12,
        naz=72,
        boundary_naz=240,
        debug=True,
        show_debug=False,
    )

    assert 'debug_figure' in mesh
    assert 'debug_axes' in mesh
    assert mesh['debug_axes'] is not None


@timer
def test_triangle_factory_circle_obscurations_smoke():
    factory = danish.DonutTriangleFactory(
        R_outer=4.18,
        R_inner=2.5498,
    )
    mesh = factory.build_annulus_mesh(
        nrad=12,
        naz=72,
        boundary_naz=240,
        debug=False,
    )

    masked = factory.apply_circle_obscurations(
        mesh,
        mask_params=Rubin_obsc,
        thx=np.deg2rad(1.6),
        thy=0.0,
        debug=True,
        show_debug=False,
        plot_vertices=True,
    )
    assert masked['triangle_area_sum'] <= mesh['triangle_area_sum'] + 1e-10
    assert masked['triangle_area_sum'] < mesh['triangle_area_sum']
    assert masked['clipped_triangle_count'] >= 0
    assert len(masked['active_circles']) > 0
    assert 'debug_figure' in masked
    assert 'debug_axes' in masked
    assert masked['debug_axes'] is not None

if __name__ == "__main__":
    runtests(__file__)
