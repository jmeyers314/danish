import os
import pickle
import time
import yaml

from scipy.optimize import least_squares
import numpy as np
import batoid
from galsim.zernike import Zernike

import danish
from danish_test_helpers import timer, runtests

directory = os.path.dirname(__file__)

Rubin_obsc = yaml.safe_load(open(os.path.join(danish.datadir, 'RubinObsc.yaml')))
AuxTel_obsc = yaml.safe_load(open(os.path.join(danish.datadir, 'AuxTelObsc.yaml')))


def plot_result(img, mod, z_fit, z_true, ylim=None, wavelength=None):
    jmax = len(z_fit)+4
    import matplotlib.pyplot as plt
    fig = plt.figure(constrained_layout=True, figsize=(10, 7))
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    ax0.imshow(img)
    ax1.imshow(mod)
    ax2.imshow(img - mod)
    ax3.axhline(0, c='k')
    if wavelength is not None:
        z_fit *= wavelength*1e9
        z_true *= wavelength*1e9
    ax3.plot(np.arange(4, jmax), z_fit, c='b', label='fit')
    ax3.plot(np.arange(4, jmax), z_true, c='k', label='truth')
    ax3.plot(
        np.arange(4, jmax),
        (z_fit-z_true),
        c='r', label='fit - truth', ls='--'
    )
    if ylim is None:
        ylim = -0.6, 0.6
    ax3.set_ylim(*ylim)
    ax3.set_xlabel("Zernike index")
    if wavelength is not None:
        ax3.set_ylabel("Residual (nm)")
    else:
        ax3.set_ylabel("Residual (Waves)")
    ax3.set_xticks(np.arange(4, jmax, dtype=int))
    ax3.legend()
    plt.show()


def plot_dz_results(imgs, mods, dz_fit, dz_true, dz_terms):
    import matplotlib.pyplot as plt
    fig = plt.figure(constrained_layout=True, figsize=(8, 10))
    gs = fig.add_gridspec(len(imgs)+3, 3)
    for i, (img, mod) in enumerate(zip(imgs, mods)):
        ax0 = fig.add_subplot(gs[i, 0])
        ax1 = fig.add_subplot(gs[i, 1])
        ax2 = fig.add_subplot(gs[i, 2])
        ax0.imshow(img)
        ax1.imshow(mod)
        ax2.imshow(img - mod)
    ax3 = fig.add_subplot(gs[-3:, :])
    ax3.axhline(0, c='k', alpha=0.1)
    ax3.plot(dz_fit, c='b', label='fit')
    ax3.plot(dz_true, c='k', label='truth')
    ax3.plot(
        (dz_fit-dz_true),
        c='r', label='fit - truth'
    )
    ax3.set_ylim(-0.6, 0.6)
    ax3.set_xlabel("Double Zernike index")
    ax3.set_ylabel("Residual (Waves)")
    ax3.set_xticks(range(len(dz_terms)))
    ax3.set_xticklabels(dz_terms)
    ax3.legend()
    plt.show()


@timer
def test_fitter_LSST_fiducial(run_slow):
    """Roundtrip using danish model to produce a test image with fiducial LSST
    transverse Zernikes plus random Zernike offsets.  Model and fitter run
    through the same code.
    """
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0, 0, 0.0015])

    wavelength = 750e-9

    rng = np.random.default_rng(234)
    if run_slow:
        niter = 10
    else:
        niter = 2
    for _ in range(niter):
        # Generate params
        thr = np.sqrt(rng.uniform(0, 1.8**2))
        ph = rng.uniform(0, 2*np.pi)
        thx, thy = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
        z_ref = batoid.zernikeTA(
            telescope, thx, thy, wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )

        z_ref *= wavelength

        z_terms = np.arange(4, 23)
        z_true = rng.uniform(-0.1, 0.1, size=19)*wavelength

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            mask_params=Rubin_obsc,
            focal_length=10.31, pixel_scale=10e-6
        )

        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref, z_terms=z_terms, thx=thx, thy=thy, bkg_order=0
        )

        dx, dy = rng.uniform(-0.5, 0.5, size=2)
        fwhm = rng.uniform(0.4, 3.0)
        sky_level = rng.uniform(1000.0, 2000.0)
        flux = rng.uniform(5e6, 1e7)

        # Generate test image
        img = fitter.model(
            flux, dx, dy, fwhm, z_true,
            sky_level=sky_level,
        )

        guess = [np.sum(img), 0.0, 0.0, 0.7]+[0.0]*19+[0.0]
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        result = fitter.unpack_params(result.x)
        z_fit = np.array(result["z_fit"])
        dx_fit = result["dx"]
        dy_fit = result["dy"]
        fwhm_fit = result["fwhm"]
        flux_fit = result["flux"]
        for i in range(4, 23):
            out = f"{i:2d}  {z_fit[i-4]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(z_fit[i-4]-z_true[i-4])/wavelength:9.3f}"
            print(out)
        print(f"rms: {np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))):.3f}")
        print(f"dx: {dx_fit:.3f}  {dx:.3f} {dx_fit-dx:.3f}")
        print(f"dy: {dy_fit:.3f}  {dy:.3f} {dy_fit-dy:.3f}")
        print(f"fwhm: {fwhm_fit:.3f}  {fwhm:.3f} {fwhm_fit-fwhm:.3f}")
        print(f"flux: {flux_fit:.3f}  {flux:.3f} {flux_fit-flux:.3f}")

        # if (
        #     np.abs(fwhm_fit-fwhm)/fwhm > 0.05 or
        #     np.any(np.abs(z_fit-z_true) > 0.1*wavelength) or
        #     np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))) > 0.1
        # ):
        #     mod = fitter.model(**result)
        #     plot_result(img, mod, z_fit/wavelength, z_true/wavelength)

        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=5e-2, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.05*wavelength)
        rms = np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength)))
        assert rms < 0.1, "rms %9.3f > 0.1" % rms

        # Try with x_offset and y_offset too
        ####################################

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

        fitter = danish.SingleDonutModel(
            factory,
            x_offset=zx, y_offset=zy,
            z_ref=z_ref*0,
            z_terms=z_terms,
            thx=thx, thy=thy, bkg_order=0
        )

        # New test image
        img = fitter.model(
            flux, dx, dy, fwhm, z_true,
            sky_level=sky_level
        )

        guess = [np.sum(img), 0.0, 0.0, 0.7]+[0.0]*19+[0.0]
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        result = fitter.unpack_params(result.x)
        z_fit = np.array(result["z_fit"])
        dx_fit = result["dx"]
        dy_fit = result["dy"]
        fwhm_fit = result["fwhm"]
        flux_fit = result["flux"]
        for i in range(4, 23):
            out = f"{i:2d}  {z_fit[i-4]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(z_fit[i-4]-z_true[i-4])/wavelength:9.3f}"
            print(out)
        print(f"rms: {np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))):.3f}")
        print(f"dx: {dx_fit:.3f}  {dx:.3f} {dx_fit-dx:.3f}")
        print(f"dy: {dy_fit:.3f}  {dy:.3f} {dy_fit-dy:.3f}")
        print(f"fwhm: {fwhm_fit:.3f}  {fwhm:.3f} {fwhm_fit-fwhm:.3f}")
        print(f"flux: {flux_fit:.3f}  {flux:.3f} {flux_fit-flux:.3f}")

        # if (
        #     np.abs(fwhm_fit-fwhm)/fwhm > 0.05 or
        #     np.any(np.abs(z_fit-z_true) > 0.1*wavelength) or
        #     np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))) > 0.1
        # ):
        #     mod = fitter.model(**result)
        #     plot_result(img, mod, z_fit/wavelength, z_true/wavelength)

        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=5e-2, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.05*wavelength)
        rms = np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength)))
        assert rms < 0.1, "rms %9.3f > 0.1" % rms


@timer
def test_fitter_LSST_rigid_perturbation(run_slow):
    """Roundtrip using danish model to produce a test image of rigid-body
    perturbed LSST transverse Zernikes.  Model and fitter run through the same
    code.
    """
    fiducial_telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    fiducial_telescope = fiducial_telescope.withGloballyShiftedOptic(
        "Detector",
        [0, 0, 0.0015]
    )

    wavelength = 750e-9

    rng = np.random.default_rng(234)
    if run_slow:
        niter = 10
    else:
        niter = 2
    for _ in range(niter):
        M2_dx, M2_dy = rng.uniform(-2e-4, 2e-4, size=2)
        M2_dz = rng.uniform(-2e-5, 2e-5)
        M2_thx, M2_thy = rng.uniform(-2e-5, 2e-5, size=2)
        cam_dx, cam_dy = rng.uniform(-2e-3, 2e-3, size=2)
        cam_dz = rng.uniform(-2e-5, 2e-5)
        cam_thx, cam_thy = rng.uniform(-2e-4, 2e-4, size=2)
        telescope = (
            fiducial_telescope
            .withGloballyShiftedOptic("M2", [M2_dx, M2_dy, M2_dz])
            .withLocallyRotatedOptic(
                "M2", batoid.RotX(M2_thx)@batoid.RotY(M2_thy)
            )
            .withGloballyShiftedOptic("LSSTCamera", [cam_dx, cam_dy, cam_dz])
            .withLocallyRotatedOptic(
                "LSSTCamera", batoid.RotX(cam_thx)@batoid.RotY(cam_thy)
            )
        )
        thr = np.sqrt(rng.uniform(0, 1.8**2))
        ph = rng.uniform(0, 2*np.pi)
        thx, thy = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
        z_ref = batoid.zernikeTA(
            fiducial_telescope, thx, thy, wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )
        z_ref *= wavelength

        z_perturb = batoid.zernikeTA(
            telescope, thx, thy, wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )
        z_perturb *= wavelength

        z_terms = np.arange(4, 23)
        z_true = (z_perturb - z_ref)[4:23]

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            mask_params=Rubin_obsc,
            focal_length=10.31, pixel_scale=10e-6
        )

        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref, z_terms=z_terms, thx=thx, thy=thy, bkg_order=0
        )

        dx, dy = 0.0, 0.0
        fwhm = 0.7
        sky_level = 1000.0
        flux = 5e6

        img = fitter.model(
            flux, dx, dy, fwhm, z_true,
            sky_level=sky_level
        )

        guess = [np.sum(img), 0.0, 0.0, 0.7]+[0.0]*19+[0.0]
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        result = fitter.unpack_params(result.x)
        z_fit = np.array(result["z_fit"])
        dx_fit = result["dx"]
        dy_fit = result["dy"]
        fwhm_fit = result["fwhm"]
        flux_fit = result["flux"]
        for i in range(4, 23):
            out = f"{i:2d}  {z_fit[i-4]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(z_fit[i-4]-z_true[i-4])/wavelength:9.3f}"
            print(out)
        print(f"rms: {np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))):.3f}")
        print(f"dx: {dx_fit:.3f}  {dx:.3f} {dx_fit-dx:.3f}")
        print(f"dy: {dy_fit:.3f}  {dy:.3f} {dy_fit-dy:.3f}")
        print(f"fwhm: {fwhm_fit:.3f}  {fwhm:.3f} {fwhm_fit-fwhm:.3f}")
        print(f"flux: {flux_fit:.3f}  {flux:.3f} {flux_fit-flux:.3f}")

        # if (
        #     np.abs(fwhm_fit-fwhm)/fwhm > 0.05 or
        #     np.any(np.abs(z_fit-z_true) > 0.1*wavelength) or
        #     np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))) > 0.1
        # ):
        #     mod = fitter.model(**result)
        #     plot_result(img, mod, z_fit/wavelength, z_true/wavelength)

        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=5e-2, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.05*wavelength)
        rms = np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength)))
        assert rms < 0.1, "rms %9.3f > 0.1" % rms


@timer
def test_fitter_LSST_z_perturbation(run_slow):
    """Roundtrip using danish model to produce a test image of rigid-body +
    M1-surface-Zernike perturbed LSST transverse Zernikes.  Model and fitter run
    through the same code.
    """
    fiducial_telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    fiducial_telescope = fiducial_telescope.withGloballyShiftedOptic(
        "Detector",
        [0, 0, 0.0015]
    )

    wavelength = 750e-9

    rng = np.random.default_rng(234)
    if run_slow:
        niter = 10
    else:
        niter = 2
    for _ in range(niter):
        M2_dx, M2_dy = rng.uniform(-2e-4, 2e-4, size=2)
        M2_dz = rng.uniform(-2e-5, 2e-5)
        M2_thx, M2_thy = rng.uniform(-2e-5, 2e-5, size=2)
        cam_dx, cam_dy = rng.uniform(-2e-3, 2e-3, size=2)
        cam_dz = rng.uniform(-2e-5, 2e-5)
        cam_thx, cam_thy = rng.uniform(-2e-4, 2e-4, size=2)
        telescope = (
            fiducial_telescope
            .withGloballyShiftedOptic("M2", [M2_dx, M2_dy, M2_dz])
            .withLocallyRotatedOptic(
                "M2", batoid.RotX(M2_thx)@batoid.RotY(M2_thy)
            )
            .withGloballyShiftedOptic("LSSTCamera", [cam_dx, cam_dy, cam_dz])
            .withLocallyRotatedOptic(
                "LSSTCamera", batoid.RotX(cam_thx)@batoid.RotY(cam_thy)
            )
        )
        M1 = telescope['M1']
        M1_a = np.zeros(23)
        M1_a[12:23] = rng.uniform(-20e-9, 20e-9, 11)
        telescope = telescope.withSurface(
            "M1",
            batoid.Sum([
                M1.surface,
                batoid.Zernike(
                    M1_a,
                    R_outer=M1.obscuration.original.outer,
                    R_inner=M1.obscuration.original.inner
                )
            ])
        )

        thr = np.sqrt(rng.uniform(0, 1.8**2))
        ph = rng.uniform(0, 2*np.pi)
        thx, thy = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
        z_ref = batoid.zernikeTA(
            fiducial_telescope, thx, thy, wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )
        z_ref *= wavelength

        z_perturb = batoid.zernikeTA(
            telescope, thx, thy, wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )
        z_perturb *= wavelength

        z_terms = np.arange(4, 23)
        z_true = (z_perturb - z_ref)[4:23]

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            mask_params=Rubin_obsc,
            focal_length=10.31, pixel_scale=10e-6
        )

        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref, z_terms=z_terms, thx=thx, thy=thy, bkg_order=0
        )

        dx, dy = 0.0, 0.0
        fwhm = 0.7
        sky_level = 1000.0
        flux = 5e6

        img = fitter.model(
            flux, dx, dy, fwhm, z_true,
            sky_level=sky_level
        )

        guess = [np.sum(img), 0.0, 0.0, 0.7]+[0.0]*19+[0.0]
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        result = fitter.unpack_params(result.x)
        z_fit = np.array(result["z_fit"])
        dx_fit = result["dx"]
        dy_fit = result["dy"]
        fwhm_fit = result["fwhm"]
        flux_fit = result["flux"]
        for i in range(4, 23):
            out = f"{i:2d}  {z_fit[i-4]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(z_fit[i-4]-z_true[i-4])/wavelength:9.3f}"
            print(out)
        print(f"rms: {np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))):.3f}")
        print(f"dx: {dx_fit:.3f}  {dx:.3f} {dx_fit-dx:.3f}")
        print(f"dy: {dy_fit:.3f}  {dy:.3f} {dy_fit-dy:.3f}")
        print(f"fwhm: {fwhm_fit:.3f}  {fwhm:.3f} {fwhm_fit-fwhm:.3f}")
        print(f"flux: {flux_fit:.3f}  {flux:.3f} {flux_fit-flux:.3f}")

        # if (
        #     np.abs(fwhm_fit-fwhm)/fwhm > 0.05 or
        #     np.any(np.abs(z_fit-z_true) > 0.1*wavelength) or
        #     np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))) > 0.1
        # ):
        #     mod = fitter.model(**result)
        #     plot_result(img, mod, z_fit/wavelength, z_true/wavelength)

        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=5e-2, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.05*wavelength)
        rms = np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength)))
        assert rms < 0.1, "rms %9.3f > 0.1" % rms


@timer
def test_fitter_LSST_kolm(run_slow):
    """Roundtrip using GalSim Kolmogorov atmosphere + batoid to produce test
    image of AOS DOF perturbed optics.  Model and fitter run independent code.
    """
    with open(
        os.path.join(directory, "data", "test_kolm_donuts.pkl"),
        'rb'
    ) as f:
        data = pickle.load(f)
    sky_level = data[0]['sky_level']
    wavelength = data[0]['wavelength']
    fwhm = data[0]['fwhm']

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6
    )
    binned_factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=20e-6
    )

    if run_slow:
        niter = 10
    else:
        niter = 2
    for datum in data[1:niter]:
        thx = datum['thx']
        thy = datum['thy']
        z_ref = datum['z_ref']
        z_actual = datum['z_actual']
        img = datum['arr']

        z_terms = np.arange(4, 23)
        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref*wavelength, z_terms=z_terms, thx=thx, thy=thy, bkg_order=0
        )
        guess = [np.sum(img), 0.0, 0.0, 0.7]+[0.0]*19+[0.0]

        t0 = time.time()
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        result = fitter.unpack_params(result.x)
        t1 = time.time()
        t1x1 = t1 - t0

        z_true = (z_actual-z_ref)[4:23]*wavelength
        z_fit = np.array(result["z_fit"])
        fwhm_fit = result["fwhm"]
        for i in range(4, 23):
            out = f"{i:2d}  {z_fit[i-4]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(z_fit[i-4]-z_true[i-4])/wavelength:9.3f}"
            print(out)
        print(f"rms: {np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))):.3f}")
        print(f"fwhm: {fwhm_fit:.3f}  {fwhm:.3f} {fwhm_fit-fwhm:.3f}")

        # if (
        #     np.abs(fwhm_fit-fwhm)/fwhm > 0.05 or
        #     np.any(np.abs(z_fit-z_true) > 0.15*wavelength) or
        #     np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength))) > 0.2
        # ):
        #     mod = fitter.model(**result)
        #     plot_result(img, mod, z_fit/wavelength, z_true/wavelength)

        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.15*wavelength)
        rms1x1 = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        assert rms1x1 < 0.2, "rms %9.3f > %9.3" % (rms1x1, 0.2)

        # Try binning 2x2
        binned_fitter = danish.SingleDonutModel(
            binned_factory, z_ref=z_ref*wavelength, z_terms=z_terms,
            thx=thx, thy=thy, npix=89, bkg_order=0
        )

        binned_img = img[:-1,:-1].reshape(90,2,90,2).mean(-1).mean(1)[:-1,:-1]
        t0 = time.time()
        lb = [-np.inf]*len(guess)
        lb[0] = 0
        lb[3] = 0.1
        ub = [np.inf]*len(guess)
        binned_result = least_squares(
            binned_fitter.chi, guess, jac=binned_fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            bounds=(lb, ub),
            args=(binned_img, 4*sky_level)
        )
        binned_result = binned_fitter.unpack_params(binned_result.x)
        t1 = time.time()
        t2x2 = t1 - t0

        z_fit = np.array(result["z_fit"])
        fwhm_fit = result["fwhm"]
        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.15*wavelength)
        rms2x2 = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        assert rms2x2 < 0.2, "rms %9.3f > %9.3" % (rms2x2, 0.2)

        print("\n"*4)
        print(f"1x1 fit time: {t1x1:.3f} sec")
        print(f"2x2 fit time: {t2x2:.3f} sec")
        print(f"1x1 rms: {rms1x1}")
        print(f"2x2 rms: {rms2x2}")
        print("\n"*4)



@timer
def test_fitter_LSST_atm(run_slow):
    """Roundtrip using GalSim phase screen atmosphere + batoid to produce test
    image of AOS DOF perturbed optics.  Model and fitter run independent code.
    """
    with open(
        os.path.join(directory, "data", "test_atm_donuts.pkl"),
        'rb'
    ) as f:
        data = pickle.load(f)
    wavelength = data[0]['wavelength']
    fwhm = data[0]['fwhm']

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6
    )
    binned_factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=20e-6
    )

    sky_level = data[0]['sky_level']
    if run_slow:
        niter = 10
    else:
        niter = 2
    for datum in data[1:niter]:
        thx = datum['thx']
        thy = datum['thy']
        z_ref = datum['z_ref']
        z_actual = datum['z_actual']
        img = datum['arr']

        z_terms = np.arange(4, 23)
        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref*wavelength, z_terms=z_terms, thx=thx, thy=thy, bkg_order=0
        )
        guess = [np.sum(img), 0.0, 0.0, 0.7] + [0.0]*19 + [0.0]

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        result = fitter.unpack_params(result.x)

        z_true = (z_actual-z_ref)[4:23]*wavelength
        z_fit = np.array(result["z_fit"])
        fwhm_fit = result["fwhm"]
        for i in range(4, 23):
            out = f"{i:2d}  {z_fit[i-4]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(z_fit[i-4]-z_true[i-4])/wavelength:9.3f}"
            print(out)
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        print(f"rms: {rms:.3f}")
        print(f"fwhm: {fwhm_fit:.3f}  {fwhm:.3f} {fwhm_fit-fwhm:.3f}")

        # if (
        #     np.abs(fwhm_fit-fwhm)/fwhm > 0.05 or
        #     np.any(np.abs(z_fit-z_true) > 0.5*wavelength) or
        #     rms > 0.66
        # ):
        #     mod = fitter.model(**result)
        #     plot_result(img, mod, z_fit/wavelength, z_true/wavelength)

        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.5*wavelength)
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.66, "rms %9.3f > 0.66" % rms

        # Try binning 2x2
        binned_fitter = danish.SingleDonutModel(
            binned_factory, z_ref=z_ref*wavelength, z_terms=z_terms,
            thx=thx, thy=thy, npix=89, bkg_order=0
        )

        binned_img = img[:-1,:-1].reshape(90,2,90,2).mean(-1).mean(1)[:-1,:-1]
        t0 = time.time()
        lb = [-np.inf]*len(guess)
        lb[0] = 0
        lb[3] = 0.1
        ub = [np.inf]*len(guess)
        binned_result = least_squares(
            binned_fitter.chi, guess, jac=binned_fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            bounds=(lb, ub),
            args=(binned_img, 4*sky_level)
        )
        binned_result = binned_fitter.unpack_params(binned_result.x)
        t1 = time.time()
        print(f"2x2 fit time: {t1-t0:.3f} sec")

        z_fit = np.array(z_fit)

        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.5*wavelength)
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.66, "rms %9.3f > 0.66" % rms


@timer
def test_fitter_AuxTel_rigid_perturbation(run_slow):
    """Roundtrip using danish model to produce a test image of rigid-body
    perturbed AuxTel transverse Zernikes.  Model and fitter run through the same
    code.
    """
    # Nominal donut mode for AuxTel is to despace M2 by 0.8 mm
    fiducial_telescope = batoid.Optic.fromYaml("AuxTel.yaml")
    fiducial_telescope = fiducial_telescope.withLocallyShiftedOptic(
        "M2",
        [0, 0, 0.0008]
    )

    wavelength = 750e-9

    rng = np.random.default_rng(234)
    if run_slow:
        niter = 10
    else:
        niter = 2
    for _ in range(niter):
        # Randomly perturb M2 alignment
        M2_dx, M2_dy = rng.uniform(-3e-4, 3e-4, size=2)
        M2_dz = rng.uniform(-3e-5, 3e-5)
        M2_thx, M2_thy = rng.uniform(-3e-5, 3e-5, size=2)
        telescope = (
            fiducial_telescope
            .withGloballyShiftedOptic("M2", [M2_dx, M2_dy, M2_dz])
            .withLocallyRotatedOptic(
                "M2", batoid.RotX(M2_thx)@batoid.RotY(M2_thy)
            )
        )
        # Random point inside 0.05 degree radius field-of-view
        thr = np.sqrt(rng.uniform(0, 0.05**2))
        ph = rng.uniform(0, 2*np.pi)
        thx, thy = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
        # Determine reference "design" zernikes.  Use the transverse aberration
        # zernikes since danish uses a transverse aberration ray-hit model.
        z_ref = batoid.zernikeTA(
            fiducial_telescope, thx, thy, wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=11, eps=0.2115/0.6
        )
        z_ref *= wavelength

        # The zernikes of the perturbed telescope.  I.e., the "truth".
        z_perturb = batoid.zernikeTA(
            telescope, thx, thy, wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=11, eps=0.2115/0.6
        )
        z_perturb *= wavelength

        z_terms = np.arange(4, 12)
        z_true = (z_perturb - z_ref)[4:12]

        # NOTE: The R_inner and focal_length here don't quite match what I've
        # seen elsewhere.  Possible location for future improvement.
        factory = danish.DonutFactory(
            R_outer=0.6, R_inner=0.2115,
            mask_params=AuxTel_obsc,
            focal_length=20.8, pixel_scale=10e-6
        )

        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref, z_terms=z_terms, thx=thx, thy=thy, npix=255, bkg_order=0
        )

        dx, dy = 0.0, 0.0
        fwhm = 0.7  # Arcsec for Kolmogorov profile
        sky_level = 1000.0  # counts per pixel
        flux = 5e6

        # Make a test image using true aberrations
        img = fitter.model(
            flux, dx, dy, fwhm, z_true,
            sky_level=sky_level
        )

        # Now guess aberrations are 0.0, and try to recover truth.
        guess = [np.sum(img), 0.0, 0.0, 0.7]+[0.0]*8+[0.0]
        # We don't ship a custom fitting algorithm; just use scipy.least_squares
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        result = fitter.unpack_params(result.x)

        z_fit = np.array(result["z_fit"])
        fwhm_fit = result["fwhm"]
        for i in range(4, 12):
            out = f"{i:2d}  {z_fit[i-4]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(z_fit[i-4]-z_true[i-4])/wavelength:9.3f}"
            print(out)
        rms = np.sqrt(np.sum(np.square((z_true-z_fit)/wavelength)))
        print(f"rms: {rms:.3f}")
        print(f"fwhm: {fwhm_fit:.3f}  {fwhm:.3f} {fwhm_fit-fwhm:.3f}")

        # # Optional visualization
        # mod = fitter.model(**result)
        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength, ylim=(-0.2, 0.2))

        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.006*wavelength)
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        assert rms < 0.1, "rms %9.3f > 0.1" % rms


@timer
def test_dz_fitter_LSST_fiducial(run_slow):
    """Roundtrip using danish model to produce test images with fiducial LSST
    transverse Zernikes plus random double Zernike offsets.  Model and fitter
    run through the same code.
    """
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0, 0, 0.0015])

    wavelength = 750e-9

    rng = np.random.default_rng(2344)
    nstar = 10
    if run_slow:
        niter = 10
    else:
        niter = 1
    for _ in range(niter):
        thr = np.sqrt(rng.uniform(0, 1.8**2, nstar))
        ph = rng.uniform(0, 2*np.pi, nstar)
        thxs, thys = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
        z_refs = np.empty((nstar, 67))
        for i, (thx, thy) in enumerate(zip(thxs, thys)):
            z_refs[i] = batoid.zernikeTA(
                telescope, thx, thy, wavelength,
                nrad=20, naz=120, reference='chief',
                jmax=66, eps=0.61
            )
        z_refs *= wavelength
        dz_terms = (
            (1, 4),                          # defocus
            (2, 4), (3, 4),                  # field tilt
            (2, 5), (3, 5), (2, 6), (3, 6),  # linear astigmatism
            (1, 7), (1, 8)                   # constant coma
        )
        dz_true = rng.uniform(-0.3, 0.3, size=len(dz_terms))*wavelength

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            mask_params=Rubin_obsc,
            focal_length=10.31, pixel_scale=10e-6
        )

        fitter = danish.DZMultiDonutModel(
            factory,
            z_refs=z_refs, dz_terms=dz_terms,
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys, bkg_order=0,
        )

        dxs = rng.uniform(-0.5, 0.5, nstar)
        dys = rng.uniform(-0.5, 0.5, nstar)
        fwhm = rng.uniform(0.5, 1.5)
        sky_levels = [1000.0]*nstar
        fluxes = rng.uniform(3e6, 1e7, nstar)

        imgs = fitter.model(
            fluxes, dxs, dys, fwhm, dz_true, sky_levels=sky_levels,
        )

        guess = [np.sum(img) for img in imgs]
        guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
        guess += [0.0]*(fitter.nbkg*nstar)

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(imgs, sky_levels)
        )
        result = fitter.unpack_params(result.x)

        fwhm_fit = result["fwhm"]
        dz_fit = result["wavefront_params"]
        np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.02)
        np.testing.assert_allclose(
            dz_fit/wavelength,
            dz_true/wavelength,
            rtol=0, atol=0.02
        )
        rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.05, "rms %9.3f > 0.05" % rms

        # mods = fitter.model(**result)
        # plot_dz_results(
        #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
        # )


@timer
def test_dz_fitter_LSST_rigid_perturbation(run_slow):
    """Roundtrip using danish model to produce a test images of rigid-body
    perturbed LSST transverse Zernikes.  Model and fitter run through the same
    code.
    """
    fiducial_telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    fiducial_telescope = fiducial_telescope.withGloballyShiftedOptic(
        "Detector",
        [0, 0, 0.0015]
    )

    wavelength = 750e-9

    rng = np.random.default_rng(1234)
    if run_slow:
        niter = 10
    else:
        niter = 1
    for _ in range(niter):
        M2_dx, M2_dy = rng.uniform(-2e-4, 2e-4, size=2)
        M2_dz = rng.uniform(-2e-5, 2e-5)
        M2_thx, M2_thy = rng.uniform(-2e-5, 2e-5, size=2)
        cam_dx, cam_dy = rng.uniform(-2e-3, 2e-3, size=2)
        cam_dz = rng.uniform(-2e-5, 2e-5)
        cam_thx, cam_thy = rng.uniform(-2e-4, 2e-4, size=2)
        telescope = (
            fiducial_telescope
            .withGloballyShiftedOptic("M2", [M2_dx, M2_dy, M2_dz])
            .withLocallyRotatedOptic(
                "M2", batoid.RotX(M2_thx)@batoid.RotY(M2_thy)
            )
            .withGloballyShiftedOptic("LSSTCamera", [cam_dx, cam_dy, cam_dz])
            .withLocallyRotatedOptic(
                "LSSTCamera", batoid.RotX(cam_thx)@batoid.RotY(cam_thy)
            )
        )
        nstar = 10
        thr = np.sqrt(rng.uniform(0, 1.8**2, nstar))
        ph = rng.uniform(0, 2*np.pi, nstar)
        thxs, thys = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
        z_refs = np.empty((nstar, 67))
        z_perturbs = np.empty((nstar, 67))
        for i, (thx, thy) in enumerate(zip(thxs, thys)):
            z_refs[i] = batoid.zernikeTA(
                fiducial_telescope, thx, thy, wavelength,
                nrad=20, naz=120, reference='chief',
                jmax=66, eps=0.61
            )
            z_perturbs[i] = batoid.zernikeTA(
                telescope, thx, thy, wavelength,
                nrad=20, naz=120, reference='chief',
                jmax=66, eps=0.61
            )
        z_refs *= wavelength
        z_perturbs *= wavelength

        dz_ref = batoid.analysis.doubleZernike(
            fiducial_telescope, np.deg2rad(1.8), wavelength, rings=10,
            kmax=10, jmax=66, eps=0.61
        )
        dz_perturb = batoid.analysis.doubleZernike(
            telescope, np.deg2rad(1.8), wavelength, rings=10,
            kmax=10, jmax=66, eps=0.61
        )

        dz_terms = (
            (1, 4),                          # defocus
            (2, 4), (3, 4),                  # field tilt
            (2, 5), (3, 5), (2, 6), (3, 6),  # linear astigmatism
            (1, 7), (1, 8)                   # constant coma
        )
        dz_true = np.empty(len(dz_terms))
        for i, term in enumerate(dz_terms):
            dz_true[i] = (dz_perturb[term] - dz_ref[term])
        dz_true *= wavelength

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            mask_params=Rubin_obsc,
            focal_length=10.31, pixel_scale=10e-6
        )

        # Toy zfitter to make test images
        fitter0 = danish.DZMultiDonutModel(
            factory, z_refs=z_perturbs, dz_terms=(),
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys, bkg_order=-1
        )

        dxs = rng.uniform(-0.5, 0.5, nstar)
        dys = rng.uniform(-0.5, 0.5, nstar)
        fwhm = rng.uniform(0.5, 1.5)
        sky_levels = [1000.0]*nstar
        fluxes = rng.uniform(3e6, 1e7, nstar)

        imgs = fitter0.model(
            fluxes, dxs, dys, fwhm, (), sky_levels=sky_levels
        )

        # Actual fitter with DOF to optimize...
        fitter = danish.DZMultiDonutModel(
            factory, z_refs=z_refs, dz_terms=dz_terms,
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys, bkg_order=0
        )

        guess = [np.sum(img) for img in imgs]
        guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
        guess += [0.0]*(fitter.nbkg*nstar)

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(imgs, sky_levels)
        )
        result = fitter.unpack_params(result.x)

        fwhm_fit = result["fwhm"]
        dz_fit = result["wavefront_params"]
        np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.02)
        np.testing.assert_allclose(
            dz_fit/wavelength,
            dz_true/wavelength,
            rtol=0, atol=0.15
        )
        rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.2, "rms %9.3f > 0.2" % rms

        # mods = fitter.model(**result)
        # plot_dz_results(
        #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
        # )


@timer
def test_dz_fitter_LSST_z_perturbation(run_slow):
    """Roundtrip using danish model to produce a test images of rigid-body
    perturbed LSST transverse Zernikes.  Model and fitter run through the same
    code.
    """
    fiducial_telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    fiducial_telescope = fiducial_telescope.withGloballyShiftedOptic(
        "Detector",
        [0, 0, 0.0015]
    )

    wavelength = 750e-9

    rng = np.random.default_rng(124)
    if run_slow:
        niter = 10
    else:
        niter = 1
    for _ in range(niter):
        M2_dx, M2_dy = rng.uniform(-2e-4, 2e-4, size=2)
        M2_dz = rng.uniform(-2e-5, 2e-5)
        M2_thx, M2_thy = rng.uniform(-2e-5, 2e-5, size=2)
        cam_dx, cam_dy = rng.uniform(-2e-3, 2e-3, size=2)
        cam_dz = rng.uniform(-2e-5, 2e-5)
        cam_thx, cam_thy = rng.uniform(-2e-4, 2e-4, size=2)
        telescope = (
            fiducial_telescope
            .withGloballyShiftedOptic("M2", [M2_dx, M2_dy, M2_dz])
            .withLocallyRotatedOptic(
                "M2", batoid.RotX(M2_thx)@batoid.RotY(M2_thy)
            )
            .withGloballyShiftedOptic("LSSTCamera", [cam_dx, cam_dy, cam_dz])
            .withLocallyRotatedOptic(
                "LSSTCamera", batoid.RotX(cam_thx)@batoid.RotY(cam_thy)
            )
        )
        M1 = telescope['M1']
        M1_a = np.zeros(23)
        M1_a[9:16] = rng.uniform(-20e-9, 20e-9, 7)
        telescope = telescope.withSurface(
            "M1",
            batoid.Sum([
                M1.surface,
                batoid.Zernike(
                    M1_a,
                    R_outer=M1.obscuration.original.outer,
                    R_inner=M1.obscuration.original.inner
                )
            ])
        )

        nstar = 10
        thr = np.sqrt(rng.uniform(0, 1.8**2, nstar))
        ph = rng.uniform(0, 2*np.pi, nstar)
        thxs, thys = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
        z_refs = np.empty((nstar, 67))
        z_perturbs = np.empty((nstar, 67))
        for i, (thx, thy) in enumerate(zip(thxs, thys)):
            z_refs[i] = batoid.zernikeTA(
                fiducial_telescope, thx, thy, wavelength,
                nrad=20, naz=120, reference='chief',
                jmax=66, eps=0.61
            )
            z_perturbs[i] = batoid.zernikeTA(
                telescope, thx, thy, wavelength,
                nrad=20, naz=120, reference='chief',
                jmax=66, eps=0.61
            )
        z_refs *= wavelength
        z_perturbs *= wavelength

        dz_ref = batoid.analysis.doubleZernike(
            fiducial_telescope, np.deg2rad(1.8), wavelength, rings=10,
            kmax=10, jmax=66, eps=0.61
        )
        dz_perturb = batoid.analysis.doubleZernike(
            telescope, np.deg2rad(1.8), wavelength, rings=10,
            kmax=10, jmax=66, eps=0.61
        )

        dz_terms = (
            (1, 4),                          # defocus
            (2, 4), (3, 4),                  # field tilt
            (2, 5), (3, 5), (2, 6), (3, 6),  # linear astigmatism
            (1, 7), (1, 8),                  # constant coma
            (1, 9), (1, 10),                 # constant trefoil
            (1, 11),                         # constant spherical
            (1, 12), (1, 13),                # second astigmatism
            (1, 14), (1, 15)                 # quatrefoil
        )
        dz_true = np.empty(len(dz_terms))
        for i, term in enumerate(dz_terms):
            dz_true[i] = (dz_perturb[term] - dz_ref[term])
        dz_true *= wavelength

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            mask_params=Rubin_obsc,
            focal_length=10.31, pixel_scale=10e-6
        )

        # Toy zfitter to make test images
        fitter0 = danish.DZMultiDonutModel(
            factory, z_refs=z_perturbs, dz_terms=(),
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys, bkg_order=-1
        )

        dxs = rng.uniform(-0.5, 0.5, nstar)
        dys = rng.uniform(-0.5, 0.5, nstar)
        fwhm = rng.uniform(0.5, 1.5)
        sky_levels = [1000.0]*nstar
        fluxes = rng.uniform(3e6, 1e7, nstar)

        imgs = fitter0.model(
            fluxes, dxs, dys, fwhm, (), sky_levels=sky_levels
        )

        # Actual fitter with DOF to optimize...
        fitter = danish.DZMultiDonutModel(
            factory, z_refs=z_refs, dz_terms=dz_terms,
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys, bkg_order=0
        )

        guess = [np.sum(img) for img in imgs]
        guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
        guess += [0.0]*(fitter.nbkg*nstar)

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(imgs, sky_levels)
        )
        result = fitter.unpack_params(result.x)


        fwhm_fit = result["fwhm"]
        dz_fit = result["wavefront_params"]
        np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.02)
        np.testing.assert_allclose(
            dz_fit/wavelength,
            dz_true/wavelength,
            rtol=0, atol=0.15
        )
        rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.2, "rms %9.3f > 0.2" % rms

        # mods = fitter.model(**result)
        # plot_dz_results(
        #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
        # )


@timer
def test_dz_fitter_LSST_kolm():
    """Roundtrip using GalSim Kolmogorov atmosphere + batoid to produce test
    image of AOS DOF perturbed optics.  Model and fitter run independent code.
    """
    with open(
        os.path.join(directory, "data", "test_kolm_donuts.pkl"),
        'rb'
    ) as f:
        data = pickle.load(f)

    obsc = Rubin_obsc.copy()
    del obsc['Spider_3D']
    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=obsc,
        focal_length=10.31, pixel_scale=10e-6
    )
    sky_level = data[0]['sky_level']
    wavelength = data[0]['wavelength']
    dz_ref = data[0]['dz_ref']
    dz_actual = data[0]['dz_actual']
    fwhm = data[0]["fwhm"]

    thxs = []
    thys = []
    z_refs = []
    z_actuals = []
    imgs = []
    for datum in data[1:]:
        thxs.append(datum['thx'])
        thys.append(datum['thy'])
        z_refs.append(datum['z_ref'])
        z_actuals.append(datum['z_actual'])
        imgs.append(datum['arr'])

    dz_terms = (
        (1, 4),                          # defocus
        (2, 4), (3, 4),                  # field tilt
        (2, 5), (3, 5), (2, 6), (3, 6),  # linear astigmatism
        (1, 7), (1, 8),                  # constant coma
        (1, 9), (1, 10),                 # constant trefoil
        (1, 11),                         # constant spherical
        (1, 12), (1, 13),                # second astigmatism
        (1, 14), (1, 15),                # quatrefoil
        (1, 16), (1, 17),
        (1, 18), (1, 19),
        (1, 20), (1, 21),
        (1, 22)
    )

    dz_true = np.empty(len(dz_terms))
    for i, term in enumerate(dz_terms):
        dz_true[i] = (dz_actual[term] - dz_ref[term])*wavelength

    fitter = danish.DZMultiDonutModel(
        factory, z_refs=np.array(z_refs)*wavelength, dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0
    )
    nstar = len(thxs)
    guess = [np.sum(img) for img in imgs]
    guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
    guess += [0.0]*(fitter.nbkg*nstar)
    sky_levels = [sky_level]*nstar

    result = least_squares(
        fitter.chi, guess, jac=fitter.jac,
        ftol=1e-3, xtol=1e-3, gtol=1e-3,
        max_nfev=20, verbose=2,
        args=(imgs, sky_levels)
    )
    result = fitter.unpack_params(result.x)

    fwhm_fit = result["fwhm"]
    dz_fit = result["wavefront_params"]
    np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.02)
    np.testing.assert_allclose(
        dz_fit/wavelength,
        dz_true/wavelength,
        rtol=0, atol=0.1
    )
    rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
    print(f"rms = {rms:9.3f} waves")
    assert rms < 0.2, "rms %9.3f > 0.2" % rms

    # mods = fitter.model(**result)
    # plot_dz_results(
    #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
    # )


@timer
def test_dz_fitter_LSST_atm():
    """Roundtrip using GalSim phase screen atmosphere + batoid to produce test
    image of AOS DOF perturbed optics.  Model and fitter run independent code.
    """
    with open(
        os.path.join(directory, "data", "test_atm_donuts.pkl"),
        'rb'
    ) as f:
        data = pickle.load(f)

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6
    )
    sky_level = data[0]['sky_level']
    wavelength = data[0]['wavelength']
    dz_ref = data[0]['dz_ref']
    dz_actual = data[0]['dz_actual']
    fwhm = data[0]["fwhm"]

    thxs = []
    thys = []
    z_refs = []
    z_actuals = []
    imgs = []
    for datum in data[1:]:
        thxs.append(datum['thx'])
        thys.append(datum['thy'])
        z_refs.append(datum['z_ref'])
        z_actuals.append(datum['z_actual'])
        imgs.append(datum['arr'])

    dz_terms = (
        (1, 4),                          # defocus
        (2, 4), (3, 4),                  # field tilt
        (2, 5), (3, 5), (2, 6), (3, 6),  # linear astigmatism
        (1, 7), (1, 8),                  # constant coma
        (1, 9), (1, 10),                 # constant trefoil
        (1, 11),                         # constant spherical
        (1, 12), (1, 13),                # second astigmatism
        (1, 14), (1, 15),                # quatrefoil
        (1, 16), (1, 17),
        (1, 18), (1, 19),
        (1, 20), (1, 21),
        (1, 22)
    )

    dz_true = np.empty(len(dz_terms))
    for i, term in enumerate(dz_terms):
        dz_true[i] = (dz_actual[term] - dz_ref[term])*wavelength

    fitter = danish.DZMultiDonutModel(
        factory, z_refs=np.array(z_refs)*wavelength, dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0
    )
    nstar = len(thxs)
    guess = [np.sum(img) for img in imgs]
    guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
    guess += [0.0]*(fitter.nbkg*nstar)
    sky_levels = [sky_level]*nstar

    result = least_squares(
        fitter.chi, guess, jac=fitter.jac,
        ftol=1e-3, xtol=1e-3, gtol=1e-3,
        max_nfev=20, verbose=2,
        args=(imgs, sky_levels)
    )
    result = fitter.unpack_params(result.x)

    fwhm_fit = result["fwhm"]
    dz_fit = result["wavefront_params"]
    np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.02)
    np.testing.assert_allclose(
        dz_fit/wavelength,
        dz_true/wavelength,
        rtol=0, atol=0.15
    )
    rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
    print(f"rms = {rms:9.3f} waves")
    assert rms < 0.2, "rms %9.3f > 0.2" % rms

    # mods = fitter.model(**result)
    # plot_dz_results(
    #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
    # )


@timer
def test_basis_dz_fitter_rigid():
    """Roundtrip using batoid to produce test images and a sensitivity matrix of
    rigid body perturbations.
    """
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    intra = telescope.withGloballyShiftedOptic("LSSTCamera", [0,0,-1.5e-3])
    wavelength = 620e-9

    # First compute the sensitivity matrix
    dz_kwargs = dict(field=np.deg2rad(1.8), wavelength=wavelength, kmax=3, jmax=11)
    dz0 = batoid.doubleZernike(telescope, **dz_kwargs) * wavelength
    sens = []

    # Mode 0 : cam dz
    perturbed = telescope.withGloballyShiftedOptic("LSSTCamera", [0,0,10e-6])
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per micron

    # Mode 1 : cam tip
    perturbed = telescope.withGloballyRotatedOptic("LSSTCamera", batoid.RotX(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per arcsec

    # Mode 2 : cam tilt
    perturbed = telescope.withGloballyRotatedOptic("LSSTCamera", batoid.RotY(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per arcsec

    # Mode 3 : M2 tip
    perturbed = telescope.withGloballyRotatedOptic("M2", batoid.RotX(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per arcsec

    # Mode 4 : M2 tilt
    perturbed = telescope.withGloballyRotatedOptic("M2", batoid.RotY(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per arcsec

    sens = np.array(sens)
    sens[:,0] = 0
    sens[...,:4] = 0

    # Run some simulations...
    rng = np.random.default_rng(57721)
    for _ in range(10):
        dz = rng.uniform(-20, 20) # microns
        dcam_rx = rng.uniform(-20, 20) # arcsec
        dcam_ry = rng.uniform(-20, 20) # arcsec
        dm2_rx =  rng.uniform(-20, 20) # arcsec
        dm2_ry =  rng.uniform(-20, 20) # arcsec
        perturbation = np.array([dz, dcam_rx, dcam_ry, dm2_rx, dm2_ry])
        perturbed = (
            intra
            .withGloballyShiftedOptic("LSSTCamera", [0,0,dz*1e-6])
            .withGloballyRotatedOptic("LSSTCamera", batoid.RotX(np.deg2rad(dcam_rx/3600)))
            .withGloballyRotatedOptic("LSSTCamera", batoid.RotY(np.deg2rad(dcam_ry/3600)))
            .withGloballyRotatedOptic("M2", batoid.RotX(np.deg2rad(dm2_rx/3600)))
            .withGloballyRotatedOptic("M2", batoid.RotY(np.deg2rad(dm2_ry/3600)))
        )
        nstar = 10
        thr = np.sqrt(rng.uniform(0, 1.8**2, nstar))
        ph = rng.uniform(0, 2*np.pi, nstar)
        thxs = thr*np.cos(ph)
        thys = thr*np.sin(ph)

        z_refs = np.empty((nstar, 67))
        z_perturbs = np.empty((nstar, 67))
        for i, (thx, thy) in enumerate(zip(thxs, thys)):
            z_refs[i] = batoid.zernikeTA(
                intra, *np.deg2rad([thx, thy]), wavelength,
                nrad=20, naz=120, reference='chief',
                jmax=66, eps=0.61
            )
            z_perturbs[i] = batoid.zernikeTA(
                perturbed, *np.deg2rad([thx, thy]), wavelength,
                nrad=20, naz=120, reference='chief',
                jmax=66, eps=0.61
            )
        z_refs *= wavelength
        z_perturbs *= wavelength

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            mask_params=Rubin_obsc,
            spider_angle=20,
            focal_length=10.31,
            pixel_scale=20e-6,  # Simulate binned pixels
        )

        dxs = rng.uniform(-0.5, 0.5, nstar)
        dys = rng.uniform(-0.5, 0.5, nstar)
        fwhm = rng.uniform(0.5, 1.5)
        sky_levels = [1000.0]*nstar
        fluxes = rng.uniform(3e6, 1e7, nstar)

        imgs = []
        for i in range(nstar):

            # Use SingleDonutModel with zernikes above to simulate donuts.
            sim_fitter = danish.SingleDonutModel(
                factory,
                z_terms=list(range(4, 12)),
                z_ref=z_refs[i],
                thx=np.deg2rad(thxs[i]),
                thy=np.deg2rad(thys[i]),
                npix=89
            )

            z_fit = (z_perturbs[i] - z_refs[i])[4:12]
            imgs.append(
                sim_fitter.model(
                    fluxes[i], dxs[i], dys[i], fwhm, z_fit, sky_level=sky_levels[i],
                )
            )

        # Now fit using the DZBasisMultiDonutModel
        fitter = danish.DZBasisMultiDonutModel(
            factory,
            sensitivity=sens,
            z_refs=z_refs,
            field_radius=np.deg2rad(1.8),
            thxs=np.deg2rad(thxs),
            thys=np.deg2rad(thys),
            npix=89,
            bkg_order=0,
            wavefront_step=0.1
        )

        guess = [np.sum(img) for img in imgs]
        guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(perturbation)
        guess += [0.0]*(fitter.nbkg*nstar)

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(imgs, sky_levels)
        )
        result = fitter.unpack_params(result.x)

        mods = fitter.model(**result)

        for i, name in enumerate(["cam dz", "cam rx", "cam ry", "m2 rx", "m2 ry"]):
            print(f"{name:10}  {perturbation[i]:6.2f} {result['wavefront_params'][i]:6.2f}")
        print()

        # Compare true/sim rms wavefront
        dz_diff = np.einsum("ikj,i->kj", sens, perturbation - result["wavefront_params"])
        rms = np.sqrt(np.sum(np.square(dz_diff)))
        assert rms/wavelength < 0.65  # Not great, but okay for unit test

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(nstar, 3, figsize=(3, nstar))
        # for i in range(nstar):
        #     axs[i, 0].imshow(imgs[i], origin='lower')
        #     axs[i, 1].imshow(mods[i], origin='lower')
        #     axs[i, 2].imshow(imgs[i]-mods[i], origin='lower')
        # for ax in axs.ravel():
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # plt.show()


def test_multi_donut_model_jac():
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    intra = telescope.withGloballyShiftedOptic("Detector", [0, 0, 0.0015])
    wavelength = 750e-9

    rng = np.random.default_rng(675849302)
    nstar = 10
    thr = np.sqrt(rng.uniform(0, 1.8**2, nstar))
    ph = rng.uniform(0, 2*np.pi, nstar)
    thxs, thys = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
    z_refs = np.empty((nstar, 67))
    for i, (thx, thy) in enumerate(zip(thxs, thys)):
        z_refs[i] = batoid.zernikeTA(
            intra, thx, thy, wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )
    z_refs *= wavelength
    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6
    )

    dz_terms = (
        (1, 4),                          # defocus
        (2, 4), (3, 4),                  # field tilt
        (2, 5), (3, 5), (2, 6), (3, 6),  # linear astigmatism
        (1, 7), (1, 8)                   # constant coma
    )
    dz_true = rng.uniform(-0.3, 0.3, size=len(dz_terms))*wavelength

    fitter = danish.DZMultiDonutModel(
        factory, z_refs=np.array(z_refs), dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0
    )

    dxs = rng.uniform(-0.5, 0.5, nstar)
    dys = rng.uniform(-0.5, 0.5, nstar)
    fwhm = rng.uniform(0.5, 1.5)
    sky_levels = [1000.0]*nstar
    fluxes = rng.uniform(3e6, 1e7, nstar)

    imgs = fitter.model(
        fluxes, dxs, dys, fwhm, dz_true, sky_levels=sky_levels,
    )

    guess = [np.sum(img) for img in imgs]
    guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
    guess += [0.0]*(fitter.nbkg*nstar)

    j1 = fitter.jac(guess, imgs, [1000]*nstar)
    j2 = fitter._jac2(guess, imgs, [1000]*nstar)
    np.testing.assert_array_equal(j1, j2)

    # Try basis fitter too
    # First compute the sensitivity matrix
    dz_kwargs = dict(field=np.deg2rad(1.8), wavelength=wavelength, kmax=3, jmax=11)
    dz0 = batoid.doubleZernike(telescope, **dz_kwargs) * wavelength
    sens = []

    # Mode 0 : cam dz
    perturbed = telescope.withGloballyShiftedOptic("LSSTCamera", [0,0,10e-6])
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per micron

    # Mode 1 : cam tip
    perturbed = telescope.withGloballyRotatedOptic("LSSTCamera", batoid.RotX(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per arcsec

    # Mode 2 : cam tilt
    perturbed = telescope.withGloballyRotatedOptic("LSSTCamera", batoid.RotY(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per arcsec

    # Mode 3 : M2 tip
    perturbed = telescope.withGloballyRotatedOptic("M2", batoid.RotX(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per arcsec

    # Mode 4 : M2 tilt
    perturbed = telescope.withGloballyRotatedOptic("M2", batoid.RotY(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10) # meters per arcsec

    sens = np.array(sens)
    sens[:,0] = 0
    sens[...,:4] = 0

    fitter = danish.DZBasisMultiDonutModel(
        factory,
        sensitivity=sens,
        z_refs=z_refs,
        field_radius=np.deg2rad(1.8),
        thxs=thxs,
        thys=thys,
        bkg_order=0,
        wavefront_step=0.1
    )

    guess = [np.sum(img) for img in imgs]
    guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*sens.shape[0]
    guess += [0.0]*(fitter.nbkg*nstar)

    j1 = fitter.jac(guess, imgs, [1000]*nstar)
    j2 = fitter._jac2(guess, imgs, [1000]*nstar)
    np.testing.assert_array_equal(j1, j2)


if __name__ == "__main__":
    runtests(__file__)
