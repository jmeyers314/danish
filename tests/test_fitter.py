import os
import pickle
import time
import yaml
import pytest

from scipy.optimize import least_squares
from scipy.sparse import issparse
import numpy as np
import batoid
from galsim.zernike import Zernike

import danish
from danish_test_helpers import timer, runtests

directory = os.path.dirname(__file__)

Rubin_obsc = yaml.safe_load(open(os.path.join(danish.datadir, 'RubinObsc.yaml')))
# v3.14 lacks the extra M1/M2/M3 baffle added in v1000 (now the RubinObsc.yaml
# default). A few fitter tests were validated against v3.14 geometry, so they
# pin to it explicitly rather than tracking the default.
Rubin_obsc_v314 = yaml.safe_load(open(
    os.path.join(danish.datadir, 'RubinObsc_v3.14_r_rtpp0_azp45_pp0d0.yaml')))
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
            ftol=1e-5, xtol=1e-5, gtol=1e-5,
            max_nfev=20, verbose=2, x_scale='jac',
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
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.06*wavelength)
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
            ftol=1e-5, xtol=1e-5, gtol=1e-5,
            max_nfev=20, verbose=2, x_scale='jac',
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
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.06*wavelength)
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
            ftol=1e-5, xtol=1e-5, gtol=1e-5,
            max_nfev=20, verbose=2, x_scale='jac',
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
            ftol=1e-5, xtol=1e-5, gtol=1e-5,
            max_nfev=20, verbose=2, x_scale='jac',
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
        mask_params=Rubin_obsc_v314,
        focal_length=10.31, pixel_scale=10e-6
    )
    binned_factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc_v314,
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
            max_nfev=20, verbose=2, x_scale='jac',
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
        assert rms1x1 < 0.2, "rms %9.3f > %9.3f" % (rms1x1, 0.2)

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
            max_nfev=20, verbose=2, x_scale='jac',
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
        assert rms2x2 < 0.2, "rms %9.3f > %9.3f" % (rms2x2, 0.2)

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
            max_nfev=20, verbose=2, x_scale='jac',
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
            max_nfev=20, verbose=2, x_scale='jac',
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
            max_nfev=20, verbose=2, x_scale='jac',
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
            max_nfev=20, verbose=2, x_scale='jac',
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
            max_nfev=20, verbose=2, x_scale='jac',
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
            max_nfev=20, verbose=2, x_scale='jac',
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

    lb = np.full(len(guess), -np.inf)
    lb[3*nstar] = 0.3  # fwhm lower bound in arcsec
    result = least_squares(
        fitter.chi, guess, jac=fitter.jac,
        bounds=(lb, np.inf),
        x_scale='jac',
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
        mask_params=Rubin_obsc_v314,
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

    lb = np.full(len(guess), -np.inf)
    lb[3*nstar] = 0.3  # fwhm lower bound in arcsec
    result = least_squares(
        fitter.chi, guess, jac=fitter.jac,
        bounds=(lb, np.inf),
        x_scale='jac',
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
    assert rms < 0.25, "rms %9.3f > 0.25" % rms

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
            max_nfev=20, verbose=2, x_scale='jac',
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


def check_jac_sparse(fitter, guess, imgs, vars, formats=('csr',), **kwargs):
    """Assert `jac_sparse` reproduces `jac` exactly, and is really sparse.

    Both differentiate the same way in the same order, so they should agree
    bit-for-bit rather than to a tolerance.  Any extra kwargs -- `method` --
    go to both.

    `formats` defaults to CSR alone since each extra format repeats the whole
    jacobian; pass both at one site to cover the CSC path, which is selectable
    from ts_wep config.
    """
    dense = fitter.jac(guess, imgs, vars, **kwargs)

    # assert_array_equal treats nan as equal to nan, so an all-nan jacobian
    # would sail through the comparisons below while testing nothing at all.
    assert np.all(np.isfinite(dense))
    assert np.any(dense != 0)

    nstar, nbkg, natm = fitter.nstar, fitter.nbkg, fitter.natm
    nwavefront = len(guess) - nbkg*nstar - (3*nstar + natm)
    band = 3 + natm + nwavefront + nbkg

    for fmt in formats:
        sparse = fitter.jac_sparse(guess, imgs, vars, format=fmt, **kwargs)
        assert issparse(sparse)
        assert sparse.format == fmt
        assert sparse.shape == dense.shape
        np.testing.assert_array_equal(sparse.toarray(), dense)

        # Each star's residuals depend only on its own flux/dx/dy/bkg plus the
        # shared atmospheric and wavefront terms, so the stored fraction should
        # be that band of columns out of len(params).  Building the dense block
        # with csc_matrix drops exact zeros, hence the inequality.  Shrinking
        # the storage is the whole point, so check that it did.
        assert sparse.nnz <= nstar * fitter.npix**2 * band
        if nstar > 1:
            assert sparse.nnz < dense.size


@timer
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
    check_jac_sparse(fitter, guess, imgs, [1000]*nstar, formats=('csr', 'csc'))

    # nbkg = 0 skips the background block entirely, which is the one branch of
    # the sparse assembly that no bkg_order=0 fitter ever takes.  Three stars
    # exercises it and keeps the extra jac evaluations cheap.
    nsub = 3
    fitter_nobkg = danish.DZMultiDonutModel(
        factory, z_refs=np.array(z_refs[:nsub]), dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs[:nsub], thys=thys[:nsub],
        bkg_order=-1
    )
    assert fitter_nobkg.nbkg == 0
    imgs_nobkg = fitter_nobkg.model(
        fluxes[:nsub], dxs[:nsub], dys[:nsub], fwhm, dz_true,
        sky_levels=sky_levels[:nsub],
    )
    guess_nobkg = [np.sum(img) for img in imgs_nobkg]
    guess_nobkg += [0.0]*nsub + [0.0]*nsub + [0.7] + [0.0]*len(dz_terms)
    check_jac_sparse(fitter_nobkg, guess_nobkg, imgs_nobkg, [1000]*nsub)

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
    check_jac_sparse(fitter, guess, imgs, [1000]*nstar)


@timer
def test_zernike_param_basis_is_linear():
    """_zernike_param_basis is only a basis if _get_z_fits is linear.

    A basis expansion sum_i params[i]*basis[..., i] reproduces a function only
    when that function is linear in params.  A parameterization that squared
    or normalized its parameters would make the expansion wrong rather than
    merely approximate, and nothing would complain, so pin it here for every
    subclass.
    """
    factory = danish.DonutTriangleFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
        nrad=10,
    )
    rng = np.random.default_rng(26457)
    nstar = 3
    thr = np.sqrt(rng.uniform(0, 1.8**2, nstar))
    ph = rng.uniform(0, 2*np.pi, nstar)
    thxs, thys = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
    z_refs = rng.uniform(-1e-6, 1e-6, (nstar, 30))
    z_refs[:, 4] += 20e-6

    dz_terms = ((1, 4), (2, 5), (3, 6), (1, 11), (1, 22))
    sens = rng.normal(size=(4, 3, 23)) * 1e-7
    fitters = [
        danish.DZMultiDonutModel(
            factory, z_refs=z_refs, dz_terms=dz_terms,
            field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0,
        ),
        danish.DZBasisMultiDonutModel(
            factory, z_refs=z_refs, sensitivity=sens,
            field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0,
        ),
    ]

    for fitter in fitters:
        n = fitter._nwavefront
        a = rng.uniform(-1e-6, 1e-6, n)
        b = rng.uniform(-1e-6, 1e-6, n)
        c = 3.7

        def z(p):
            return np.array(fitter._get_z_fits(p))

        scale = np.abs(z(a + b)).max()
        # Additive and homogeneous.  Also check z(0) == 0 explicitly: a stray
        # constant term would leave every recovered basis slice offset by it,
        # and the expansion below would then only agree where sum(params) == 1.
        np.testing.assert_allclose(z(np.zeros(n)), 0.0, rtol=0, atol=0)
        np.testing.assert_allclose(z(a + b), z(a) + z(b), rtol=0, atol=1e-12*scale)
        np.testing.assert_allclose(z(c*a), c*z(a), rtol=0, atol=1e-12*scale)

        # So the basis expansion reproduces _get_z_fits.
        basis = fitter._zernike_param_basis()
        assert basis.shape == (nstar, z(a).shape[1], n)
        np.testing.assert_allclose(
            sum(a[i]*basis[..., i] for i in range(n)), z(a),
            rtol=0, atol=1e-12*scale
        )


@timer
def test_multi_donut_model_projection_basis():
    """The precomputed projection must reproduce a full Zernike projection.

    It drops the Zernike coefficients the parameterization cannot reach and
    folds them into a constant, so this checks nothing was lost: the stored
    basis must agree with projecting the whole aberration vector, and its
    product with _zernike_param_basis must be the derivative with respect to the
    fitted parameters.
    """
    from galsim.zernike import Zernike

    factory = danish.DonutTriangleFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
        nrad=10,
    )
    rng = np.random.default_rng(31622)
    nstar = 2
    thxs = np.deg2rad(np.array([0.5, -1.2]))
    thys = np.deg2rad(np.array([-0.9, 0.3]))
    # Structure past the fitted terms, so the constant part carries something.
    z_refs = rng.uniform(-1e-6, 1e-6, (nstar, 40))
    z_refs[:, 4] += 20e-6

    dz_terms = tuple((1, j) for j in list(range(4, 12)) + [22, 23])
    fitter = danish.DZMultiDonutModel(
        factory, z_refs=z_refs, dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0,
    )
    nwf = len(dz_terms)

    # Unreachable coefficients must have been excluded.
    assert len(fitter._zk_reach) == nwf
    assert set(fitter._zk_reach) == {j for _, j in dz_terms}

    dz = rng.uniform(-2e-6, 2e-6, nwf)
    for i, (thx, thy) in enumerate(zip(thxs, thys)):
        x0, y0, dxdz, dydz = fitter._projections[i]
        verts = fitter._meshes[(thx, thy)]["vertices"]
        z_fit = fitter._get_z_fits(dz)[i]
        ab = np.array(z_refs[i])
        ab[:len(z_fit)] += z_fit

        xe, ye = danish.pupil_to_focal(
            verts[:, 0], verts[:, 1], aberrations=ab,
            R_outer=4.18, R_inner=2.5498, focal_length=10.31,
        )
        head = ab[fitter._zk_reach]
        np.testing.assert_allclose(
            x0 + dxdz @ head, xe/10e-6, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            y0 + dydz @ head, ye/10e-6, rtol=1e-10, atol=1e-10
        )

        # d(vertex)/d(wavefront parameter), against a difference quotient.
        dxdp = dxdz @ fitter._dzkdp[i]
        assert dxdp.shape == (len(verts), nwf)
        step = 1e-9
        for j in (0, nwf - 1):
            pert = np.zeros(nwf)
            pert[j] = step
            zf1 = fitter._get_z_fits(pert)[i]
            ab1 = np.array(z_refs[i])
            ab1[:len(zf1)] += zf1
            x1, _ = danish.pupil_to_focal(
                verts[:, 0], verts[:, 1], aberrations=ab1,
                R_outer=4.18, R_inner=2.5498, focal_length=10.31,
            )
            x0_, _ = danish.pupil_to_focal(
                verts[:, 0], verts[:, 1], aberrations=z_refs[i],
                R_outer=4.18, R_inner=2.5498, focal_length=10.31,
            )
            np.testing.assert_allclose(
                (x1 - x0_)/10e-6/step, dxdp[:, j], rtol=1e-6, atol=1e-6
            )


def _triangle_jac_fitter(nstar=3, nrad=12, bkg_order=0, **kwargs):
    """A triangle-mode multi-donut fitter plus a guess, images and variances."""
    rng = np.random.default_rng(14142)
    thr = np.sqrt(rng.uniform(0, 1.8**2, nstar))
    ph = rng.uniform(0, 2*np.pi, nstar)
    thxs, thys = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
    z_refs = rng.uniform(-2e-7, 2e-7, (nstar, 30))
    z_refs[:, 4] += 20e-6

    factory = danish.DonutTriangleFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
        nrad=nrad,
    )
    dz_terms = tuple((1, j) for j in range(4, 12))
    fitter = danish.DZMultiDonutModel(
        factory, z_refs=z_refs, dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys,
        bkg_order=bkg_order, **kwargs
    )
    fluxes = rng.uniform(3e6, 1e7, nstar)
    imgs = fitter.model(
        fluxes, [0.0]*nstar, [0.0]*nstar, 0.7,
        rng.uniform(-1e-7, 1e-7, len(dz_terms)),
        bkgs=[(0.0,)*fitter.nbkg]*nstar,
    )
    guess = np.array(
        list(fluxes) + [0.0]*nstar + [0.0]*nstar + [0.7]
        + [0.0]*len(dz_terms) + [0.0]*(fitter.nbkg*nstar)
    )
    return fitter, guess, imgs, [1000.0]*nstar


@timer
def test_analytic_jac_matches_finite_difference():
    """The analytic wavefront columns must be the limit the FD ones approach.

    Nothing else in this suite pins a Jacobian against an independent
    reference -- `_jac2` repeats `jac`'s own one-sided differences -- so a
    wrong derivative would otherwise pass.  Shrinking the difference step has
    to walk the FD columns toward the analytic ones until roundoff takes over,
    which both validates the analytic path and shows it is the more accurate
    of the two.
    """
    nstar = 2
    fitter, guess, imgs, vars_ = _triangle_jac_fitter(nstar=nstar, nrad=10)
    assert fitter._can_use_analytic()

    nwf = fitter._nwavefront
    wf0 = 3*nstar + fitter.natm
    cols = slice(wf0, wf0 + nwf)

    analytic = fitter.jac(guess, imgs, vars_)[:, cols]
    assert np.all(np.isfinite(analytic))
    assert np.any(analytic != 0)

    # Central differences, so the error falls with the step until roundoff in
    # the difference takes over and it climbs again.
    steps = (1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
    errs = []
    for step in steps:
        fd = np.empty_like(analytic)
        for k in range(nwf):
            lo = np.array(guess); lo[wf0 + k] -= step
            hi = np.array(guess); hi[wf0 + k] += step
            fd[:, k] = (
                fitter.chi(hi, imgs, vars_) - fitter.chi(lo, imgs, vars_)
            )/(2*step)
        errs.append(
            np.linalg.norm(fd - analytic)/np.linalg.norm(analytic)
        )

    # Monotone while truncation dominates, i.e. up to the turn.
    best = int(np.argmin(errs))
    assert best > 0, errs
    assert errs[:best+1] == sorted(errs[:best+1], reverse=True), errs
    # And it gets close enough that no wrong derivative could be mistaken for
    # it -- a column off by any fixed factor would plateau at that factor.
    assert errs[best] < 1e-6, errs

    # And the shipped one-sided scheme is the inaccurate one, by percents.
    fd_default = fitter.jac(guess, imgs, vars_, method='fd')[:, cols]
    rel = np.linalg.norm(fd_default - analytic)/np.linalg.norm(analytic)
    assert 1e-4 < rel < 0.5, rel


@timer
def test_model_many_honors_unreachable_zernikes():
    """A perturbation the parameterization cannot reach must still render.

    `model_many` takes arbitrary per-donut Zernike coefficients, while the
    precomputed projection spans only the fitted subspace and holds everything
    else at the reference.  Moving an unreachable coefficient therefore has to
    take the full-projection path, which this pins by comparing against a
    model whose reference already carries the perturbation -- an independent
    route to the same wavefront.
    """
    factory = danish.DonutTriangleFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
        nrad=8,
    )
    # Noll 5 alone is fitted, so defocus (Noll 4) is outside the subspace.
    dz_terms = ((1, 5),)
    field_radius = np.deg2rad(1.8)
    thxs, thys = np.deg2rad([1.0]), np.zeros(1)
    z_refs = np.zeros((1, 30))
    z_refs[:, 4] = 20e-6

    fitter = danish.DZMultiDonutModel(
        factory, z_refs=z_refs, dz_terms=dz_terms,
        field_radius=field_radius, thxs=thxs, thys=thys, bkg_order=0,
    )
    assert fitter._zk_reach.tolist() == [5]

    base = fitter.model_many(
        [1e6], [0.0], [0.0], 1.0, 0.0, 1.0, np.zeros((1, 30)), bkgs=[(0.0,)]
    )

    # 20 um of defocus -> 30 um, through the unreachable coefficient.
    z_fits = np.zeros((1, 30))
    z_fits[0, 4] = 10e-6
    got = fitter.model_many(
        [1e6], [0.0], [0.0], 1.0, 0.0, 1.0, z_fits, bkgs=[(0.0,)]
    )
    # A silently dropped coefficient would leave this identical to `base`.
    assert not np.allclose(got, base)

    # The same wavefront reached by moving the reference instead, where the
    # projection covers it and no fallback is involved.
    shifted = np.array(z_refs)
    shifted[0, 4] = 30e-6
    ref_model = danish.DZMultiDonutModel(
        factory, z_refs=shifted, dz_terms=dz_terms,
        field_radius=field_radius, thxs=thxs, thys=thys, bkg_order=0,
    )
    expected = ref_model.model_many(
        [1e6], [0.0], [0.0], 1.0, 0.0, 1.0, np.zeros((1, 30)), bkgs=[(0.0,)]
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10*expected.max())

    # A perturbation inside the subspace keeps the fast path, bit for bit.
    inside = np.zeros((1, 30))
    inside[0, 5] = 1e-7
    fast = fitter.model_many(
        [1e6], [0.0], [0.0], 1.0, 0.0, 1.0, inside, bkgs=[(0.0,)]
    )
    direct = fitter._model(
        1e6, 0.0, 0.0, 1.0, 0.0, 1.0,
        tuple(np.array(z_refs[0]) + inside[0]), (0.0,),
        thx=thxs[0], thy=thys[0], idx=0,
    )
    np.testing.assert_array_equal(fast[0], direct)


@timer
def test_repeated_field_angle_keeps_per_donut_references():
    """Two donuts at one field angle must keep their own reference wavefronts.

    The same sky position imaged twice -- two exposures of one star -- is a
    real fit, and the two donuts share a mesh but not a reference: each
    exposure has its own fixed aberrations.  Only the triangulation depends on
    the angle alone, so only it may be shared.

    Checked where the split is observable from outside: a donut's image must
    not depend on whether some *other* donut happens to sit at the same angle.
    The references below differ in Noll 20, outside the fitted dz_terms, so the
    difference lives entirely in the fixed part the projection constants fold
    in.
    """
    factory = danish.DonutTriangleFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
        nrad=8,
    )
    dz_terms = tuple((1, j) for j in range(4, 12))
    field_radius = np.deg2rad(1.8)
    thxs = np.deg2rad([1.0, 1.0, 0.5])
    thys = np.zeros(3)

    z_refs = np.zeros((3, 30))
    z_refs[:, 4] = 20e-6
    # Distinct *unreachable* aberrations: Noll 20 is absent from dz_terms, so
    # it is fixed for the whole fit and folded into each donut's projection
    # constant rather than carried as a column.
    z_refs[:, 20] = [1e-6, 5e-6, 9e-6]

    fluxes = [1e6]*3
    joint = danish.DZMultiDonutModel(
        factory, z_refs=z_refs, dz_terms=dz_terms,
        field_radius=field_radius, thxs=thxs, thys=thys, bkg_order=0,
    )
    imgs = joint.model(
        fluxes, [0.0]*3, [0.0]*3, 0.7, [0.0]*len(dz_terms), bkgs=[(0.0,)]*3
    )

    # The mesh really is shared -- otherwise this test proves nothing about
    # the sharing -- while every donut keeps its own projection.
    assert len(joint._meshes) == 2
    assert len(joint._projections) == 3

    # Each donut alone, where no dedup is possible, is the reference.
    for i in range(3):
        solo = danish.DZMultiDonutModel(
            factory, z_refs=z_refs[i:i+1], dz_terms=dz_terms,
            field_radius=field_radius, thxs=thxs[i:i+1], thys=thys[i:i+1],
            bkg_order=0,
        )
        expected = solo.model(
            fluxes[:1], [0.0], [0.0], 0.7, [0.0]*len(dz_terms), bkgs=[(0.0,)]
        )[0]
        np.testing.assert_allclose(imgs[i], expected, rtol=0, atol=0)

    # The analytic Jacobian reads the same projections, so it has to survive
    # the repeated angle too.
    guess = joint.pack_params(
        fluxes=fluxes, dxs=[0.0]*3, dys=[0.0]*3, fwhm=0.7,
        wavefront_params=[0.0]*len(dz_terms), bkgs=[(0.0,)]*3,
    )
    vars_ = [1000.0]*3
    analytic = joint.jac(guess, imgs, vars_)
    fd = joint.jac(guess, imgs, vars_, method='fd')
    wf0 = 3*3 + joint.natm
    cols = slice(wf0, wf0 + len(dz_terms))
    rel = (
        np.linalg.norm(fd[:, cols] - analytic[:, cols])
        / np.linalg.norm(analytic[:, cols])
    )
    assert rel < 0.5, rel


@timer
def test_analytic_jac_contracts():
    """Switching methods must not disturb the rest of the Jacobian.

    Only the wavefront columns differ between methods, `jac` and `jac_sparse`
    have to agree bit-for-bit under either, and 'fd' has to stay exactly what
    shipped before the analytic path existed -- it is the reference the
    convergence test above leans on.
    """
    for nstar, bkg_order in ((1, 0), (3, 1)):
        fitter, guess, imgs, vars_ = _triangle_jac_fitter(
            nstar=nstar, nrad=10, bkg_order=bkg_order
        )
        nwf = fitter._nwavefront
        wf0 = 3*nstar + fitter.natm
        cols = slice(wf0, wf0 + nwf)

        analytic = fitter.jac(guess, imgs, vars_)
        fd = fitter.jac(guess, imgs, vars_, method='fd')

        # Every non-wavefront column is produced the same way either way.
        np.testing.assert_array_equal(
            np.delete(analytic, cols, axis=1), np.delete(fd, cols, axis=1)
        )
        # 'fd' is still the pre-existing scheme, bit for bit.
        np.testing.assert_array_equal(fd, fitter._jac2(guess, imgs, vars_))

        for method in ('analytic', 'fd'):
            check_jac_sparse(
                fitter, guess, imgs, vars_, formats=('csr', 'csc'),
                method=method,
            )

        # An unrecognized method must not quietly fall back to differencing.
        for call in (fitter.jac, fitter.jac_sparse):
            with pytest.raises(ValueError, match="method must be one of"):
                call(guess, imgs, vars_, method='exact')


@timer
def test_analytic_jac_with_systematic_loss():
    """The loss' own dependence on the model must be differentiated too.

    `systematic_loss` divides by an error built from the model, so d(chi)/dp
    picks up a term beyond -1/sqrt(var + model).  Dropping it would still look
    plausible, so check a nonzero alpha against finite differences.
    """
    nstar = 1
    fitter, guess, imgs, vars_ = _triangle_jac_fitter(
        nstar=nstar, nrad=10, loss_fn=danish.systematic_loss(0.02)
    )
    nwf = fitter._nwavefront
    wf0 = 3*nstar + fitter.natm

    analytic = fitter.jac(guess, imgs, vars_)[:, wf0:wf0+nwf]
    step = 1e-10
    fd = np.empty_like(analytic)
    for k in range(nwf):
        lo = np.array(guess); lo[wf0 + k] -= step
        hi = np.array(guess); hi[wf0 + k] += step
        fd[:, k] = (
            fitter.chi(hi, imgs, vars_) - fitter.chi(lo, imgs, vars_)
        )/(2*step)
    rel = np.linalg.norm(fd - analytic)/np.linalg.norm(analytic)
    assert rel < 1e-5, rel


@timer
def test_analytic_jac_recovers_wavefront():
    """A fit driven by the analytic Jacobian must still converge."""
    nstar = 3
    fitter, guess, imgs, vars_ = _triangle_jac_fitter(nstar=nstar, nrad=12)
    rng = np.random.default_rng(16180)
    nwf = fitter._nwavefront
    truth = rng.uniform(-2e-7, 2e-7, nwf)

    fluxes = guess[:nstar]
    data = fitter.model(
        fluxes, [0.0]*nstar, [0.0]*nstar, 0.7, truth,
        bkgs=[(0.0,)*fitter.nbkg]*nstar,
    )
    result = least_squares(
        fitter.chi, guess, jac=fitter.jac, x_scale='jac',
        ftol=1e-4, xtol=1e-4, gtol=1e-4, max_nfev=30,
        args=(data, vars_),
    )
    fit = fitter.unpack_params(result.x)
    np.testing.assert_allclose(
        np.array(fit["wavefront_params"]), truth, rtol=0, atol=2e-8
    )
    np.testing.assert_allclose(fit["fwhm"], 0.7, rtol=0, atol=0.02)


def _single_jac_fitter(factory, nrad=12, **kwargs):
    """A SingleDonutModel plus a guess, image and variance.

    Mirrors `_triangle_jac_fitter` for the one-donut parameterization, but
    takes the factory so the same setup can exercise the analytic path and the
    fallback.
    """
    rng = np.random.default_rng(14142)
    thx, thy = np.deg2rad(0.9), np.deg2rad(-1.1)
    z_ref = rng.uniform(-2e-7, 2e-7, 30)
    z_ref[4] += 20e-6

    z_terms = np.arange(4, 23)
    fitter = danish.SingleDonutModel(
        factory, z_ref=z_ref, z_terms=z_terms, thx=thx, thy=thy,
        bkg_order=0, **kwargs
    )
    flux = rng.uniform(3e6, 1e7)
    img = fitter.model(
        flux, 0.0, 0.0, 0.7, rng.uniform(-1e-7, 1e-7, len(z_terms)),
        bkg=(0.0,)*fitter.nbkg,
    )
    guess = np.array(
        [flux, 0.0, 0.0, 0.7] + [0.0]*len(z_terms) + [0.0]*fitter.nbkg
    )
    return fitter, guess, img, 1000.0


@timer
def test_single_donut_analytic_jac():
    """SingleDonutModel is the nstar=1 case, so it gets the same guarantees.

    One donut reaches chi through the same kernel, convolution and loss as
    many, so the analytic Zernike columns must converge to the same limit the
    central differences do, and must leave every other column untouched.
    """
    factory = danish.DonutTriangleFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
        nrad=10,
    )
    fitter, guess, img, var = _single_jac_fitter(factory, nrad=10)
    assert fitter._can_use_analytic()

    nwf = len(fitter.z_terms)
    # (flux, dx, dy, fwhm, *z_fit, *bkg) -- 3 offsets plus the atm parameter.
    wf0 = 3 + fitter.natm
    cols = slice(wf0, wf0 + nwf)

    analytic = fitter.jac(guess, img, var)
    assert np.all(np.isfinite(analytic))
    assert np.any(analytic[:, cols] != 0)

    # Central differences, so the error falls with the step until roundoff in
    # the difference takes over.
    steps = (1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
    errs = []
    for step in steps:
        fd = np.empty((fitter.npix**2, nwf))
        for k in range(nwf):
            lo = np.array(guess); lo[wf0 + k] -= step
            hi = np.array(guess); hi[wf0 + k] += step
            fd[:, k] = (
                fitter.chi(hi, img, var) - fitter.chi(lo, img, var)
            )/(2*step)
        errs.append(
            np.linalg.norm(fd - analytic[:, cols])
            / np.linalg.norm(analytic[:, cols])
        )

    best = int(np.argmin(errs))
    assert best > 0, errs
    assert errs[:best+1] == sorted(errs[:best+1], reverse=True), errs
    assert errs[best] < 1e-6, errs

    # Switching methods must disturb nothing else, and 'fd' must stay exactly
    # the scheme that shipped before the analytic path existed.
    fd_default = fitter.jac(guess, img, var, method='fd')
    np.testing.assert_array_equal(
        np.delete(analytic, cols, axis=1), np.delete(fd_default, cols, axis=1)
    )
    # And the one-sided scheme is the inaccurate one, by percents.
    rel = (
        np.linalg.norm(fd_default[:, cols] - analytic[:, cols])
        / np.linalg.norm(analytic[:, cols])
    )
    assert 1e-4 < rel < 0.5, rel

    # An unrecognized method must not quietly fall back to differencing.
    with pytest.raises(ValueError, match="method must be one of"):
        fitter.jac(guess, img, var, method='exact')


@timer
def test_single_donut_analytic_jac_falls_back():
    """Without a mesh there is no analytic path, so 'analytic' must difference.

    The inverse factory renders focal-to-pupil and builds no projection basis,
    so the default method has to degrade to exactly the 'fd' result rather
    than fail.
    """
    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
    )
    fitter, guess, img, var = _single_jac_fitter(factory)
    assert not fitter._can_use_analytic()

    np.testing.assert_array_equal(
        fitter.jac(guess, img, var),
        fitter.jac(guess, img, var, method='fd'),
    )


@timer
def test_single_donut_analytic_jac_with_offsets():
    """x_offset / y_offset are constant across a fit, so they join the basis.

    `_precompute_projections` folds them into the constant part of the
    projection, which is only correct if the analytic columns still match a
    difference quotient taken through the offset model.
    """
    factory = danish.DonutTriangleFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
        nrad=10,
    )
    rng = np.random.default_rng(27182)
    offset = Zernike(
        np.concatenate([np.zeros(4), rng.uniform(-1e-7, 1e-7, 10)]),
        R_outer=4.18, R_inner=4.18*0.61,
    )
    fitter, guess, img, var = _single_jac_fitter(
        factory, x_offset=offset, y_offset=offset
    )
    assert fitter._can_use_analytic()

    nwf = len(fitter.z_terms)
    wf0 = 3 + fitter.natm
    analytic = fitter.jac(guess, img, var)[:, wf0:wf0+nwf]

    step = 1e-11
    fd = np.empty_like(analytic)
    for k in range(nwf):
        lo = np.array(guess); lo[wf0 + k] -= step
        hi = np.array(guess); hi[wf0 + k] += step
        fd[:, k] = (
            fitter.chi(hi, img, var) - fitter.chi(lo, img, var)
        )/(2*step)
    rel = np.linalg.norm(fd - analytic)/np.linalg.norm(analytic)
    assert rel < 1e-6, rel


@timer
def test_donut_model_without_wavefront_params():
    """A model with no wavefront parameters must still build and run.

    The projection basis spans the Zernike coefficients a fit varies, and a
    model used only to simulate images varies none, so there is no basis to
    build and it has to fall back to projecting from a Zernike series.
    """
    factory = danish.DonutTriangleFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
        nrad=10,
    )
    thxs = np.deg2rad(np.array([0.5, -1.2]))
    thys = np.deg2rad(np.array([-0.9, 0.3]))
    z_refs = np.zeros((2, 30))
    z_refs[:, 4] = 20e-6

    fitter = danish.DZMultiDonutModel(
        factory, z_refs=z_refs, dz_terms=(),
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0,
    )
    assert fitter._projections == {}
    imgs = fitter.model([1e6, 1e6], [0.0, 0.0], [0.0, 0.0], 0.7, ())
    assert np.all(np.isfinite(imgs))
    assert imgs.sum() > 0


@timer
def test_dz_multi_spot_model_fiducial():
    """Roundtrip using spot model to produce test images with fiducial LSST
    transverse Zernikes plus random double Zernike offsets, then fit to recover.
    """
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")

    wavelength = 750e-9

    rng = np.random.default_rng(2344)
    nstar = 10
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
    z_refs[:, 4] += 2e-6  # Works better if we're on a clear side of focus

    dz_terms = (
        (1, 4),                         # defocus
        (2, 4), (3, 4),                 # field tilt
        # (2, 5), (3, 5), (2, 6), (3, 6), # linear astigmatism
        (1, 7), (1, 8),                 # constant coma
        (1, 9), (1, 10),                # constant trefoil
        (1, 11),                        # constant spherical
    )
    dz_true = rng.uniform(-1, 1, size=len(dz_terms))*wavelength

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6
    )

    fitter = danish.DZMultiSpotModel(
        factory,
        z_refs=z_refs, dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8),
        thxs=thxs, thys=thys, bkg_order=-1,
        spot_nrad=40,
        gq_kwargs=dict(nrad=10, rmax=3.5),
    )

    dxs = rng.uniform(-0.5, 0.5, nstar)
    dys = rng.uniform(-0.5, 0.5, nstar)
    # fwhm = rng.uniform(0.5, 1.5)
    fwhm = 0.7
    sky_levels = [1000.0]*nstar
    fluxes = rng.uniform(3e6, 1e7, nstar)

    imgs = fitter.model(
        fluxes, dxs, dys, fwhm, dz_true, sky_levels=sky_levels,
    )

    guess = [np.sum(img) for img in imgs]
    guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
    guess += [0.0]*(fitter.nbkg*nstar)

    print()
    result = least_squares(
        fitter.chi, guess, jac=fitter.jac,
        ftol=1e-3, xtol=1e-3, gtol=1e-3,
        max_nfev=20, verbose=2, x_scale='jac',
        args=(imgs, sky_levels)
    )
    result = fitter.unpack_params(result.x)

    # mods = fitter.model(**result)
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

    fwhm_fit = result["fwhm"]
    dz_fit = result["wavefront_params"]

    for i, term in enumerate(dz_terms):
        print(f"term {term}  true {dz_true[i]/wavelength:6.3f}  fit {dz_fit[i]/wavelength:6.3f}")

    np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.1)
    np.testing.assert_allclose(
        dz_fit/wavelength,
        dz_true/wavelength,
        rtol=0, atol=0.25
    )
    rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
    print(f"spot rms = {rms:9.3f} waves")
    assert rms < 0.5, "rms %9.3f > 0.5" % rms


@timer
def test_dz_multi_spot_model_rigid_perturbation():
    """Roundtrip using spot model with rigid-body perturbations.
    """
    fiducial = batoid.Optic.fromYaml("LSST_i.yaml")
    # Go a bit out of focus to make the spot shapes more sensitive to perturbations
    fiducial = fiducial.withGloballyShiftedOptic("Detector", [0, 0, 50e-6])
    wavelength = 750e-9

    rng = np.random.default_rng(1234)
    M2_dx, M2_dy = rng.uniform(-200e-6, 200e-6, size=2) # meters
    M2_dz = rng.uniform(-20e-6, 20e-6) # meters
    M2_thx, M2_thy = rng.uniform(-1e-4, 1e-4, size=2) # radians
    cam_dx, cam_dy = rng.uniform(-2000e-6, 2000e-6, size=2) # meters
    cam_dz = rng.uniform(-20e-6, 20e-6) # meters
    cam_thx, cam_thy = rng.uniform(-1e-4, 1e-4, size=2) # radians
    telescope = (
        fiducial
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
            fiducial, thx, thy, wavelength,
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
        fiducial, np.deg2rad(1.8), wavelength, rings=10,
        kmax=10, jmax=66, eps=0.61
    )
    dz_perturb = batoid.analysis.doubleZernike(
        telescope, np.deg2rad(1.8), wavelength, rings=10,
        kmax=10, jmax=66, eps=0.61
    )

    dz_terms = (
        (1, 4),                         # defocus
        (2, 4), (3, 4),                 # field tilt
        (2, 5), (3, 5), (2, 6), (3, 6), # linear astigmatism
        (1, 7), (1, 8)                  # constant coma
    )
    dz_true = np.empty(len(dz_terms))
    for i, term in enumerate(dz_terms):
        dz_true[i] = (dz_perturb[term] - dz_ref[term])
    dz_true *= wavelength

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc_v314,
        focal_length=10.31, pixel_scale=10e-6
    )

    # Toy fitter to make test images from perturbed z_refs
    fitter0 = danish.DZMultiSpotModel(
        factory, z_refs=z_perturbs, dz_terms=(),
        field_radius=np.deg2rad(1.8),
        thxs=thxs, thys=thys, bkg_order=-1,
        spot_nrad=40,
        gq_kwargs=dict(nrad=10, rmax=3.5)
    )

    dxs = rng.uniform(-0.5, 0.5, nstar)
    dys = rng.uniform(-0.5, 0.5, nstar)
    fwhm = 0.5
    sky_levels = [1000.0]*nstar
    fluxes = rng.uniform(3e6, 1e7, nstar)

    imgs = fitter0.model(
        fluxes, dxs, dys, fwhm, (), sky_levels=sky_levels
    )

    # Actual fitter
    fitter = danish.DZMultiSpotModel(
        factory, z_refs=z_refs, dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8),
        thxs=thxs, thys=thys, bkg_order=0,
        spot_nrad=40,
        gq_kwargs=dict(nrad=10, rmax=3.5)
    )

    guess = [np.sum(img) for img in imgs]
    guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
    guess += [0.0]*(fitter.nbkg*nstar)

    print()
    result = least_squares(
        fitter.chi, guess, jac=fitter.jac,
        ftol=1e-3, xtol=1e-3, gtol=1e-3,
        max_nfev=20, verbose=2, x_scale='jac',
        args=(imgs, sky_levels)
    )
    result = fitter.unpack_params(result.x)

    # mods = fitter.model(**result)
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

    fwhm_fit = result["fwhm"]
    dz_fit = result["wavefront_params"]

    for i, term in enumerate(dz_terms):
        print(f"term {term}  true {dz_true[i]/wavelength:6.3f}  fit {dz_fit[i]/wavelength:6.3f}")

    np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.1)
    np.testing.assert_allclose(
        dz_fit/wavelength,
        dz_true/wavelength,
        rtol=0, atol=0.15
    )
    rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
    print(f"spot rigid rms = {rms:9.3f} waves")
    assert rms < 0.1, "rms %9.3f > 0.1" % rms


@timer
def test_dz_basis_multi_spot_model_rigid():
    """Roundtrip using spot model with sensitivity matrix for rigid body
    perturbations.
    """
    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    # Go a bit out of focus to make the spot images more sensitive for this test
    telescope = telescope.withGloballyShiftedOptic("LSSTCamera", [0,0,60e-6])
    wavelength = 620e-9

    dz_kwargs = dict(field=np.deg2rad(1.8), wavelength=wavelength, kmax=3, jmax=11)
    dz0 = batoid.doubleZernike(telescope, **dz_kwargs) * wavelength
    sens = []

    perturbed = telescope.withGloballyShiftedOptic("LSSTCamera", [0,0,10e-6])
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10)

    perturbed = telescope.withGloballyRotatedOptic("LSSTCamera", batoid.RotX(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10)

    perturbed = telescope.withGloballyRotatedOptic("LSSTCamera", batoid.RotY(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10)

    perturbed = telescope.withGloballyRotatedOptic("M2", batoid.RotX(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10)

    perturbed = telescope.withGloballyRotatedOptic("M2", batoid.RotY(np.deg2rad(10/3600)))
    dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
    sens.append((dz1 - dz0)/10)

    sens = np.array(sens)
    sens[:,0] = 0
    sens[...,:4] = 0

    rng = np.random.default_rng(57721)
    dz = rng.uniform(-20, 20)
    dcam_rx = rng.uniform(-20, 20)
    dcam_ry = rng.uniform(-20, 20)
    dm2_rx = rng.uniform(-20, 20)
    dm2_ry = rng.uniform(-20, 20)
    perturbation = np.array([dz, dcam_rx, dcam_ry, dm2_rx, dm2_ry])
    perturbed = (
        telescope
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
            telescope, *np.deg2rad([thx, thy]), wavelength,
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
        focal_length=10.31, pixel_scale=10e-6
    )

    dxs = rng.uniform(-0.5, 0.5, nstar)
    dys = rng.uniform(-0.5, 0.5, nstar)
    fwhm = 0.5
    sky_levels = [1000.0]*nstar
    fluxes = rng.uniform(3e6, 1e7, nstar)

    # Simulate spot images using perturbed zernikes
    sim_fitter = danish.DZMultiSpotModel(
        factory,
        z_refs=z_perturbs,
        dz_terms=(),
        field_radius=np.deg2rad(1.8),
        thxs=np.deg2rad(thxs),
        thys=np.deg2rad(thys),
        bkg_order=-1,
        spot_nrad=40,
        gq_kwargs=dict(nrad=10, rmax=3.5),
    )
    imgs = sim_fitter.model(
        fluxes, dxs, dys, fwhm, (), sky_levels=sky_levels
    )

    # Fit using sensitivity matrix
    fitter = danish.DZBasisMultiSpotModel(
        factory,
        sensitivity=sens,
        z_refs=z_refs,
        field_radius=np.deg2rad(1.8),
        thxs=np.deg2rad(thxs),
        thys=np.deg2rad(thys),
        bkg_order=0,
        wavefront_step=0.1,
        spot_nrad=40,
        gq_kwargs=dict(nrad=10, rmax=3.5),
    )

    guess = [np.sum(img) for img in imgs]
    guess += [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(perturbation)
    guess += [0.0]*(fitter.nbkg*nstar)

    print()
    result = least_squares(
        fitter.chi, guess, jac=fitter.jac,
        ftol=1e-3, xtol=1e-3, gtol=1e-3,
        max_nfev=20, verbose=2, x_scale='jac',
        args=(imgs, sky_levels)
    )
    result = fitter.unpack_params(result.x)

    # mods = fitter.model(**result)
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

    for i, name in enumerate(["cam dz", "cam rx", "cam ry", "m2 rx", "m2 ry"]):
        print(f"{name:10}  {perturbation[i]:6.2f} {result['wavefront_params'][i]:6.2f}")

    dz_diff = np.einsum("ikj,i->kj", sens, perturbation - result["wavefront_params"])
    rms = np.sqrt(np.sum(np.square(dz_diff)))
    assert rms/wavelength < 1.0  # Not great, but okay for this unit test


@timer
def test_multi_spot_model_jac():
    """Verify jac == _jac2 for spot models."""
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
        (1, 4),
        (2, 4), (3, 4),
        (2, 5), (3, 5), (2, 6), (3, 6),
        (1, 7), (1, 8)
    )
    dz_true = rng.uniform(-0.3, 0.3, size=len(dz_terms))*wavelength

    # Test fwhm mode
    fitter = danish.DZMultiSpotModel(
        factory, z_refs=np.array(z_refs), dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0,
        spot_nrad=40, gq_kwargs=dict(rmax=3.5),
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

    # Test ixx mode
    fitter_ixx = danish.DZMultiSpotModel(
        factory, z_refs=np.array(z_refs), dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0,
        spot_nrad=40, gq_kwargs=dict(rmax=3.5),
        atm_mode='ixx',
    )

    Ixx = 0.2
    Ixy = 0.01
    Iyy = 0.18

    imgs_ixx = fitter_ixx.model(
        fluxes, dxs, dys, wavefront_params=dz_true,
        Ixx=Ixx, Ixy=Ixy, Iyy=Iyy, sky_levels=sky_levels,
    )

    guess_ixx = list(fluxes)
    guess_ixx += [0.0]*nstar + [0.0]*nstar
    guess_ixx += [Ixx, Ixy, Iyy]
    guess_ixx += [0.0]*len(dz_terms)
    guess_ixx += [0.0]*(fitter_ixx.nbkg*nstar)

    j1_ixx = fitter_ixx.jac(guess_ixx, imgs_ixx, [1000]*nstar)
    j2_ixx = fitter_ixx._jac2(guess_ixx, imgs_ixx, [1000]*nstar)
    np.testing.assert_array_equal(j1_ixx, j2_ixx)

    # Test ixx mode on donut model too
    fitter_donut_ixx = danish.DZMultiDonutModel(
        factory, z_refs=np.array(z_refs), dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys, bkg_order=0,
        atm_mode='ixx',
    )

    imgs_donut = fitter_donut_ixx.model(
        fluxes, dxs, dys, wavefront_params=dz_true,
        Ixx=Ixx, Ixy=Ixy, Iyy=Iyy, sky_levels=sky_levels,
    )

    guess_donut_ixx = list(fluxes)
    guess_donut_ixx += [0.0]*nstar + [0.0]*nstar
    guess_donut_ixx += [Ixx, Ixy, Iyy]
    guess_donut_ixx += [0.0]*len(dz_terms)
    guess_donut_ixx += [0.0]*(fitter_donut_ixx.nbkg*nstar)

    j1_d = fitter_donut_ixx.jac(guess_donut_ixx, imgs_donut, [1000]*nstar)
    j2_d = fitter_donut_ixx._jac2(guess_donut_ixx, imgs_donut, [1000]*nstar)
    np.testing.assert_array_equal(j1_d, j2_d)
    # Only the donut model has jac_sparse; ixx mode is also the only natm=3
    # coverage it gets, and natm sets where the dense block starts.
    check_jac_sparse(fitter_donut_ixx, guess_donut_ixx, imgs_donut, [1000]*nstar)


@timer
def test_spot_image_regression():
    """Verify BaseSpotModel._model matches factory.spot_image for same inputs."""
    from danish.spot_model import BaseSpotModel

    fiducial = batoid.Optic.fromYaml("Rubin_v3.14_r.yaml")
    fiducial = fiducial.withGloballyShiftedOptic("LSSTCamera", [0,0,30e-6])
    wavelength = 622e-9
    focal_length = 10.31
    eps = 0.621

    rng = np.random.default_rng(31415)
    amplitude = 1500e-9
    jmax = 15
    coefs = rng.uniform(-1, 1, size=jmax+1)*amplitude/np.sqrt(jmax+1)
    coefs[:4] = 0.0
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

    size = 0.5/2.35
    sigma = size*5*10e-6
    s = sigma**2
    Ixx_m = s
    Iyy_m = s
    Ixy_m = 0.0

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=4.18*eps,
        mask_params=Rubin_obsc,
        focal_length=focal_length, pixel_scale=10e-6
    )

    # Get the factory result
    cov_meters = np.array([[Ixx_m, Ixy_m], [Ixy_m, Iyy_m]])
    factory_img, _, _, _ = factory.spot_factory.spot_image(
        aberrations=zTA,
        thx=thx, thy=thy,
        nrad=nrad,
        gq_kwargs=dict(cov=cov_meters, rmax=3.5)
    )

    # Get the model result using arcsec^2 covariance
    sky_scale = 3600*np.rad2deg(1/focal_length)*10e-6
    m2a = sky_scale / 10e-6  # meters_to_arcsec per component = sky_scale/pixel_scale
    # Actually: 1 meter = (1/arcsec_to_meters) arcsec
    arcsec_to_meters = 10e-6 / sky_scale
    Ixx_as = Ixx_m / arcsec_to_meters**2
    Iyy_as = Iyy_m / arcsec_to_meters**2
    Ixy_as = Ixy_m / arcsec_to_meters**2

    model = BaseSpotModel(
        factory, npix=15, seed=577215, bkg_order=-1,
        atm_mode='ixx', spot_nrad=nrad,
        gq_kwargs=dict(rmax=3.5),
    )
    model_img = model._model(
        1.0, 0.0, 0.0,
        Ixx_as, Ixy_as, Iyy_as,
        tuple(zTA), (),
        thx=thx, thy=thy,
    )

    # Normalize both to unit sum and compare
    factory_norm = factory_img / np.sum(factory_img)
    model_norm = model_img / np.sum(model_img)
    np.testing.assert_allclose(model_norm, factory_norm, atol=1e-10)


def _triangle_joint_fitter(kind='dz', nd=2, ns=2, nrad=10, spot_weight=2.0):
    """A joint fitter whose donut half renders from a triangle mesh.

    Mirrors `_triangle_jac_fitter` for the joint parameterization.  Only the
    donut sub-model needs the triangle factory -- the spot renderer has no
    analytic path either way -- so the spots keep the inverse factory the
    other joint tests use.  `kind` selects the DZ or sensitivity-matrix
    parameterization.
    """
    rng = np.random.default_rng(27182)
    n = nd + ns
    thr = np.sqrt(rng.uniform(0, 1.8**2, n))
    ph = rng.uniform(0, 2*np.pi, n)
    thxs, thys = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
    z_refs = rng.uniform(-2e-7, 2e-7, (n, 30))
    # Defocus the donuts only.  The same 20 um on a spot would smear it far
    # outside its 21-pixel stamp and leave the spot images identically zero,
    # which would make every spot-side assertion below vacuous.
    z_refs[:nd, 4] += 20e-6

    common = dict(
        R_outer=4.18, R_inner=2.5498, mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6,
    )
    tri_factory = danish.DonutTriangleFactory(nrad=nrad, **common)
    spot_factory = danish.DonutFactory(**common)

    field_radius = np.deg2rad(1.8)
    if kind == 'dz':
        dz_terms = tuple((1, j) for j in range(4, 12))
        nwf = len(dz_terms)
        shared = dict(dz_terms=dz_terms)
        donut_cls = danish.DZMultiDonutModel
        spot_cls = danish.DZMultiSpotModel
        joint_cls = danish.DZJointModel
        wf_kwargs = {}
    else:
        # A synthetic sensitivity matrix: batoid's doubleZernike is far slower
        # than anything being tested here, and the analytic path only cares
        # that _get_z_fits is linear in the parameters.
        nwf = 4
        sens = np.zeros((nwf, 4, 12))
        for m in range(nwf):
            sens[m, 1 + m % 3, 4 + m] = 1e-7
        shared = dict(sensitivity=sens)
        donut_cls = danish.DZBasisMultiDonutModel
        spot_cls = danish.DZBasisMultiSpotModel
        joint_cls = danish.DZBasisJointModel
        wf_kwargs = dict(wavefront_step=0.1)

    donut_model = donut_cls(
        tri_factory, z_refs=z_refs[:nd], field_radius=field_radius,
        thxs=thxs[:nd], thys=thys[:nd], bkg_order=0, **shared, **wf_kwargs
    )
    spot_model = spot_cls(
        spot_factory, z_refs=z_refs[nd:], field_radius=field_radius,
        thxs=thxs[nd:], thys=thys[nd:], bkg_order=0,
        spot_nrad=40, gq_kwargs=dict(rmax=3.5), **shared, **wf_kwargs
    )
    joint = joint_cls(donut_model, spot_model, spot_weight=spot_weight)

    d_fluxes = rng.uniform(3e6, 1e7, nd)
    s_fluxes = rng.uniform(3e6, 1e7, ns)
    d_imgs = donut_model.model(
        d_fluxes, [0.0]*nd, [0.0]*nd, 0.7, [0.0]*nwf,
        bkgs=[(0.0,)]*nd,
    )
    s_imgs = spot_model.model(
        s_fluxes, [0.0]*ns, [0.0]*ns, 0.7, [0.0]*nwf,
        bkgs=[(0.0,)]*ns,
    )
    guess = joint.pack_params(
        d_fluxes=d_fluxes, d_dxs=[0.0]*nd, d_dys=[0.0]*nd,
        s_fluxes=s_fluxes, s_dxs=[0.0]*ns, s_dys=[0.0]*ns,
        fwhm=0.7, wavefront_params=[0.0]*nwf,
        d_bkgs=[(0.0,)]*nd, s_bkgs=[(0.0,)]*ns,
    )
    return joint, guess, d_imgs, [1000.0]*nd, s_imgs, [1000.0]*ns


@timer
def test_joint_analytic_jac_contracts():
    """The analytic path must touch only the donut half's wavefront columns.

    The donut sub-model renders from a mesh here, so 'analytic' is really
    engaged; the spot sub-model never has an analytic path.  Everything
    outside the donut wavefront block therefore has to come out of either
    method the same way, and 'fd' has to stay exactly the finite-difference
    reference `_jac2` provides.
    """
    for kind in ('dz', 'basis'):
        joint, guess, d_imgs, d_vars, s_imgs, s_vars = (
            _triangle_joint_fitter(kind=kind)
        )
        assert joint.donut_model._can_use_analytic()

        nd, ns = joint.nd, joint.ns
        nwf = joint.donut_model._nwavefront
        wf0 = 3*nd + 3*ns + joint.natm
        cols = slice(wf0, wf0 + nwf)

        args = (d_imgs, d_vars, s_imgs, s_vars)
        analytic = joint.jac(guess, *args)
        fd = joint.jac(guess, *args, method='fd')
        assert np.all(np.isfinite(analytic))
        assert np.any(analytic != 0)

        # Only the wavefront columns are allowed to differ by method.
        np.testing.assert_array_equal(
            np.delete(analytic, cols, axis=1),
            np.delete(fd, cols, axis=1),
        )

        # And within those, only the donut rows.  The spot rows go through a
        # divide-then-multiply by sqrt_spot_weight that the joint-chi path
        # does not, so they agree to roundoff rather than bit-for-bit.
        np.testing.assert_allclose(
            analytic[joint.nchi_d:, cols], fd[joint.nchi_d:, cols],
            rtol=1e-12, atol=0,
        )

        # The spot rows have to be the spot sub-model's own wavefront columns,
        # scaled by sqrt_spot_weight.  Comparing against the sub-model is what
        # pins *which* spot parameter each joint column perturbed -- the check
        # above only says the two joint methods agree, which a uniformly wrong
        # index would satisfy too.
        s_jac = joint.spot_model.jac(
            np.array(joint._spot_packed(joint.unpack_params(guess))),
            s_imgs, s_vars,
        )
        s_wf0 = 3*ns + joint.natm
        s_wf = s_jac[:, s_wf0:s_wf0 + nwf]
        # Guard against the comparison being vacuous: a spot image is a
        # flux-normalized histogram of ray positions, so too small a step
        # moves no ray across a bin edge and every column comes out zero.
        assert np.any(s_wf != 0)
        np.testing.assert_allclose(
            analytic[joint.nchi_d:, cols],
            joint.sqrt_spot_weight * s_wf,
            rtol=1e-12, atol=0,
        )

        # The donut rows are where the two methods genuinely disagree: a
        # one-sided difference is wrong by percents, which is the whole point.
        # Were the analytic block silently returning the FD values, or were it
        # off by some fixed factor, this would fall outside the band.
        d_an = analytic[:joint.nchi_d, cols]
        rel = (
            np.linalg.norm(fd[:joint.nchi_d, cols] - d_an)
            / np.linalg.norm(d_an)
        )
        assert 1e-4 < rel < 0.5, rel

        # 'fd' is still the pre-existing scheme.
        np.testing.assert_allclose(
            fd, joint._jac2(guess, *args), atol=1e-10
        )

        # An unrecognized method must not quietly fall back to differencing.
        with pytest.raises(ValueError, match="method must be one of"):
            joint.jac(guess, *args, method='exact')


@timer
def test_joint_analytic_jac_rejects_wavefront_mismatch():
    """A params vector that disagrees with the donut sub-model must not pass.

    The joint wavefront count is inferred from `len(params)` while the donut
    sub-model knows its own, so a mismatch would otherwise mispack the donut
    params and misalign the analytic columns.
    """
    joint, guess, d_imgs, d_vars, s_imgs, s_vars = _triangle_joint_fitter()
    # Drop a wavefront parameter: every other count is unchanged, so the
    # inferred nwf comes out one short of the donut sub-model's.
    short = np.delete(np.asarray(guess, dtype=float), -1 - joint.ns)
    with pytest.raises(ValueError, match="wavefront parameter count mismatch"):
        joint.jac(short, d_imgs, d_vars, s_imgs, s_vars)


@timer
def test_joint_analytic_jac_recovers_wavefront():
    """A joint fit driven by the analytic Jacobian must still converge."""
    joint, guess, d_imgs, d_vars, s_imgs, s_vars = _triangle_joint_fitter()
    nwf = joint.donut_model._nwavefront
    rng = np.random.default_rng(31415)
    truth = rng.uniform(-2e-7, 2e-7, nwf)

    jd = joint.unpack_params(guess)
    d_data = joint.donut_model.model(
        jd['d_fluxes'], [0.0]*joint.nd, [0.0]*joint.nd, 0.7, truth,
        bkgs=[(0.0,)]*joint.nd,
    )
    s_data = joint.spot_model.model(
        jd['s_fluxes'], [0.0]*joint.ns, [0.0]*joint.ns, 0.7, truth,
        bkgs=[(0.0,)]*joint.ns,
    )
    result = least_squares(
        joint.chi, guess, jac=joint.jac, x_scale='jac',
        ftol=1e-4, xtol=1e-4, gtol=1e-4, max_nfev=30,
        args=(d_data, d_vars, s_data, s_vars),
    )
    fit = joint.unpack_params(result.x)
    # The residual floor here is the fixture's, not the Jacobian's: tightening
    # the fit tolerances by four orders of magnitude leaves it at 2.5e-8, and
    # the same fit driven by 'fd' is no better.  What this pins is that the
    # analytic Jacobian still points least_squares at the right answer.
    np.testing.assert_allclose(
        np.array(fit["wavefront_params"]), truth, rtol=0, atol=4e-8
    )
    np.testing.assert_allclose(fit["fwhm"], 0.7, rtol=0, atol=0.02)


@timer
def test_joint_model_jac():
    """Verify JointModel.jac == JointModel._jac2."""
    from danish.joint_model import DZJointModel, DZBasisJointModel

    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    intra = telescope.withGloballyShiftedOptic("Detector", [0, 0, 0.0015])
    wavelength = 750e-9

    rng = np.random.default_rng(675849302)
    nstar = 4
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
        (1, 4),
        (2, 4), (3, 4),
        (2, 5), (3, 5), (2, 6), (3, 6),
        (1, 7), (1, 8)
    )

    # Use first 2 stars for donuts, last 2 for spots
    nd = 2
    ns = nstar - nd

    donut_model = danish.DZMultiDonutModel(
        factory,
        z_refs=np.array(z_refs[:nd]),
        dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8),
        thxs=thxs[:nd], thys=thys[:nd],
        bkg_order=0,
    )
    spot_model = danish.DZMultiSpotModel(
        factory,
        z_refs=np.array(z_refs[nd:]),
        dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8),
        thxs=thxs[nd:], thys=thys[nd:],
        bkg_order=0,
        spot_nrad=40, gq_kwargs=dict(rmax=3.5),
    )

    joint = DZJointModel(donut_model, spot_model, spot_weight=2.0)

    dz_true = rng.uniform(-0.3, 0.3, size=len(dz_terms)) * wavelength
    fwhm = 0.7
    d_fluxes = rng.uniform(3e6, 1e7, nd)
    s_fluxes = rng.uniform(3e6, 1e7, ns)
    d_dxs = rng.uniform(-0.3, 0.3, nd)
    d_dys = rng.uniform(-0.3, 0.3, nd)
    s_dxs = rng.uniform(-0.3, 0.3, ns)
    s_dys = rng.uniform(-0.3, 0.3, ns)

    # Generate data
    d_imgs = donut_model.model(
        d_fluxes, d_dxs, d_dys, fwhm, dz_true,
        sky_levels=[1000.0]*nd,
    )
    s_imgs = spot_model.model(
        s_fluxes, s_dxs, s_dys, fwhm, dz_true,
        sky_levels=[1000.0]*ns,
    )

    guess = joint.pack_params(
        d_fluxes=[np.sum(img) for img in d_imgs],
        d_dxs=[0.0]*nd, d_dys=[0.0]*nd,
        s_fluxes=[np.sum(img) for img in s_imgs],
        s_dxs=[0.0]*ns, s_dys=[0.0]*ns,
        fwhm=0.7,
        wavefront_params=[0.0]*len(dz_terms),
        d_bkgs=[(0.0,)]*nd,
        s_bkgs=[(0.0,)]*ns,
    )

    d_vars = [1000.0]*nd
    s_vars = [1000.0]*ns

    # 'fd' explicitly: _jac2 is the finite-difference reference, and this
    # factory has no mesh, so the default would only reach the same place by
    # falling back.  test_joint_analytic_jac_contracts covers the analytic one.
    j1 = joint.jac(guess, d_imgs, d_vars, s_imgs, s_vars, method='fd')
    j2 = joint._jac2(guess, d_imgs, d_vars, s_imgs, s_vars)
    np.testing.assert_allclose(j1, j2, atol=1e-10)

    # --- Also test DZBasisJointModel ---
    dz_kwargs = dict(field=np.deg2rad(1.8), wavelength=wavelength, kmax=3, jmax=11)
    dz0 = batoid.doubleZernike(telescope, **dz_kwargs) * wavelength
    sens = []
    for mode_name, optic, transform in [
        ("cam_dz", "LSSTCamera", lambda t: t.withGloballyShiftedOptic("LSSTCamera", [0,0,10e-6])),
        ("cam_rx", "LSSTCamera", lambda t: t.withGloballyRotatedOptic("LSSTCamera", batoid.RotX(np.deg2rad(10/3600)))),
        ("cam_ry", "LSSTCamera", lambda t: t.withGloballyRotatedOptic("LSSTCamera", batoid.RotY(np.deg2rad(10/3600)))),
    ]:
        perturbed = transform(telescope)
        dz1 = batoid.doubleZernike(perturbed, **dz_kwargs) * wavelength
        sens.append((dz1 - dz0) / 10)
    sens = np.array(sens)
    sens[:, 0] = 0
    sens[..., :4] = 0

    donut_basis = danish.DZBasisMultiDonutModel(
        factory,
        sensitivity=sens,
        z_refs=z_refs[:nd],
        field_radius=np.deg2rad(1.8),
        thxs=thxs[:nd], thys=thys[:nd],
        bkg_order=0,
        wavefront_step=0.1,
    )
    spot_basis = danish.DZBasisMultiSpotModel(
        factory,
        sensitivity=sens,
        z_refs=z_refs[nd:],
        field_radius=np.deg2rad(1.8),
        thxs=thxs[nd:], thys=thys[nd:],
        bkg_order=0,
        spot_nrad=40, gq_kwargs=dict(rmax=3.5),
        wavefront_step=0.1,
    )

    joint_basis = DZBasisJointModel(donut_basis, spot_basis, spot_weight=2.0)

    guess_basis = joint_basis.pack_params(
        d_fluxes=[np.sum(img) for img in d_imgs],
        d_dxs=[0.0]*nd, d_dys=[0.0]*nd,
        s_fluxes=[np.sum(img) for img in s_imgs],
        s_dxs=[0.0]*ns, s_dys=[0.0]*ns,
        fwhm=0.7,
        wavefront_params=[0.0]*sens.shape[0],
        d_bkgs=[(0.0,)]*nd,
        s_bkgs=[(0.0,)]*ns,
    )

    j1b = joint_basis.jac(
        guess_basis, d_imgs, d_vars, s_imgs, s_vars, method='fd'
    )
    j2b = joint_basis._jac2(guess_basis, d_imgs, d_vars, s_imgs, s_vars)
    np.testing.assert_allclose(j1b, j2b, atol=1e-10)


@timer
def test_dz_joint_model_roundtrip():
    """Joint fit of intra-focal donut, extra-focal donut, and in-focus spot
    from the same field position.  Verify recovery of Zernike wavefront.
    """
    fiducial = batoid.Optic.fromYaml("LSST_i.yaml")
    fiducial = fiducial.withGloballyShiftedOptic("Detector", [0, 0, 100e-6])
    wavelength = 750e-9

    # Apply a rigid-body perturbation
    rng = np.random.default_rng(9876)
    M2_dx, M2_dy = rng.uniform(-300e-6, 300e-6, size=2)
    M2_rx, M2_ry = rng.uniform(-30, 30, size=2)
    M2_dz = rng.uniform(-20e-6, 20e-6)
    cam_dx, cam_dy = rng.uniform(-2000e-6, 2000e-6, size=2)
    cam_dz = rng.uniform(-20e-6, 20e-6)
    perturbed = (
        fiducial
        .withGloballyShiftedOptic("M2", [M2_dx, M2_dy, M2_dz])
        .withGloballyRotatedOptic("M2", batoid.RotX(np.deg2rad(M2_rx/3600)))
        .withGloballyRotatedOptic("M2", batoid.RotY(np.deg2rad(M2_ry/3600)))
        .withGloballyShiftedOptic("LSSTCamera", [cam_dx, cam_dy, cam_dz])
    )

    # Stars at a single field position
    nstar = 1
    thx = np.deg2rad(1.5)
    thy = np.deg2rad(0.3)
    thxs = [thx]
    thys = [thy]

    # Compute reference zernikes for intra, extra, and in-focus
    intra_fiducial = fiducial.withGloballyShiftedOptic(
        "Detector", [0, 0, 0.0015]
    )
    extra_fiducial = fiducial.withGloballyShiftedOptic(
        "Detector", [0, 0, -0.0015]
    )
    intra_perturbed = perturbed.withGloballyShiftedOptic(
        "Detector", [0, 0, 0.0015]
    )
    extra_perturbed = perturbed.withGloballyShiftedOptic(
        "Detector", [0, 0, -0.0015]
    )

    zk_kwargs = dict(
        wavelength=wavelength,
        nrad=20, naz=120, reference='chief',
        jmax=66, eps=0.61
    )
    z_ref_intra = batoid.zernikeTA(
        intra_fiducial, thx, thy, **zk_kwargs
    ) * wavelength
    z_ref_extra = batoid.zernikeTA(
        extra_fiducial, thx, thy, **zk_kwargs
    ) * wavelength
    z_ref_spot = batoid.zernikeTA(
        fiducial, thx, thy, **zk_kwargs
    ) * wavelength
    z_perturb_intra = batoid.zernikeTA(
        intra_perturbed, thx, thy, **zk_kwargs
    ) * wavelength
    z_perturb_extra = batoid.zernikeTA(
        extra_perturbed, thx, thy, **zk_kwargs
    ) * wavelength
    z_perturb_spot = batoid.zernikeTA(
        perturbed, thx, thy, **zk_kwargs
    ) * wavelength

    # DZ terms with k=1 only (constant across field)
    dz_terms = (
        (1, 4),                  # defocus
        (1, 5), (1, 6),         # astigmatism
        (1, 7), (1, 8),         # coma
        (1, 9), (1, 10),        # trefoil
        (1, 11),                # spherical
    )

    # Compute ground truth DZ coefficients
    dz_ref_fiducial = batoid.analysis.doubleZernike(
        fiducial, np.deg2rad(1.8), wavelength, rings=10,
        kmax=10, jmax=66, eps=0.61
    )
    dz_perturbed = batoid.analysis.doubleZernike(
        perturbed, np.deg2rad(1.8), wavelength, rings=10,
        kmax=10, jmax=66, eps=0.61
    )
    dz_true = np.empty(len(dz_terms))
    for i, (k, j) in enumerate(dz_terms):
        dz_true[i] = (dz_perturbed[k, j] - dz_ref_fiducial[k, j]) * wavelength

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        mask_params=Rubin_obsc,
        focal_length=10.31, pixel_scale=10e-6
    )

    fwhm = 0.5
    sky_level_donut = 1000.0
    sky_level_spot = 1000.0
    flux_donut = 5e7
    flux_spot = 5e6

    # --- Generate synthetic intra-focal donut ---
    fitter0_intra = danish.DZMultiDonutModel(
        factory, z_refs=z_perturb_intra[np.newaxis], dz_terms=(),
        field_radius=np.deg2rad(1.8),
        thxs=thxs, thys=thys, bkg_order=-1
    )
    intra_img = fitter0_intra.model(
        [flux_donut], [0.0], [0.0], fwhm, (),
        sky_levels=[sky_level_donut]
    )

    # --- Generate synthetic extra-focal donut ---
    fitter0_extra = danish.DZMultiDonutModel(
        factory, z_refs=z_perturb_extra[np.newaxis], dz_terms=(),
        field_radius=np.deg2rad(1.8),
        thxs=thxs, thys=thys, bkg_order=-1
    )
    extra_img = fitter0_extra.model(
        [flux_donut], [0.0], [0.0], fwhm, (),
        sky_levels=[sky_level_donut]
    )

    # Stack intra + extra as 2-donut dataset
    donut_imgs = np.concatenate([intra_img, extra_img], axis=0)
    donut_vars = [sky_level_donut, sky_level_donut]

    # --- Generate synthetic in-focus spot ---
    fitter0_spot = danish.DZMultiSpotModel(
        factory, z_refs=z_perturb_spot[np.newaxis], dz_terms=(),
        field_radius=np.deg2rad(1.8),
        thxs=thxs, thys=thys, bkg_order=-1,
        spot_nrad=40,
        gq_kwargs=dict(nrad=10, rmax=3.5),
    )
    spot_img = fitter0_spot.model(
        [flux_spot], [0.0], [0.0], fwhm, (),
        sky_levels=[sky_level_spot]
    )
    spot_vars = [sky_level_spot]

    # --- Build the donut sub-model (intra + extra as 2 stars) ---
    # The donut model uses intra/extra z_refs for two "stars" at the same field
    donut_model = danish.DZMultiDonutModel(
        factory,
        z_refs=np.stack([z_ref_intra, z_ref_extra]),
        dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8),
        thxs=thxs*2, thys=thys*2,
        bkg_order=0,
    )

    # --- Build the spot sub-model ---
    spot_model = danish.DZMultiSpotModel(
        factory,
        z_refs=z_ref_spot[np.newaxis],
        dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8),
        thxs=thxs, thys=thys,
        bkg_order=-1,
        spot_nrad=40,
        gq_kwargs=dict(nrad=10, rmax=3.5),
    )

    joint = danish.DZJointModel(donut_model, spot_model)

    nd = donut_model.nstar  # 2
    ns = spot_model.nstar   # 1

    guess = joint.pack_params(
        d_fluxes=[np.sum(donut_imgs[0]), np.sum(donut_imgs[1])],
        d_dxs=[0.0]*nd, d_dys=[0.0]*nd,
        s_fluxes=[np.sum(spot_img[0])],
        s_dxs=[0.0]*ns, s_dys=[0.0]*ns,
        fwhm=0.7,
        wavefront_params=[0.0]*len(dz_terms),
        d_bkgs=[(0.0,)]*nd,
    )

    print()
    result = least_squares(
        joint.chi, guess, jac=joint.jac,
        ftol=1e-3, xtol=1e-3, gtol=1e-3,
        max_nfev=30, verbose=2, x_scale='jac',
        args=(donut_imgs, donut_vars, spot_img, spot_vars)
    )
    result = joint.unpack_params(result.x)

    # donut_mods, spot_mod = joint.model(**result)
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    # axs[0, 0].imshow(donut_imgs[0], origin='lower')
    # axs[0, 1].imshow(donut_mods[0], origin='lower')
    # axs[0, 2].imshow(donut_imgs[0]-donut_mods[0], origin='lower')
    # axs[1, 0].imshow(donut_imgs[1], origin='lower')
    # axs[1, 1].imshow(donut_mods[1], origin='lower')
    # axs[1, 2].imshow(donut_imgs[1]-donut_mods[1], origin='lower')
    # axs[2, 0].imshow(spot_img[0], origin='lower')
    # axs[2, 1].imshow(spot_mod[0], origin='lower')
    # axs[2, 2].imshow(spot_img[0]-spot_mod[0], origin='lower')
    # for ax in axs.ravel():
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # plt.show()

    dz_fit = np.array(result["wavefront_params"])
    fwhm_fit = result["fwhm"]

    print(f"fwhm: true={fwhm:.3f}  fit={fwhm_fit:.3f}")
    for i, term in enumerate(dz_terms):
        print(
            f"  {term}  true={dz_true[i]*1e6:7.3f}  "
            f"fit={dz_fit[i]*1e6:7.3f}  "
            f"diff={(dz_fit[i]-dz_true[i])*1e6:7.3f}"
        )

    rms = np.sqrt(np.sum(((dz_true - dz_fit) * 1e6) ** 2))
    print(f"joint rms = {rms:.3f} micron")

    np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=0.1)
    np.testing.assert_allclose(
        dz_fit * 1e6,
        dz_true * 1e6,
        rtol=0, atol=0.5,
    )
    assert rms < 0.5, f"rms {rms:.3f} > 0.5"


@timer
def test_systematic_loss():
    rng = np.random.default_rng(12345)
    data = rng.poisson(100, size=(20, 20)).astype(float)
    model = np.full((20, 20), 100.0)
    var = np.full((20, 20), 5.0)

    # systematic_loss(0) must equal chi2_loss (the default)
    r0 = danish.systematic_loss(0)(data, model, var)
    r_chi2 = danish.chi2_loss(data, model, var)
    np.testing.assert_array_equal(r0, r_chi2)

    # systematic_loss(alpha > 0) must produce smaller |residuals| than chi2_loss
    r_sys = danish.systematic_loss(0.1)(data, model, var)
    assert np.all(np.abs(r_sys) <= np.abs(r_chi2))

    # A SingleDonutModel constructed with systematic_loss should store it correctly
    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        focal_length=10.31, pixel_scale=10e-6
    )
    loss = danish.systematic_loss(0.05)
    fitter = danish.SingleDonutModel(factory, loss_fn=loss)
    assert fitter.loss_fn is loss


if __name__ == "__main__":
    runtests(__file__)
