import os
import pickle
import time
import yaml

from scipy.optimize import least_squares
import numpy as np
import batoid
from galsim.zernike import Zernike

import danish
from test_helpers import timer

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

    ax0.imshow(img/np.sum(img))
    ax1.imshow(mod/np.sum(mod))
    ax2.imshow(img/np.sum(img) - mod/np.sum(mod))
    ax3.axhline(0, c='k')
    if wavelength is not None:
        z_fit *= wavelength*1e9
        z_true *= wavelength*1e9
    ax3.plot(np.arange(4, jmax), z_fit, c='b', label='fit')
    ax3.plot(np.arange(4, jmax), z_true, c='k', label='truth')
    ax3.plot(
        np.arange(4, jmax),
        (z_fit-z_true),
        c='r', label='fit - truth'
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
        ax0.imshow(img/np.sum(img))
        ax1.imshow(mod/np.sum(mod))
        ax2.imshow(img/np.sum(img) - mod/np.sum(mod))
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
def test_fitter_LSST_fiducial():
    """ Roundtrip using danish model to produce a test image with fiducial LSST
    transverse Zernikes plus random Zernike offsets.  Model and fitter run
    through the same code.
    """
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0, 0, 0.0015])

    wavelength = 750e-9

    rng = np.random.default_rng(234)
    if __name__ == "__main__":
        niter = 10
    else:
        niter = 2
    for _ in range(niter):
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
            obsc_radii=Rubin_obsc['radii'],
            obsc_centers=Rubin_obsc['centers'],
            obsc_th_mins=Rubin_obsc['th_mins'],
            focal_length=10.31, pixel_scale=10e-6
        )

        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref, z_terms=z_terms, thx=thx, thy=thy
        )

        dx, dy = rng.uniform(-0.5, 0.5, size=2)
        fwhm = rng.uniform(0.5, 1.5)
        sky_level = 1000.0

        img = fitter.model(
            dx, dy, fwhm, z_true,
            sky_level=sky_level, flux=5e6
        )

        guess = [0.0, 0.0, 0.7]+[0.0]*19
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        for i in range(4, 23):
            out = f"{i:2d}  {result.x[i-1]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(result.x[i-1]-z_true[i-4])/wavelength:9.3f}"
            print(out)

        dx_fit, dy_fit, fwhm_fit, *z_fit = result.x
        z_fit = np.array(z_fit)

        # mod = fitter.model(
        #     dx_fit, dy_fit, fwhm_fit, z_fit
        # )
        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength)

        np.testing.assert_allclose(dx_fit, dx, rtol=0, atol=5e-2)
        np.testing.assert_allclose(dy_fit, dy, rtol=0, atol=5e-2)
        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.05*wavelength)
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
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
            thx=thx, thy=thy
        )

        dx, dy = rng.uniform(-0.5, 0.5, size=2)
        fwhm = rng.uniform(0.5, 1.5)
        sky_level = 1000.0

        img = fitter.model(
            dx, dy, fwhm, z_true,
            sky_level=sky_level, flux=5e6
        )

        guess = [0.0, 0.0, 0.7]+[0.0]*19
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        for i in range(4, 23):
            out = f"{i:2d}  {result.x[i-1]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(result.x[i-1]-z_true[i-4])/wavelength:9.3f}"
            print(out)

        dx_fit, dy_fit, fwhm_fit, *z_fit = result.x
        z_fit = np.array(z_fit)

        # mod = fitter.model(
        #     dx_fit, dy_fit, fwhm_fit, z_fit
        # )
        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength)

        np.testing.assert_allclose(dx_fit, dx, rtol=0, atol=5e-2)
        np.testing.assert_allclose(dy_fit, dy, rtol=0, atol=5e-2)
        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.05*wavelength)
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        assert rms < 0.1, "rms %9.3f > 0.1" % rms


@timer
def test_fitter_LSST_rigid_perturbation():
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
    if __name__ == "__main__":
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
            obsc_radii=Rubin_obsc['radii'],
            obsc_centers=Rubin_obsc['centers'],
            obsc_th_mins=Rubin_obsc['th_mins'],
            focal_length=10.31, pixel_scale=10e-6
        )

        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref, z_terms=z_terms, thx=thx, thy=thy
        )

        dx, dy = 0.0, 0.0
        fwhm = 0.7
        sky_level = 1000.0

        img = fitter.model(
            dx, dy, fwhm, z_true,
            sky_level=sky_level, flux=5e6
        )

        guess = [0.0, 0.0, 0.7]+[0.0]*19
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        for i in range(4, 23):
            out = f"{i:2d}  {result.x[i-1]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(result.x[i-1]-z_true[i-4])/wavelength:9.3f}"
            print(out)

        dx_fit, dy_fit, fwhm_fit, *z_fit = result.x
        z_fit = np.array(z_fit)
        # mod = fitter.model(
        #     dx_fit, dy_fit, fwhm_fit, z_fit
        # )
        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength)

        np.testing.assert_allclose(dx_fit, dx, rtol=0, atol=5e-2)
        np.testing.assert_allclose(dy_fit, dy, rtol=0, atol=5e-2)
        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.06*wavelength)
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        assert rms < 0.1, "rms %9.3f > 0.1" % rms


@timer
def test_fitter_LSST_z_perturbation():
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
    if __name__ == "__main__":
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
            obsc_radii=Rubin_obsc['radii'],
            obsc_centers=Rubin_obsc['centers'],
            obsc_th_mins=Rubin_obsc['th_mins'],
            focal_length=10.31, pixel_scale=10e-6
        )

        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref, z_terms=z_terms, thx=thx, thy=thy
        )

        dx, dy = 0.0, 0.0
        fwhm = 0.7
        sky_level = 1000.0

        img = fitter.model(
            dx, dy, fwhm, z_true,
            sky_level=sky_level, flux=5e6
        )

        guess = [0.0, 0.0, 0.7]+[0.0]*19
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        for i in range(4, 23):
            out = f"{i:2d}  {result.x[i-1]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(result.x[i-1]-z_true[i-4])/wavelength:9.3f}"
            print(out)

        dx_fit, dy_fit, fwhm_fit, *z_fit = result.x
        z_fit = np.array(z_fit)
        # mod = fitter.model(
        #     dx_fit, dy_fit, fwhm_fit, z_fit
        # )
        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength, wavelength=wavelength, ylim=(-200, 200))

        np.testing.assert_allclose(dx_fit, dx, rtol=0, atol=5e-2)
        np.testing.assert_allclose(dy_fit, dy, rtol=0, atol=5e-2)
        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.05*wavelength)
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        assert rms < 0.1, "rms %9.3f > 0.02" % rms


@timer
def test_fitter_LSST_kolm():
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
        obsc_radii=Rubin_obsc['radii'],
        obsc_centers=Rubin_obsc['centers'],
        obsc_th_mins=Rubin_obsc['th_mins'],
        focal_length=10.31, pixel_scale=10e-6
    )
    binned_factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        obsc_radii=Rubin_obsc['radii'],
        obsc_centers=Rubin_obsc['centers'],
        obsc_th_mins=Rubin_obsc['th_mins'],
        focal_length=10.31, pixel_scale=20e-6
    )

    if __name__ == "__main__":
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
            factory, z_ref=z_ref*wavelength, z_terms=z_terms, thx=thx, thy=thy
        )
        guess = [0.0, 0.0, 0.7] + [0.0]*19

        t0 = time.time()
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )
        t1 = time.time()
        t1x1 = t1 - t0

        z_true = (z_actual-z_ref)[4:23]*wavelength
        for i in range(4, 23):
            out = f"{i:2d}  {result.x[i-1]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(result.x[i-1]-z_true[i-4])/wavelength:9.3f}"
            print(out)

        dx_fit, dy_fit, fwhm_fit, *z_fit = result.x
        z_fit = np.array(z_fit)
        # mod = fitter.model(
        #     dx_fit, dy_fit, fwhm_fit, z_fit
        # )
        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength)
        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength, wavelength=wavelength, ylim=(-200, 200))

        # One fit is problematic.  It has a large field angle, so flip based on
        # that.
        if np.rad2deg(np.hypot(thx, thy)) > 1.7:
            tol = 0.7
        else:
            tol = 0.25

        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=tol*wavelength)
        rms1x1 = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        assert rms1x1 < 2*tol, "rms %9.3f > %9.3" % (rms1x1, tol)

        # Try binning 2x2
        binned_fitter = danish.SingleDonutModel(
            binned_factory, z_ref=z_ref*wavelength, z_terms=z_terms,
            thx=thx, thy=thy, npix=89
        )

        binned_img = img[:-1,:-1].reshape(90,2,90,2).mean(-1).mean(1)[:-1,:-1]
        t0 = time.time()
        binned_result = least_squares(
            binned_fitter.chi, guess, jac=binned_fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(binned_img, 4*sky_level)
        )
        t1 = time.time()
        t2x2 = t1 - t0
        dx_fit, dy_fit, fwhm_fit, *z_fit = binned_result.x
        z_fit = np.array(z_fit)
        # mod = binned_fitter.model(
        #     dx_fit, dy_fit, fwhm_fit, z_fit
        # )
        # plot_result(binned_img, mod, z_fit/wavelength, z_true/wavelength)
        # plot_result(binned_img, mod, z_fit/wavelength, z_true/wavelength, wavelength=wavelength, ylim=(-200, 200))

        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=tol*wavelength)
        rms2x2 = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        assert rms2x2 < 2*tol, "rms %9.3f > %9.3" % (rms2x2, tol)

        print("\n"*4)
        print(f"1x1 fit time: {t1x1:.3f} sec")
        print(f"2x2 fit time: {t2x2:.3f} sec")
        print(f"1x1 rms: {rms1x1}")
        print(f"2x2 rms: {rms2x2}")
        print("\n"*4)



@timer
def test_fitter_LSST_atm(plot=False):
    """Roundtrip using GalSim phase screen atmosphere + batoid to produce test
    image of AOS DOF perturbed optics.  Model and fitter run independent code.
    """
    with open(
        os.path.join(directory, "data", "test_atm_donuts.pkl"),
        'rb'
    ) as f:
        data = pickle.load(f)
    wavelength = data[0]['wavelength']

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        obsc_radii=Rubin_obsc['radii'],
        obsc_centers=Rubin_obsc['centers'],
        obsc_th_mins=Rubin_obsc['th_mins'],
        focal_length=10.31, pixel_scale=10e-6
    )
    binned_factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        obsc_radii=Rubin_obsc['radii'],
        obsc_centers=Rubin_obsc['centers'],
        obsc_th_mins=Rubin_obsc['th_mins'],
        focal_length=10.31, pixel_scale=20e-6
    )

    sky_level = data[0]['sky_level']
    if __name__ == "__main__":
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
            factory, z_ref=z_ref*wavelength, z_terms=z_terms, thx=thx, thy=thy
        )
        guess = [0.0, 0.0, 0.7] + [0.0]*19

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )

        z_true = (z_actual-z_ref)[4:23]*wavelength
        for i in range(4, 23):
            out = f"{i:2d}  {result.x[i-1]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(result.x[i-1]-z_true[i-4])/wavelength:9.3f}"
            print(out)

        dx_fit, dy_fit, fwhm_fit, *z_fit = result.x
        z_fit = np.array(z_fit)

        # mod = fitter.model(
        #     dx_fit, dy_fit, fwhm_fit, z_fit
        # )
        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength)
        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength, wavelength=wavelength, ylim=(-200, 200))

        np.testing.assert_allclose(
            z_fit/wavelength, z_true/wavelength,
            rtol=0, atol=0.5
        )
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.66, "rms %9.3f > 0.66" % rms

        # Try binning 2x2
        binned_fitter = danish.SingleDonutModel(
            binned_factory, z_ref=z_ref*wavelength, z_terms=z_terms,
            thx=thx, thy=thy, npix=89
        )

        binned_img = img[:-1,:-1].reshape(90,2,90,2).mean(-1).mean(1)[:-1,:-1]
        t0 = time.time()
        binned_result = least_squares(
            binned_fitter.chi, guess, jac=binned_fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(binned_img, 4*sky_level)
        )
        t1 = time.time()
        print(f"2x2 fit time: {t1-t0:.3f} sec")

        dx_fit, dy_fit, fwhm_fit, *z_fit = binned_result.x
        z_fit = np.array(z_fit)

        # mod = binned_fitter.model(
        #     dx_fit, dy_fit, fwhm_fit, z_fit
        # )
        # plot_result(binned_img, mod, z_fit/wavelength, z_true/wavelength)
        # plot_result(binned_img, mod, z_fit/wavelength, z_true/wavelength, wavelength=wavelength, ylim=(-200, 200))

        np.testing.assert_allclose(
            z_fit/wavelength,
            z_true/wavelength,
            rtol=0, atol=0.5
        )
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.66, "rms %9.3f > 0.66" % rms


@timer
def test_fitter_AuxTel_rigid_perturbation():
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
    if __name__ == "__main__":
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
            obsc_radii=AuxTel_obsc['radii'],
            obsc_centers=AuxTel_obsc['centers'],
            obsc_th_mins=AuxTel_obsc['th_mins'],
            focal_length=20.8, pixel_scale=10e-6
        )

        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref, z_terms=z_terms, thx=thx, thy=thy, npix=255
        )

        dx, dy = 0.0, 0.0
        fwhm = 0.7  # Arcsec for Kolmogorov profile
        sky_level = 1000.0  # counts per pixel

        # Make a test image using true aberrations
        img = fitter.model(
            dx, dy, fwhm, z_true,
            sky_level=sky_level, flux=5e6
        )

        # Now guess aberrations are 0.0, and try to recover truth.
        guess = [0.0, 0.0, 0.7]+[0.0]*8
        # We don't ship a custom fitting algorithm; just use scipy.least_squares
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, sky_level)
        )

        for i in range(4, 12):
            out = f"{i:2d}  {result.x[i-1]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(result.x[i-1]-z_true[i-4])/wavelength:9.3f}"
            print(out)

        dx_fit, dy_fit, fwhm_fit, *z_fit = result.x
        z_fit = np.array(z_fit)

        # # Optional visualization
        # mod = fitter.model(
        #     dx_fit, dy_fit, fwhm_fit, z_fit
        # )

        # plot_result(img, mod, z_fit/wavelength, z_true/wavelength, ylim=(-0.2, 0.2))

        np.testing.assert_allclose(dx_fit, dx, rtol=0, atol=1e-2)
        np.testing.assert_allclose(dy_fit, dy, rtol=0, atol=1e-2)
        np.testing.assert_allclose(fwhm_fit, fwhm, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.006*wavelength)
        rms = np.sqrt(np.sum(((z_true-z_fit)/wavelength)**2))
        assert rms < 0.1, "rms %9.3f > 0.1" % rms


@timer
def test_dz_fitter_LSST_fiducial():
    """ Roundtrip using danish model to produce test images with fiducial LSST
    transverse Zernikes plus random double Zernike offsets.  Model and fitter
    run through the same code.
    """
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", [0, 0, 0.0015])

    wavelength = 750e-9

    rng = np.random.default_rng(2344)
    nstar = 10
    if __name__ == "__main__":
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
            obsc_radii=Rubin_obsc['radii'],
            obsc_centers=Rubin_obsc['centers'],
            obsc_th_mins=Rubin_obsc['th_mins'],
            focal_length=10.31, pixel_scale=10e-6
        )

        fitter = danish.MultiDonutModel(
            factory,
            z_refs=z_refs, dz_terms=dz_terms,
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys
        )

        dxs = rng.uniform(-0.5, 0.5, nstar)
        dys = rng.uniform(-0.5, 0.5, nstar)
        fwhm = rng.uniform(0.5, 1.5)
        sky_levels = [1000.0]*nstar
        fluxes = [5e6]*nstar

        imgs = fitter.model(
            dxs, dys, fwhm, dz_true, sky_levels=sky_levels, fluxes=fluxes
        )

        guess = [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(imgs, sky_levels)
        )

        dxs_fit, dys_fit, fwhm_fit, dz_fit = fitter.unpack_params(result.x)

        np.testing.assert_allclose(dxs, dxs_fit, rtol=0, atol=0.2)
        np.testing.assert_allclose(dys, dys_fit, rtol=0, atol=0.2)
        np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.05)
        np.testing.assert_allclose(
            dz_fit/wavelength,
            dz_true/wavelength,
            rtol=0, atol=0.1
        )
        rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.05, "rms %9.3f > 0.05" % rms

        # dxs_fit, dys_fit, fwhm_fit, dz_fit = fitter.unpack_params(result.x)
        # mods = fitter.model(
        #     dxs_fit, dys_fit, fwhm_fit, dz_fit
        # )

        # plot_dz_results(
        #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
        # )


@timer
def test_dz_fitter_LSST_rigid_perturbation():
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
    if __name__ == "__main__":
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
            obsc_radii=Rubin_obsc['radii'],
            obsc_centers=Rubin_obsc['centers'],
            obsc_th_mins=Rubin_obsc['th_mins'],
            focal_length=10.31, pixel_scale=10e-6
        )

        # Toy zfitter to make test images
        fitter0 = danish.MultiDonutModel(
            factory, z_refs=z_perturbs, dz_terms=(),
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys
        )

        dxs = rng.uniform(-0.5, 0.5, nstar)
        dys = rng.uniform(-0.5, 0.5, nstar)
        fwhm = rng.uniform(0.5, 1.5)
        sky_levels = [1000.0]*nstar
        fluxes = [5e6]*nstar

        imgs = fitter0.model(
            dxs, dys, fwhm, (), sky_levels=sky_levels, fluxes=fluxes
        )

        # Actual fitter with DOF to optimize...
        fitter = danish.MultiDonutModel(
            factory, z_refs=z_refs, dz_terms=dz_terms,
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys
        )

        guess = [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(imgs, sky_levels)
        )

        dxs_fit, dys_fit, fwhm_fit, dz_fit = fitter.unpack_params(result.x)

        np.testing.assert_allclose(dxs, dxs_fit, rtol=0, atol=0.2)
        np.testing.assert_allclose(dys, dys_fit, rtol=0, atol=0.2)
        np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.05)
        np.testing.assert_allclose(
            dz_fit/wavelength,
            dz_true/wavelength,
            rtol=0, atol=0.1
        )
        rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.1, "rms %9.3f > 0.1" % rms

        # mods = fitter.model(
        #     dxs_fit, dys_fit, fwhm_fit, dz_fit
        # )

        # plot_dz_results(
        #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
        # )


@timer
def test_dz_fitter_LSST_z_perturbation():
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
    if __name__ == "__main__":
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
            obsc_radii=Rubin_obsc['radii'],
            obsc_centers=Rubin_obsc['centers'],
            obsc_th_mins=Rubin_obsc['th_mins'],
            focal_length=10.31, pixel_scale=10e-6
        )

        # Toy zfitter to make test images
        fitter0 = danish.MultiDonutModel(
            factory, z_refs=z_perturbs, dz_terms=(),
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys
        )

        dxs = rng.uniform(-0.5, 0.5, nstar)
        dys = rng.uniform(-0.5, 0.5, nstar)
        fwhm = rng.uniform(0.5, 1.5)
        sky_levels = [1000.0]*nstar
        fluxes = [5e6]*nstar

        imgs = fitter0.model(
            dxs, dys, fwhm, (), sky_levels=sky_levels, fluxes=fluxes
        )

        # Actual fitter with DOF to optimize...
        fitter = danish.MultiDonutModel(
            factory, z_refs=z_refs, dz_terms=dz_terms,
            field_radius=np.deg2rad(1.8),
            thxs=thxs, thys=thys
        )

        guess = [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(imgs, sky_levels)
        )

        dxs_fit, dys_fit, fwhm_fit, dz_fit = fitter.unpack_params(result.x)

        np.testing.assert_allclose(dxs, dxs_fit, rtol=0, atol=0.2)
        np.testing.assert_allclose(dys, dys_fit, rtol=0, atol=0.2)
        np.testing.assert_allclose(fwhm, fwhm_fit, rtol=0, atol=0.05)
        np.testing.assert_allclose(
            dz_fit/wavelength,
            dz_true/wavelength,
            rtol=0, atol=0.1
        )
        rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
        print(f"rms = {rms:9.3f} waves")
        assert rms < 0.2, "rms %9.3f > 0.2" % rms

        # mods = fitter.model(
        #     dxs_fit, dys_fit, fwhm_fit, dz_fit
        # )

        # plot_dz_results(
        #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
        # )


@timer
def test_dz_fitter_LSST_kolm():
    with open(
        os.path.join(directory, "data", "test_kolm_donuts.pkl"),
        'rb'
    ) as f:
        data = pickle.load(f)

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        obsc_radii=Rubin_obsc['radii'],
        obsc_centers=Rubin_obsc['centers'],
        obsc_th_mins=Rubin_obsc['th_mins'],
        focal_length=10.31, pixel_scale=10e-6
    )
    sky_level = data[0]['sky_level']
    wavelength = data[0]['wavelength']
    dz_ref = data[0]['dz_ref']
    dz_actual = data[0]['dz_actual']

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

    fitter = danish.MultiDonutModel(
        factory, z_refs=np.array(z_refs)*wavelength, dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys
    )
    nstar = len(thxs)
    guess = [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
    sky_levels = [sky_level]*nstar

    result = least_squares(
        fitter.chi, guess, jac=fitter.jac,
        ftol=1e-3, xtol=1e-3, gtol=1e-3,
        max_nfev=20, verbose=2,
        args=(imgs, sky_levels)
    )

    dxs_fit, dys_fit, fwhm_fit, dz_fit = fitter.unpack_params(result.x)

    np.testing.assert_allclose(
        dz_fit/wavelength,
        dz_true/wavelength,
        rtol=0, atol=0.1
    )
    rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
    print(f"rms = {rms:9.3f} waves")
    assert rms < 0.2, "rms %9.3f > 0.2" % rms

    # mods = fitter.model(
    #     dxs_fit, dys_fit, fwhm_fit, dz_fit
    # )

    # plot_dz_results(
    #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
    # )


@timer
def test_dz_fitter_LSST_atm():
    with open(
        os.path.join(directory, "data", "test_atm_donuts.pkl"),
        'rb'
    ) as f:
        data = pickle.load(f)

    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=2.5498,
        obsc_radii=Rubin_obsc['radii'],
        obsc_centers=Rubin_obsc['centers'],
        obsc_th_mins=Rubin_obsc['th_mins'],
        focal_length=10.31, pixel_scale=10e-6
    )
    sky_level = data[0]['sky_level']
    wavelength = data[0]['wavelength']
    dz_ref = data[0]['dz_ref']
    dz_actual = data[0]['dz_actual']

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

    fitter = danish.MultiDonutModel(
        factory, z_refs=np.array(z_refs)*wavelength, dz_terms=dz_terms,
        field_radius=np.deg2rad(1.8), thxs=thxs, thys=thys
    )
    nstar = len(thxs)
    guess = [0.0]*nstar + [0.0]*nstar + [0.7] + [0.0]*len(dz_terms)
    sky_levels = [sky_level]*nstar

    result = least_squares(
        fitter.chi, guess, jac=fitter.jac,
        ftol=1e-3, xtol=1e-3, gtol=1e-3,
        max_nfev=20, verbose=2,
        args=(imgs, sky_levels)
    )

    dxs_fit, dys_fit, fwhm_fit, dz_fit = fitter.unpack_params(result.x)

    np.testing.assert_allclose(
        dz_fit/wavelength,
        dz_true/wavelength,
        rtol=0, atol=0.2
    )
    rms = np.sqrt(np.sum(((dz_true-dz_fit)/wavelength)**2))
    print(f"rms = {rms:9.3f} waves")
    assert rms < 0.4, "rms %9.3f > 0.4" % rms

    # mods = fitter.model(
    #     dxs_fit, dys_fit, fwhm_fit, dz_fit
    # )

    # plot_dz_results(
    #     imgs, mods, dz_fit/wavelength, dz_true/wavelength, dz_terms
    # )


if __name__ == "__main__":
    test_fitter_LSST_fiducial()
    test_fitter_LSST_rigid_perturbation()
    test_fitter_LSST_z_perturbation()
    test_fitter_LSST_kolm()
    test_fitter_LSST_atm()

    test_fitter_AuxTel_rigid_perturbation()

    test_dz_fitter_LSST_fiducial()
    test_dz_fitter_LSST_rigid_perturbation()
    test_dz_fitter_LSST_z_perturbation()
    test_dz_fitter_LSST_kolm()
    test_dz_fitter_LSST_atm()
