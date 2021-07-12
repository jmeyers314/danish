from danish.fitter import SingleDonutModel
import numpy as np

import danish


def test_fitter():
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
        thx, thy = np.deg2rad(thr*np.cos(ph)), np.deg2rad(thr*np.sin(ph))
        z_ref = batoid.analysis.zernikeTransverseAberration(
            telescope, thx, thy, wavelength,
            nrad=20, naz=120, reference='chief',
            jmax=66, eps=0.61
        )

        z_ref *= wavelength

        z_terms = np.arange(4, 23)
        z_true = rng.uniform(-0.1, 0.1, size=19)*wavelength

        factory = danish.DonutFactory(
            R_outer=4.18, R_inner=2.5498,
            obsc_radii=obsc_radii, obsc_motion=obsc_motion,
            focal_length=10.31, pixel_scale=10e-6
        )

        fitter = SingleDonutModel(
            factory, z_ref=z_ref, z_terms=z_terms,
        )

        dx, dy = 0.0, 0.0
        fwhm = 0.7
        sky_level = 1000.0

        img = fitter.model(
            dx, dy, fwhm, z_true,
            thx=thx, thy=thy,
            sky_level=sky_level, flux=5e6
        )

        from scipy.optimize import least_squares
        guess = [0.0, 0.0, 0.7]+[0.0]*19
        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=2,
            args=(img, thx, thy, sky_level)
        )
        for i in range(4, 23):
            out = f"{i:2d}  {result.x[i-1]/wavelength:9.3f}"
            out += f"  {z_true[i-4]/wavelength:9.3f}"
            out += f"  {(result.x[i-1]-z_true[i-4])/wavelength:9.3f}"
            print(out)

        dx, dy, fwhm, *z_fit = result.x

        # import matplotlib.pyplot as plt
        # mod = fitter.model(
        #     dx, dy, fwhm, z_fit, thx, thy
        # )
        # fig, axes = plt.subplots(ncols=4, figsize=(10, 3))
        # a0 = axes[0].imshow(img/np.sum(img))
        # a1 = axes[1].imshow(mod/np.sum(mod))
        # a2 = axes[2].imshow(img/np.sum(img) - mod/np.sum(mod))
        # axes[3].axhline(0, c='k')
        # axes[3].plot(np.arange(4, 23), result.x[3:]/wavelength, c='b')
        # axes[3].plot(np.arange(4, 23), z_true/wavelength, c='k')
        # axes[3].plot(np.arange(4, 23), (result.x[3:]-z_true)/wavelength, c='r')
        # axes[3].set_xlabel("Zernike index")
        # axes[3].set_ylabel("Residual")
        # plt.colorbar(a0, ax=axes[0])
        # plt.colorbar(a1, ax=axes[1])
        # plt.colorbar(a2, ax=axes[2])
        # fig.tight_layout()
        # plt.show()

        np.testing.assert_allclose(dx, 0.0, rtol=0, atol=5e-2)
        np.testing.assert_allclose(dy, 0.0, rtol=0, atol=5e-2)
        np.testing.assert_allclose(fwhm, 0.7, rtol=0, atol=5e-2)
        np.testing.assert_allclose(z_fit, z_true, rtol=0, atol=0.05*wavelength)


if __name__ == "__main__":
    test_fitter()