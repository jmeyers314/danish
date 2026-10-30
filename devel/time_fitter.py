import argparse
import os
import galsim
from functools import partial
from cProfile import Profile
import pstats
import time
import pickle
import numpy as np
from scipy.optimize import least_squares
import batoid
import danish
import yaml

directory = os.path.dirname(__file__)
Rubin_obsc = yaml.safe_load(open(os.path.join(danish.datadir, 'RubinObsc.yaml')))
# del Rubin_obsc["Spider_3D"]

def main():
    """Roundtrip using GalSim Kolmogorov atmosphere + batoid to produce test
    image of AOS DOF perturbed optics.  Model and fitter run independent code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use-aoi', action='store_true',
        help="Apply AOI-dependent throughput correction (bandpass_filter='r', default Tbb and airmass)"
    )
    parser.add_argument(
        '--factory', choices=('inverse', 'triangle'), default='inverse',
        help="Rendering strategy: inverse (focal-to-pupil) or triangle "
             "(forward mesh accumulation).  Only triangle can supply an "
             "analytic jacobian."
    )
    parser.add_argument(
        '--nrad', type=int, default=15,
        help="Radial rings in the triangle mesh.  Ignored for --factory inverse."
    )
    parser.add_argument(
        '--jac', choices=('analytic', 'fd'), default='analytic',
        help="Differentiation method for the Zernike columns.  'analytic' "
             "falls back to 'fd' when the factory cannot support it."
    )
    parser.add_argument(
        '--binned', action='store_true',
        help="Time the 2x2-binned fit (20 um pixels, 89x89) instead of the "
             "native-resolution one."
    )
    args = parser.parse_args()

    bandpass_filter = 'r' if args.use_aoi else None

    def make_factory(pixel_scale):
        """Build the selected factory, passing only kwargs it accepts."""
        kwargs = dict(
            R_outer=4.18, R_inner=2.5498,
            mask_params=Rubin_obsc,
            focal_length=10.31, pixel_scale=pixel_scale,
            bandpass_filter=bandpass_filter,
        )
        if args.factory == 'triangle':
            return danish.DonutTriangleFactory(nrad=args.nrad, **kwargs)
        return danish.DonutInverseFactory(**kwargs)

    with open(
        os.path.join(directory, "..", "tests", "data", "test_kolm_donuts.pkl"),
        'rb'
    ) as f:
        data = pickle.load(f)
    sky_level = data[0]['sky_level']
    wavelength = data[0]['wavelength']
    fwhm = data[0]['fwhm']

    # Binning 2x2 doubles the pixel scale and quarters the pixel count, so the
    # sky level per pixel goes up by the same factor the area does.
    if args.binned:
        factory = make_factory(20e-6)
        npix = 89
        sky_level *= 4
    else:
        factory = make_factory(10e-6)
        npix = 181

    print(f"AOI dependence: {'ON (r-band, default Tbb/airmass)' if bandpass_filter else 'OFF'}")
    factory_desc = args.factory
    if args.factory == 'triangle':
        factory_desc += f" (nrad={args.nrad})"
    print(f"Factory: {factory_desc}")
    print(f"Sampling: {'2x2 binned' if args.binned else 'native'} ({npix}x{npix})")

    # Whether the requested method is the one that will actually run.  A
    # factory with no mesh falls back to differencing silently, and a silent
    # fallback timed as 'analytic' is a misreported measurement.
    jac_honored = None

    # start a profile
    t0 = time.time()
    prof = Profile(builtins=False)
    prof.enable()
    niter = 10
    for datum in data[1:niter+1]:
        thx = datum['thx']
        thy = datum['thy']
        z_ref = datum['z_ref']
        z_actual = datum['z_actual']
        img = datum['arr']

        if args.binned:
            # 181 -> 90 pairs -> 89, matching the fitter's odd-npix rule.
            # Sum rather than average: the loss reads model values as counts,
            # and the sky variance below is scaled by the same factor.
            img = img[:-1, :-1].reshape(90, 2, 90, 2).sum(-1).sum(1)[:-1, :-1]

        z_terms = np.arange(4, 23)
        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref*wavelength, z_terms=z_terms, thx=thx, thy=thy,
            npix=npix,
        )
        guess = [np.sum(img), 0.0, 0.0, 0.7] + [0.0]*19
        lb = [-np.inf]*len(guess)
        ub = [np.inf]*len(guess)
        lb[0] = 0.0  # flux
        lb[3] = 0.1  # fwhm

        if jac_honored is None:
            jac_honored = (
                args.jac == 'fd' or fitter._can_use_analytic()
            )

        result = least_squares(
            fitter.chi, guess,
            jac=partial(fitter.jac, method=args.jac),
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=0,
            bounds=(lb, ub),
            args=(img, sky_level)
        )
    prof.disable()
    t1 = time.time()
    jac_desc = args.jac
    if not jac_honored:
        jac_desc += f" (requested, fell back to 'fd': {args.factory} factory "
        jac_desc += "renders without a mesh)"
    print(f"Jacobian: {jac_desc}")
    print(
        f"Time for fitter: {(t1-t0)/niter*1e3:.2f} ms"
    )
    # Print out the profile
    stats = pstats.Stats(prof)
    stats.sort_stats('time').print_stats(10)


if __name__ == "__main__":
    main()
