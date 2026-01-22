import os
import galsim
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
    with open(
        os.path.join(directory, "..", "tests", "data", "test_kolm_donuts.pkl"),
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

        z_terms = np.arange(4, 23)
        fitter = danish.SingleDonutModel(
            factory, z_ref=z_ref*wavelength, z_terms=z_terms, thx=thx, thy=thy
        )
        guess = [np.sum(img), 0.0, 0.0, 0.7] + [0.0]*19
        lb = [-np.inf]*len(guess)
        ub = [np.inf]*len(guess)
        lb[0] = 0.0  # flux
        lb[3] = 0.1  # fwhm

        result = least_squares(
            fitter.chi, guess, jac=fitter.jac,
            ftol=1e-3, xtol=1e-3, gtol=1e-3,
            max_nfev=20, verbose=0,
            bounds=(lb, ub),
            args=(img, sky_level)
        )

        # # Try binning 2x2
        # binned_fitter = danish.SingleDonutModel(
        #     binned_factory, z_ref=z_ref*wavelength, z_terms=z_terms,
        #     thx=thx, thy=thy, npix=89
        # )

        # binned_img = img[:-1,:-1].reshape(90,2,90,2).mean(-1).mean(1)[:-1,:-1]
        # binned_result = least_squares(
        #     binned_fitter.chi, guess, jac=binned_fitter.jac,
        #     ftol=1e-3, xtol=1e-3, gtol=1e-3,
        #     max_nfev=20, verbose=0,
        #     bounds=(lb, ub),
        #     args=(binned_img, 4*sky_level)
        # )
    prof.disable()
    t1 = time.time()
    print(
        f"Time for fitter: {(t1-t0)/niter*1e3:.2f} ms"
    )
    # Print out the profile
    stats = pstats.Stats(prof)
    stats.sort_stats('time').print_stats(10)


if __name__ == "__main__":
    main()
