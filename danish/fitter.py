from functools import lru_cache

import numpy as np
import galsim
import cv2


class SingleDonutModel:
    """
    Parameters
    ----------
    factory : DonutFactory
    z_ref : array of float
        Constant reference Zernike coefs to add to fitted coefficients.
        [0] is ignored, [1] is piston, [4] is defocus, etc.
    z_terms : sequence of int
        Which Zernike coefficients to include in the fit.  E.g.,
        [4,5,6,11] will fit defocus, astigmatism, and spherical.
    N : int
        Size of image
    seed : int
        Random seed for use when creating noisy donut images with this class.
    """
    def __init__(
        self,
        factory,
        z_ref=None,
        z_terms=(),
        N=90,
        seed=57721
    ):
        self.factory = factory
        # arcseconds per pixel
        self.skyscale = 3600*np.rad2deg(1/factory.focal_length)*factory.pixel_scale
        self.z_ref = z_ref
        self.z_terms = z_terms
        self.N = N
        self.gsrng = galsim.BaseDeviate(seed)

    @lru_cache(maxsize=1000)
    def _atmKernel(self, dx, dy, fwhm):
        fwhm = np.clip(fwhm, 0.1, 2.0)
        obj = galsim.Kolmogorov(fwhm=fwhm).shift(dx, dy)
        return obj.drawImage(nx=self.N, ny=self.N, scale=self.skyscale).array

    @lru_cache(maxsize=1000)
    def _optImage(self, z_fit, thx, thy):
        aberrations = np.array(self.z_ref)
        for i, term in enumerate(self.z_terms):
            aberrations[term] += z_fit[i]
        return self.factory.image(
            aberrations=aberrations, thx=thx, thy=thy, N=self.N
        )

    def model(
        self,
        dx, dy, fwhm, z_fit,
        thx, thy,
        sky_level=None, flux=None
    ):
        atm = self._atmKernel(dx, dy, fwhm)
        opt = self._optImage(tuple(z_fit), thx, thy)
        arr = cv2.filter2D(opt, -1, atm, borderType=cv2.BORDER_CONSTANT)
        img = galsim.Image(arr)  # Does this make a copy?
        if flux is not None:
            img.array[:] *= flux/np.sum(img.array)
        if sky_level is not None:
            pn = galsim.PoissonNoise(self.gsrng, sky_level=sky_level)
            img.addNoise(pn)
        return img.array

    def chi(
        self, params, data, thx, thy, sky_level
    ):
        dx, dy, fwhm, *z_fit = params
        mod = self.model(dx, dy, fwhm, z_fit, thx, thy)
        mod *= np.sum(data)/np.sum(mod)
        _chi = ((data-mod)/np.sqrt(sky_level + mod)).ravel()
        return _chi

    def jac(
        self, params, data, thx, thy, sky_level
    ):
        NN = 2*self.N+1
        out = np.zeros((NN**2, len(params)))
        chi0 = self.chi(params, data, thx, thy, sky_level)

        step = [0.01, 0.01, 0.01]+[1e-8]*len(self.z_terms)
        for i, p in enumerate(params):
            params1 = np.array(params)
            params1[i] += step[i]
            chi1 = self.chi(params1, data, thx, thy, sky_level)
            out[:, i] = (chi1-chi0)/step[i]
        return out
