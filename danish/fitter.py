from functools import lru_cache

import numpy as np
import galsim
import cv2


class SingleDonutModel:
    """Model individual donuts using single Zernike offsets to a reference
    single Zernike series.

    Parameters
    ----------
    factory : DonutFactory
    z_ref : array of float
        Constant reference Zernike coefs to add to fitted coefficients.
        [0] is ignored, [1] is piston, [4] is defocus, etc.
    z_terms : sequence of int
        Which Zernike coefficients to include in the fit.  E.g.,
        [4,5,6,11] will fit defocus, astigmatism, and spherical.
    thx, thy : float
        Field angle in radians.
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
        thx=None, thy=None,
        N=90,
        seed=57721
    ):
        self.factory = factory
        # arcseconds per pixel
        # This is only used for the atmospheric part of the model, where we
        # ignore distortion and draw a circular profile in image coordinates.
        self.sky_scale = (
            3600*np.rad2deg(1/factory.focal_length)*factory.pixel_scale
        )
        self.z_ref = z_ref
        self.z_terms = z_terms
        self.thx = thx
        self.thy = thy
        self.N = N
        self.gsrng = galsim.BaseDeviate(seed)

    @lru_cache(maxsize=1000)
    def _atmKernel(self, dx, dy, fwhm):
        obj = galsim.Kolmogorov(fwhm=fwhm).shift(dx, dy)
        return obj.drawImage(nx=self.N, ny=self.N, scale=self.sky_scale).array

    @lru_cache(maxsize=1000)
    def _optImage(self, z_fit):
        aberrations = np.array(self.z_ref)
        for i, term in enumerate(self.z_terms):
            aberrations[term] += z_fit[i]
        return self.factory.image(
            aberrations=aberrations, thx=self.thx, thy=self.thy, N=self.N
        )

    def model(
        self,
        dx, dy, fwhm, z_fit,
        sky_level=None, flux=None
    ):
        """
        Parameters
        ----------
        dx, dy : float
            Offset in pixels(?)
        fwhm : float
            Full width half maximum of Kolmogorov kernel.
        z_fit : sequence of float
            Zernike perturbations.
        sky_level : float
            Sky level to use when adding Poisson noise to image.
        flux : float
            Flux level at which to set image.

        Returns
        -------
        img : array of float
            Model image.
        """
        atm = self._atmKernel(dx, dy, fwhm)
        opt = self._optImage(tuple(z_fit))
        arr = cv2.filter2D(opt, -1, atm, borderType=cv2.BORDER_CONSTANT)
        img = galsim.Image(arr)  # Does this make a copy?
        if flux is not None:
            img.array[:] *= flux/np.sum(img.array)
        if sky_level is not None:
            pn = galsim.PoissonNoise(self.gsrng, sky_level=sky_level)
            img.addNoise(pn)
        return img.array

    def chi(
        self, params, data, sky_level
    ):
        """
        Parameters
        ----------
        params : sequence of float
            Order is: (dx, dy, fwhm, *z_fit)
        data : array of float
            Image against which to compute chi.
        sky_level : float
            Sky level to use when computing pixel errors.

        Returns
        -------
        chi : array of float
            Flattened array of chi residuals.
        """
        dx, dy, fwhm, *z_fit = params
        mod = self.model(dx, dy, fwhm, z_fit)
        mod *= np.sum(data)/np.sum(mod)
        _chi = ((data-mod)/np.sqrt(sky_level + mod)).ravel()
        return _chi

    def jac(
        self, params, data, sky_level
    ):
        """
        Parameters
        ----------
        params : sequence of float
            Order is: (dx, dy, fwhm, *z_fit)
        data : array of float
            Image against which to compute chi.
        sky_level : float
            Sky level to use when computing pixel errors.

        Returns
        -------
        jac : array of float
            Jacobian array d(chi)/d(param).  First index is chi, second index is
            param.
        """
        NN = 2*self.N+1
        out = np.zeros((NN**2, len(params)))
        chi0 = self.chi(params, data, sky_level)

        step = [0.01, 0.01, 0.01]+[1e-8]*len(self.z_terms)
        for i in range(len(params)):
            params1 = np.array(params)
            params1[i] += step[i]
            chi1 = self.chi(params1, data, sky_level)
            out[:, i] = (chi1-chi0)/step[i]
        return out
