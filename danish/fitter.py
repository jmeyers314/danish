# Copyright (c) 2021, Lawrence Livermore National Laboratory, LLC.
# All rights reserved.
# LLNL-CODE-826307

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from functools import lru_cache

import numpy as np
import galsim

try:
    import cv2
except ImportError:
    try:
        from scipy.signal import fftconvolve
    except ImportError:
        raise ValueError("Either python-opencv or scipy must be installed")
    else:
        def convolve(A, B):
            return fftconvolve(A, B, mode='same')
else:
    def convolve(A, B):
        return cv2.filter2D(A, -1, B[::-1,::-1], borderType=cv2.BORDER_CONSTANT)


class BaseDonutModel:
    """Base class for donut models.

    Parameters
    ----------
    factory : DonutFactory
    npix : int
        Number of pixels along image edge. Must be odd.
    seed : int
        Random seed for use when creating noisy donut images.
    """
    def __init__(self, factory, npix, seed):
        assert npix % 2 == 1, "npix must be odd"

        self.factory = factory
        self.npix = npix
        self.no2 = (npix-1)//2
        self.gsrng = galsim.BaseDeviate(seed)
        # arcseconds per pixel
        # This is only used for the atmospheric part of the model, where we
        # ignore distortion and draw a circular profile in image coordinates.
        self.sky_scale = (
            3600*np.rad2deg(1/factory.focal_length)*factory.pixel_scale
        )

    @lru_cache(maxsize=1000)
    def _atm_kernel(self, dx, dy, fwhm):
        """Compute atmospheric kernel.

        Parameters
        ----------
        dx, dy : float
            Offset in pixels.
        fwhm : float
            Full width half maximum of Kolmogorov kernel.

        Returns
        -------
        array of float
            Atmospheric kernel array.
        """
        obj = galsim.Kolmogorov(fwhm=fwhm).shift(dx, dy)
        img = obj.drawImage(nx=self.no2, ny=self.no2, scale=self.sky_scale)
        return img.array

    def _assemble_image(self, opt, atm, flux=None, sky_level=None):
        """Apply convolution and add optional noise.

        Parameters
        ----------
        opt : array of float
            Optical model array.
        atm : array of float
            Atmospheric kernel array.
        flux : float, optional
            Flux level at which to set image.
        sky_level : float, optional
            Sky level to use when adding Poisson noise.

        Returns
        -------
        array of float
            Convolved and potentially noisy image array.
        """
        arr = convolve(opt, atm)
        img = galsim.Image(arr)
        if flux is not None:
            img.array[:] *= flux/np.sum(img.array)
        if sky_level is not None:
            pn = galsim.PoissonNoise(self.gsrng, sky_level=sky_level)
            img.addNoise(pn)
        return img.array

    def _chi_single(self, model, datum, var):
        """Compute chi for a single image.

        Parameters
        ----------
        model : array of float
            Model image.
        datum : array of float
            Data image.
        var : float or array of float
            Variance of the sky only.

        Returns
        -------
        array of float
            Flattened chi residuals.
        """
        model = model.copy()
        model *= np.sum(datum)/np.sum(model)
        return ((datum-model)/np.sqrt(var+model)).ravel()


class SingleDonutModel(BaseDonutModel):
    """Model individual donuts using single Zernike offsets to a reference
    single Zernike series.

    Parameters
    ----------
    factory : DonutFactory
    z_ref : array of float
        Constant reference Zernike coefs to add to fitted coefficients.
        [0] is ignored, [1] is piston, [4] is defocus, etc.
    x_offset, y_offset : galsim.zernike.Zernike, optional
        Additional focal plane offsets (in meters) represented as Zernike
        series.
    z_terms : sequence of int
        Which Zernike coefficients to include in the fit.  E.g.,
        [4,5,6,11] will fit defocus, astigmatism, and spherical.
    thx, thy : float
        Field angle in radians.
    npix : int
        Number of pixels along image edge.  Must be odd.
    seed : int
        Random seed for use when creating noisy donut images with this class.
    """
    def __init__(
        self,
        factory, *,
        z_ref=None,
        x_offset=None, y_offset=None,
        z_terms=(),
        thx=None, thy=None,
        npix=181,
        seed=57721
    ):
        super().__init__(factory, npix, seed)
        self.z_ref = z_ref
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_terms = z_terms
        self.thx = thx
        self.thy = thy

    @lru_cache(maxsize=1000)
    def _opt_image(self, z_fit):
        aberrations = np.array(self.z_ref)
        for i, term in enumerate(self.z_terms):
            aberrations[term] += z_fit[i]
        return self.factory.image(
            aberrations=aberrations,
            x_offset=self.x_offset, y_offset=self.y_offset,
            thx=self.thx, thy=self.thy,
            npix=self.npix
        )

    def model(
        self,
        dx, dy, fwhm, z_fit, *,
        sky_level=None, flux=None
    ):
        """Compute donut model image.

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
        atm = self._atm_kernel(dx, dy, fwhm)
        opt = self._opt_image(tuple(z_fit))
        return self._assemble_image(opt, atm, flux, sky_level)

    def chi(
        self, params, data, var
    ):
        """Compute chi = (data - model)/error.

        The error is modeled as sqrt(model + var).

        Parameters
        ----------
        params : sequence of float
            Order is: (dx, dy, fwhm, *z_fit)
        data : array of float (npix, npix)
            Image against which to compute chi.
        var : float or array of float (npix, npix)
            Variance of the sky only.  Do not include Poisson contribution of
            the signal, as this will be added from the current model.

        Returns
        -------
        chi : array of float
            Flattened array of chi residuals.
        """
        dx, dy, fwhm, *z_fit = params
        mod = self.model(dx, dy, fwhm, z_fit)
        return self._chi_single(mod, data, var)

    def jac(
        self, params, data, var
    ):
        """Compute jacobian d(chi)/d(param).

        Parameters
        ----------
        params : sequence of float
            Order is: (dx, dy, fwhm, *z_fit)
        data : array of float
            Image against which to compute chi.
        var : float or array of float (npix, npix)
            Variance of the sky only.  Do not include Poisson contribution of
            the signal, as this will be added from the current model.

        Returns
        -------
        jac : array of float
            Jacobian array d(chi)/d(param).  First dimenison is chi, second
            dimension is param.
        """
        out = np.zeros((self.npix**2, len(params)))
        chi0 = self.chi(params, data, var)

        step = [0.01, 0.01, 0.01]+[1e-9]*len(self.z_terms)
        for i in range(len(params)):
            params1 = np.array(params)
            params1[i] += step[i]
            chi1 = self.chi(params1, data, var)
            out[:, i] = (chi1-chi0)/step[i]
        return out


class DoubleZernike:
    """Double Zernike model for including both pupil and field dependence of
    wavefront.

    Parameters
    ----------
    coefs : array of float
        Double Zernike coefficients.
    field_radius : float
        Outer field radius to use to normalize coefficients.
    """
    def __init__(self, coefs, *, field_radius=1.0):
        self.coefs = coefs
        self.field_radius = field_radius

        self.Zs = [
            galsim.zernike.Zernike(coef, R_inner=0.0, R_outer=field_radius)
            for coef in coefs.T
        ]

    def __call__(self, thx, thy):
        """Return pupil Zernike coefficients.

        Parameters
        ----------
        thx, thy : float
            Field angles in radians.
        """
        return np.array([Z(thx, thy) for Z in self.Zs])

    def __add__(self, rhs):
        assert self.field_radius == rhs.field_radius
        return DoubleZernike(
            self.coefs+rhs.coefs,
            field_radius=self.field_radius
        )

    def __mul__(self, rhs):
        return DoubleZernike(
            self.coefs*rhs,
            field_radius=self.field_radius
        )

    @property
    def jmax(self):
        """Maximum Noll index for pupil dimension."""
        return self.coefs.shape[1]-1

    @property
    def kmax(self):
        """Maximum Noll index for field dimension."""
        return self.coefs.shape[0]-1


class MultiDonutModel(BaseDonutModel):
    """Model double Zernikes on top of reference per-donut single Zernikes to
    multiple donuts simultaneously.

    Parameters
    ----------
    factory : DonutFactory
    dz_ref : DoubleZernike
        Double Zernike coefficients to use for constructing Single Zernike
        reference coefficients to use for each modeled donut.  Either this kwarg
        or `z_refs` must be set.
    z_refs : array of float
        Single Zernike reference coefficients for each donut.  First dimension
        is donut, second dimension is pupil Zernike coefficient.
    field_radius : float
        Field radius in radians.  Ignored if dz_ref is provided.
    dz_terms : sequence of (int, int)
        Which double Zernike coefficients to include in the fit.
    thxs, thys : float
        Field angles in radians.
    npix : int
        Number of pixels along image edge.  Must be odd.
    seed : int
        Random seed for use when creating noisy donut images with this class.
    """
    def __init__(
        self,
        factory, *,
        dz_ref=None,
        z_refs=None,
        field_radius=None,
        dz_terms=(),
        thxs=None, thys=None,
        npix=181,
        seed=577215
    ):
        super().__init__(factory, npix, seed)

        if dz_ref is None and z_refs is None:
            raise ValueError("Must provide dz_ref or z_refs")
        if dz_ref is not None and z_refs is not None:
            raise ValueError("Cannot provide both dz_ref and z_refs")
        if z_refs is None:
            z_refs = dz_ref(thxs, thys)
        self.z_refs = z_refs
        if field_radius is None:
            if dz_ref is None:
                raise ValueError("Must provide dz_ref or field_radius")
            field_radius = dz_ref.field_radius
        self.field_radius = field_radius
        self.dz_terms = dz_terms
        self.thxs = thxs
        self.thys = thys
        self.nstar = len(thxs)
        self.jmax_fit = max((dz_term[1] for dz_term in dz_terms), default=0)
        self.kmax_fit = max((dz_term[0] for dz_term in dz_terms), default=0)

    @lru_cache(maxsize=1000)
    def _opt1(self, aberrations, thx, thy):
        return self.factory.image(
            aberrations=tuple(aberrations), thx=thx, thy=thy, npix=self.npix
        )

    def _model1(
        self,
        dx, dy, fwhm, aberrations,
        thx, thy, *,
        sky_level=None, flux=None
    ):
        atm = self._atm_kernel(dx, dy, fwhm)
        opt = self._opt1(tuple(aberrations), thx, thy)
        return self._assemble_image(opt, atm, flux, sky_level)

    def _dz(self, dz_fit):
        dzarr = np.zeros((self.kmax_fit+1, self.jmax_fit+1))
        for i, zterm in enumerate(self.dz_terms):
            dzarr[zterm] = dz_fit[i]
        dz = DoubleZernike(dzarr, field_radius=self.field_radius)
        return dz

    def model(
        self, dxs, dys, fwhm, dz_fit, *, sky_levels=None, fluxes=None
    ):
        """Compute model for all donuts.

        Parameters
        ----------
        dxs, dys : float
            Offsets in pixels(?)
        fwhm : float
            Full width half maximum of Kolmogorov kernel.
        dz_fit : sequence of float
            Double Zernike perturbations.
        sky_levels : sequence of float
            Sky levels to use when adding Poisson noise to images.
        fluxes : sequence of float
            Flux levels at which to set images.

        Returns
        -------
        imgs : array of float.  Shape: (nstar, npix, npix)
            Model images.
        """
        nstar = self.nstar
        npix = self.npix
        if sky_levels is None:
            sky_levels = [None]*nstar
        if fluxes is None:
            fluxes = [None]*nstar

        out = np.empty((nstar, npix, npix))
        dz = self._dz(dz_fit)

        for i, (thx, thy) in enumerate(zip(self.thxs, self.thys)):
            aberrations = np.array(self.z_refs[i])
            z_fit = dz(thx, thy)
            aberrations[:len(z_fit)] += z_fit
            out[i] = self._model1(
                dxs[i], dys[i],
                fwhm,
                aberrations,
                thx, thy,
                sky_level=sky_levels[i], flux=fluxes[i]
            )
        return out

    def unpack_params(self, params):
        """Utility method to unpack params list

        Parameters
        ----------
        params : list
            Parameter list to unpack

        Returns
        -------
        dxs, dys : list
            Model position offsets.
        fwhm : float
            Model FWHM.
        dz_fit : list
            Model double Zernike coefficients.
        """
        dxs = params[:self.nstar]
        dys = params[self.nstar:2*self.nstar]
        fwhm = params[2*self.nstar]
        dz_fit = params[2*self.nstar+1:]
        return dxs, dys, fwhm, dz_fit

    def chi(
        self, params, data, vars
    ):
        """Compute chi = (data - model)/error.

        The error is modeled as sqrt(model + var) for each donut.

        Parameters
        ----------
        params : sequence of float
            Order is: (dx, dy, fwhm, *z_fit)
        data : array of float.  Shape: (nstar, npix, npix)
            Images against which to compute chi.
        vars : sequence of array (npix, npix) or sequence of float
            Variances of the sky only.  Do not include Poisson contribution of
            the signal, as this will be added from the current model.

        Returns
        -------
        chi : array of float
            Flattened array of chi residuals.
        """
        dxs, dys, fwhm, dz_fit = self.unpack_params(params)
        mods = self.model(dxs, dys, fwhm, dz_fit)
        chis = np.empty((self.nstar, self.npix, self.npix))
        for i, (mod, datum) in enumerate(zip(mods, data)):
            chis[i] = self._chi_single(mod, datum, vars[i]).reshape(self.npix, self.npix)
        return chis.ravel()

    def _chi1(self, dx, dy, fwhm, aberrations, thx, thy, datum, var):
        mod1 = self._model1(dx, dy, fwhm, aberrations, thx, thy)
        return self._chi_single(mod1, datum, var)

    def jac(
        self, params, data, vars
    ):
        """Compute jacobian d(chi)/d(param).

        Parameters
        ----------
        params : sequence of float
            Order is: (dx, dy, fwhm, *z_fit)
        data : array of float.  Shape: (nstar, npix, npix)
            Image against which to compute chi.
        vars : sequence of array (npix, npix) or sequence of float
            Variances of the sky only.  Do not include Poisson contribution of
            the signal, as this will be added from the current model.

        Returns
        -------
        jac : array of float
            Jacobian array d(chi)/d(param).  First dimenison is chi, second
            dimension is param.
        """
        nstar = self.nstar
        npix = self.npix

        out = np.zeros((nstar*npix**2, len(params)))
        dxs, dys, fwhm, dz_fit = self.unpack_params(params)
        dz = self._dz(dz_fit)

        chi0 = np.zeros(nstar*npix**2)

        # Star dx, dy terms are sparse
        for i in range(self.nstar):
            thx, thy = self.thxs[i], self.thys[i]
            aberrations = np.array(self.z_refs[i])
            z_fit = dz(thx, thy)
            aberrations[:len(z_fit)] += z_fit
            s = slice(i*npix**2, (i+1)*npix**2)

            c0 = self._chi1(
                dxs[i], dys[i], fwhm,
                aberrations,
                thx, thy, data[i], vars[i]
            )
            cx = self._chi1(
                dxs[i]+0.01, dys[i], fwhm,
                aberrations,
                thx, thy, data[i], vars[i]
            )
            cy = self._chi1(
                dxs[i], dys[i]+0.01, fwhm,
                aberrations,
                thx, thy, data[i], vars[i]
            )

            out[s, i] = (cx-c0)/0.01
            out[s, i+nstar] = (cy-c0)/0.01
            chi0[s] = c0

        # FWHM
        params1 = np.array(params)
        params1[2*nstar] += 0.001
        chi1 = self.chi(params1, data, vars)
        out[:, 2*nstar] = (chi1-chi0)/0.001

        # DZ terms
        for i in range(2*nstar+1, len(params)):
            params1 = np.array(params)
            params1[i] += 1e-8
            chi1 = self.chi(params1, data, vars)
            out[:, i] = (chi1-chi0)/1e-8

        return out


class VModeDonutModel(BaseDonutModel):
    """ Model multiple donuts using modes of double Zernike coefficients.

    Parameters
    ----------
    factory : DonutFactory
    dz_ref : DoubleZernike
        Double Zernike coefficients to use for constructing Single Zernike
        reference coefficients to use for each modeled donut.  Either this kwarg or `z_refs` must be set.
    z_refs : array of float
        Single Zernike reference coefficients for each donut.  First dimension
        is donut, second dimension is pupil Zernike coefficient.
    field_radius : float
        Field radius in radians.  Ignored if dz_ref is provided.
    sensitivity : array of float
        Matrix of shape (nmode, kmax+1, jmax+1) giving the linear combination of
        double Zernike coefficients for each mode.
    thxs, thys : float
        Field angles in radians.
    npix : int
        Number of pixels along image edge.  Must be odd.
    seed : int
        Random seed for use when creating noisy donut images with this class.
    """
    def __init__(
        self,
        factory, *,
        dz_ref=None,
        z_refs=None,
        field_radius=None,
        sensitivity=None,
        thxs=None, thys=None,
        npix=181,
        seed=5772156
    ):
        if sensitivity is None:
            raise ValueError("Must provide sensitivity")
        self.sensitivity = sensitivity
        self.nmode = sensitivity.shape[0]
        self.kmax = sensitivity.shape[1]-1
        self.jmax = sensitivity.shape[2]-1

        super().__init__(factory, npix, seed)

        if dz_ref is None and z_refs is None:
            raise ValueError("Must provide dz_ref or z_refs")
        if dz_ref is not None and z_refs is not None:
            raise ValueError("Cannot provide both dz_ref and z_refs")
        if z_refs is None:
            z_refs = dz_ref(thxs, thys)
        self.z_refs = z_refs
        if field_radius is None:
            if dz_ref is None:
                raise ValueError("Must provide dz_ref or field_radius")
            field_radius = dz_ref.field_radius
        self.field_radius = field_radius
        self.thxs = thxs
        self.thys = thys
        self.nstar = len(thxs)

    @lru_cache(maxsize=1000)
    def _opt1(self, aberrations, thx, thy):
        return self.factory.image(
            aberrations=tuple(aberrations), thx=thx, thy=thy, npix=self.npix
        )

    def _model1(
        self,
        dx, dy, fwhm, aberrations,
        thx, thy, *,
        sky_level=None, flux=None
    ):
        atm = self._atm_kernel(dx, dy, fwhm)
        opt = self._opt1(tuple(aberrations), thx, thy)
        return self._assemble_image(opt, atm, flux, sky_level)

    def model(
        self, dxs, dys, fwhm, mode_fit, *, sky_levels=None, fluxes=None
    ):
        """Compute model for all donuts.

        Parameters
        ----------
        dxs, dys : float
            Offsets in pixels(?)
        fwhm : float
            Full width half maximum of Kolmogorov kernel.
        mode_fit : sequence of float
            Mode perturbations.
        sky_levels : sequence of float
            Sky levels to use when adding Poisson noise to images.
        fluxes : sequence of float
            Flux levels at which to set images.

        Returns
        -------
        imgs : array of float.  Shape: (nstar, npix, npix)
            Model images.
        """
        nstar = self.nstar
        npix = self.npix
        if sky_levels is None:
            sky_levels = [None]*nstar
        if fluxes is None:
            fluxes = [None]*nstar

        out = np.empty((nstar, npix, npix))
        dzarr = np.einsum(
            "i,ikj->kj",
            mode_fit,
            self.sensitivity
        )
        dz = DoubleZernike(dzarr, field_radius=self.field_radius)

        for i, (thx, thy) in enumerate(zip(self.thxs, self.thys)):
            aberrations = np.array(self.z_refs[i])
            z_fit = dz(thx, thy)
            aberrations[:len(z_fit)] += z_fit
            out[i] = self._model1(
                dxs[i], dys[i],
                fwhm,
                aberrations,
                thx, thy,
                sky_level=sky_levels[i], flux=fluxes[i]
            )
        return out

    def unpack_params(self, params):
        """Utility method to unpack params list

        Parameters
        ----------
        params : list
            Parameter list to unpack

        Returns
        -------
        dxs, dys : list
            Model position offsets.
        fwhm : float
            Model FWHM.
        mode_fit : list
            Model mode coefficients.
        """
        dxs = params[:self.nstar]
        dys = params[self.nstar:2*self.nstar]
        fwhm = params[2*self.nstar]
        mode_fit = params[2*self.nstar+1:]
        return dxs, dys, fwhm, mode_fit

    def chi(
        self, params, data, vars
    ):
        """Compute chi = (data - model)/error.

        The error is modeled as sqrt(model + var) for each donut.

        Parameters
        ----------
        params : sequence of float
            Order is: (dx, dy, fwhm, *mode_fit)
        data : array of float.  Shape: (nstar, npix, npix)
            Images against which to compute chi.
        vars : sequence of array (npix, npix) or sequence of float
            Variances of the sky only.  Do not include Poisson contribution of
            the signal, as this will be added from the current model.

        Returns
        -------
        chi : array of float
            Flattened array of chi residuals.
        """
        dxs, dys, fwhm, mode_fit = self.unpack_params(params)
        mods = self.model(dxs, dys, fwhm, mode_fit)
        chis = np.empty((self.nstar, self.npix, self.npix))
        for i, (mod, datum) in enumerate(zip(mods, data)):
            chis[i] = self._chi_single(mod, datum, vars[i]).reshape(self.npix, self.npix)
        return chis.ravel()

    def _chi1(self, dx, dy, fwhm, aberrations, thx, thy, datum, var):
        mod1 = self._model1(dx, dy, fwhm, aberrations, thx, thy)
        return self._chi_single(mod1, datum, var)

    def jac(
        self, params, data, vars
    ):
        """Compute jacobian d(chi)/d(param).

        Parameters
        ----------
        params : sequence of float
            Order is: (dx, dy, fwhm, *mode_fit)
        data : array of float.  Shape: (nstar, npix, npix)
            Image against which to compute chi.
        vars : sequence of array (npix, npix) or sequence of float
            Variances of the sky only.  Do not include Poisson contribution of
            the signal, as this will be added from the current model.

        Returns
        -------
        jac : array of float
            Jacobian array d(chi)/d(param).  First dimenison is chi, second
            dimension is param.
        """
        nstar = self.nstar
        npix = self.npix

        out = np.zeros((nstar*npix**2, len(params)))
        dxs, dys, fwhm, mode_fit = self.unpack_params(params)

        dzarr = np.einsum(
            "i,ikj->kj",
            mode_fit,
            self.sensitivity
        )
        dz = DoubleZernike(dzarr, field_radius=self.field_radius)

        chi0 = np.zeros(nstar*npix**2)

        # Star dx, dy terms are sparse
        for i in range(self.nstar):
            thx, thy = self.thxs[i], self.thys[i]
            aberrations = np.array(self.z_refs[i])
            z_fit = dz(thx, thy)
            aberrations[:len(z_fit)] += z_fit
            s = slice(i*npix**2, (i+1)*npix**2)

            c0 = self._chi1(
                dxs[i], dys[i], fwhm,
                aberrations,
                thx, thy, data[i], vars[i]
            )
            cx = self._chi1(
                dxs[i]+0.01, dys[i], fwhm,
                aberrations,
                thx, thy, data[i], vars[i]
            )
            cy = self._chi1(
                dxs[i], dys[i]+0.01, fwhm,
                aberrations,
                thx, thy, data[i], vars[i]
            )

            out[s, i] = (cx-c0)/0.01
            out[s, i+nstar] = (cy-c0)/0.01
            chi0[s] = c0

        # FWHM
        params1 = np.array(params)
        params1[2*nstar] += 0.001
        chi1 = self.chi(params1, data, vars)
        out[:, 2*nstar] = (chi1-chi0)/0.001

        # Mode terms
        for i in range(2*nstar+1, len(params)):
            params1 = np.array(params)
            params1[i] += 1e-8
            chi1 = self.chi(params1, data, vars)
            out[:, i] = (chi1-chi0)/1e-8

        return out
