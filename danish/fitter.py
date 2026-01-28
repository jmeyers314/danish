# Copyright (c) 2021-2026, Lawrence Livermore National Laboratory, LLC.
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
    bkg_order : int
        Order of polynomial background model to use.  If -1, no background.
    npix : int
        Number of pixels along image edge. Must be odd.
    seed : int
        Random seed for use when creating noisy donut images.
    """
    def __init__(self, factory, npix, seed, bkg_order):
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
        self.bkg_order = bkg_order
        self.nbkg = (bkg_order+1)*(bkg_order+2)//2

    @lru_cache(maxsize=1000)
    def _atm_kernel(self, dx, dy, fwhm):
        """Compute atmospheric kernel.

        Parameters
        ----------
        dx, dy : float
            Offset in arcseconds.
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

    @lru_cache(maxsize=1000)
    def _opt_kernel(
        self,
        zk,
        thx, thy,
        x_offset=None, y_offset=None,
    ):
        """Compute optical kernel.

        Parameters
        ----------
        zk : array of float
            Zernike coefficients.
        thx, thy : float
            Field angle in radians.
        x_offset, y_offset : galsim.zernike.Zernike, optional
            Additional pupil distortion coefficients.

        Returns
        -------
        array of float
            Optical kernel array.
        """
        result = self.factory.image(
            aberrations=zk,
            x_offset=x_offset, y_offset=y_offset,
            thx=thx, thy=thy,
            npix=self.npix
        )
        return result

    @lru_cache(maxsize=1000)
    def _bkg(
        self,
        bkg
    ):
        """Compute background model.

        Parameters
        ----------
        bkg : array of float
            Background polynomial coefficients.

        Returns
        -------
        array of float
            Background model array.
        """
        no2 = self.no2
        zkbkg = galsim.zernike.Zernike([0]+list(bkg), R_outer=no2)
        result = zkbkg(*np.mgrid[-no2:no2+1, -no2:no2+1][::-1])
        return result

    @lru_cache(maxsize=100)
    def _model(
        self,
        flux, dx, dy, fwhm, zk, bkg,
        thx, thy,
        x_offset=None, y_offset=None,
        sky_level=None
    ):
        """Compute model image.

        Parameters
        ----------
        flux : float
            Flux level at which to set image.
        dx, dy : float
            Offset in arcseconds.
        fwhm : float
            Full width half maximum of Kolmogorov kernel.
        zk : array of float
            Zernike coefficients.
        bkg : array of float
            Background polynomial coefficients.
        thx, thy : float
            Field angle in radians.
        x_offset, y_offset : galsim.zernike.Zernike, optional
            Additional pupil distortion coefficients.
        sky_level : float, optional
            Sky level to use when adding Poisson noise to image.

        Returns
        -------
        array of float
            Model image array.
        """
        atm = self._atm_kernel(dx, dy, fwhm)
        opt = self._opt_kernel(zk, thx, thy, x_offset=x_offset, y_offset=y_offset)
        arr = convolve(opt, atm)
        arr *= flux/np.sum(arr)

        if bkg:
            arr += self._bkg(bkg)

        if sky_level is not None:
            pn = galsim.PoissonNoise(self.gsrng, sky_level=sky_level)
            img = galsim.Image(arr)
            img.addNoise(pn)
            arr = img.array

        return arr

    def _chi(
        self,
        data,
        model,
        var
    ):
        """Compute chi = (data - model)/error.

        Parameters
        ----------
        data : array of float
            Observed image data.
        model : array of float
            Model image data.
        var : array of float
            Variance of the observed data.

        Returns
        -------
        array of float
            Chi values for each pixel.
        """
        result = ((data-model)/np.sqrt(var+model)).ravel()
        return result


class SingleDonutModel(BaseDonutModel):
    """Model individual donuts using single Zernike offsets to a reference
    single Zernike series.

    Parameters
    ----------
    factory : DonutFactory
    bkg_order : int, optional
        Order of polynomial background model to use.  If -1, no background.
    z_ref : array of float
        Constant reference Zernike coefs to add to fitted coefficients.
        [0] is ignored, [1] is piston, [4] is defocus, etc.
    x_offset, y_offset : galsim.zernike.Zernike, optional
        Additional pupil distortion coefficients.
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
        bkg_order=-1,
        z_ref=None,
        x_offset=None, y_offset=None,
        z_terms=(),
        thx=None, thy=None,
        npix=181,
        seed=57721,
    ):
        super().__init__(factory, npix, seed, bkg_order)
        self.z_ref = z_ref
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_terms = z_terms
        self.thx = thx
        self.thy = thy

    def model(
        self,
        flux, dx, dy, fwhm, z_fit, *,
        bkg=(),
        sky_level=None,
    ):
        """Compute donut model image.

        Parameters
        ----------
        flux : float
            Flux level at which to set image.
        dx, dy : float
            Offset in arcseconds.
        fwhm : float
            Full width half maximum of Kolmogorov kernel.
        z_fit : sequence of float
            Zernike perturbations.
        bkg : sequence of float
            Background polynomial coefficients.
        sky_level : float
            Sky level to use when adding Poisson noise to image.

        Returns
        -------
        img : array of float
            Model image.
        """
        zk = np.array(self.z_ref)
        for i, term in enumerate(self.z_terms):
            zk[term] += z_fit[i]
        return self._model(
            flux, dx, dy, fwhm,
            tuple(zk), tuple(bkg),
            self.thx, self.thy,
            x_offset=self.x_offset, y_offset=self.y_offset,
            sky_level=sky_level
        )

    def pack_params(self, *, flux, dx, dy, fwhm, z_fit, bkg=()):
        """Pack parameters into a single tuple for optimization.

        Parameters
        ----------
        flux : float
            Flux level at which to set image.
        dx, dy : float
            Offset in arcseconds.
        fwhm : float
            Full width half maximum of Kolmogorov kernel.
        z_fit : array of float
            Zernike perturbations.
        bkg : array of float
            Background polynomial coefficients.

        Returns
        -------
        tuple of float
            Packed parameters.
        """
        return (flux, dx, dy, fwhm, *z_fit, *bkg)

    def unpack_params(self, params):
        """
        Unpack parameters from the single tuple used for optimization.

        Parameters
        ----------
        params : sequence of float
            Packed parameters tuple as returned by `pack_params`.

        Returns
        -------
        dict
            Dictionary with keys 'flux', 'dx', 'dy', 'fwhm', 'z_fit', and 'bkg'.
        """
        flux, dx, dy, fwhm, *rest = params
        out = dict(
            flux=flux,
            dx=dx,
            dy=dy,
            fwhm=fwhm,
        )
        split = len(rest) - self.nbkg
        out["z_fit"] = rest[:split]
        out["bkg"] = rest[split:]
        return out

    def chi(
        self, params, data, var
    ):
        """Compute chi = (data - model)/error.

        The error is modeled as sqrt(model + var).

        Parameters
        ----------
        params : sequence of float
            Order is: (flux, dx, dy, fwhm, *z_fit, *bkg)
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
        mod = self.model(**self.unpack_params(params))
        return self._chi(data, mod, var)

    def jac(
        self, params, data, var
    ):
        """Compute jacobian d(chi)/d(param).

        Parameters
        ----------
        params : sequence of float
            Order is: (flux, dx, dy, fwhm, *z_fit, *bkg)
        data : array of float
            Image against which to compute chi.
        var : float or array of float (npix, npix)
            Variance of the sky only.  Do not include Poisson contribution of
            the signal, as this will be added from the current model.

        Returns
        -------
        jac : array of float
            Jacobian array d(chi)/d(param).  First dimension is pixels, second
            dimension is param.
        """
        out = np.zeros((self.npix**2, len(params)))
        chi0 = self.chi(params, data, var)

        step = [
            0.01, # flux
            0.01, # dx
            0.01, # dy
            0.01  # fwhm
        ]
        step += [1e-8]*len(self.z_terms)  # Zernikes (in meters)
        step += [0.01]*self.nbkg

        for i in range(len(params)):
            params1 = np.array(params)
            params1[i] += step[i]
            chi1 = self.chi(params1, data, var)
            out[:, i] = (chi1-chi0)/step[i]

        return out


class BaseMultiDonutModel(BaseDonutModel):
    """Model multiple donuts simultaneously.

    Parameters
    ----------
    factory : DonutFactory
    bkg_order : int, optional
        Order of the background polynomial to fit.  If -1, no background.
    dz_ref : DoubleZernike
        Double Zernike coefficients to use for constructing Single Zernike
        reference coefficients to use for each modeled donut.  Either this kwarg
        or `z_refs` must be set.
    z_refs : array of float
        Single Zernike reference coefficients for each donut.  First dimension
        is donut, second dimension is pupil Zernike coefficient.
    field_radius : float
        Field radius in radians.  If dz_ref is provided, then this is ignored and
        the field radius will be inferred from dz_ref.
    thxs, thys : float
        Field angles in radians.
    wavefront_step : float, optional
        Step size for wavefront parameters.
    npix : int
        Number of pixels along image edge.  Must be odd.
    seed : int
        Random seed for use when creating noisy donut images with this class.
    """
    def __init__(
        self,
        factory, *,
        bkg_order=-1,
        dz_ref=None,
        z_refs=None,
        field_radius=None,
        thxs=None, thys=None,
        wavefront_step=1e-8,
        npix=181,
        seed=577215
    ):
        super().__init__(factory, npix, seed, bkg_order)

        if dz_ref is None and z_refs is None:
            raise ValueError("Must provide dz_ref or z_refs")
        if dz_ref is not None and z_refs is not None:
            raise ValueError("Cannot provide both dz_ref and z_refs")
        if z_refs is None:
            z_refs = dz_ref.xycoef(thxs, thys)
        self.z_refs = z_refs
        if field_radius is None:
            if dz_ref is None:
                raise ValueError("Must provide dz_ref or field_radius")
            field_radius = dz_ref.field_radius
        self.field_radius = field_radius
        self.thxs = thxs
        self.thys = thys
        self.nstar = len(thxs)
        self.wavefront_step = wavefront_step

    def model(
        self, fluxes, dxs, dys, fwhm, wavefront_params, *,
        bkgs=None, sky_levels=None
    ):
        """Compute the model image for a single set of parameters.

        Parameters
        ----------
        fluxes : sequence of float
            Flux levels at which to set images.
        dxs : sequence of float
            Offsets in arcseconds along the x-axis.
        dys : sequence of float
            Offsets in arcseconds along the y-axis.
        fwhm : float
            Full width half maximum of Kolmogorov kernel. (Same for all donuts).
        wavefront_params : sequence of float
            Wavefront parameters for the model.
        bkgs : tuple of tuple of float, optional
            Background polynomial coefficients for each donut.
        sky_levels : sequence of float, optional
            Sky levels to use when adding Poisson noise to images.

        Returns
        -------
        imgs : array of float
            Model images for each donut.
        """
        z_fits = self._get_z_fits(wavefront_params)
        return self.model_many(fluxes, dxs, dys, fwhm, z_fits, bkgs=bkgs, sky_levels=sky_levels)

    def model_many(
        self, fluxes, dxs, dys, fwhm, z_fits, *,
        bkgs=None,
        sky_levels=None
    ):
        """Compute models for all donuts.

        Parameters
        ----------
        fluxes : float
            Flux levels at which to set images.
        dxs, dys : float
            Offsets in arcseconds.
        fwhm : float
            Full width half maximum of Kolmogorov kernel. (Same for all donuts).
        z_fits : sequence of tuple of float
            Single Zernike perturbations for each donut.
        bkgs : tuple of tuple of float
            Background polynomial coefficients for each donut.
        sky_levels : sequence of float, optional
            Sky levels to use when adding Poisson noise to images.

        Returns
        -------
        imgs : array of float.  Shape: (nstar, npix, npix)
            Model images.
        """
        nstar = self.nstar
        npix = self.npix
        if bkgs is None:
            bkgs = [()] * nstar
        if sky_levels is None:
            sky_levels = [None]*nstar

        out = np.empty((nstar, npix, npix))

        for i in range(nstar):
            aberrations = np.array(self.z_refs[i])
            aberrations[:len(z_fits[i])] += z_fits[i]
            out[i] = self._model(
                fluxes[i],
                dxs[i], dys[i],
                fwhm,
                tuple(aberrations),
                bkgs[i],
                thx=self.thxs[i], thy=self.thys[i],
                sky_level=sky_levels[i],
            )
        return out

    def pack_params(
        self, *, fluxes, dxs, dys, fwhm, wavefront_params, bkgs=None
    ):
        """Pack parameters into a single tuple for optimization.

        Parameters
        ----------
        fluxes : sequence of float
            Flux levels at which to set images.
        dxs, dys : sequence of float
            Offsets in arcseconds.
        fwhm : float
            Full width half maximum of Kolmogorov kernel. (Same for all donuts).
        wavefront_params : sequence of float
            Wavefront parameters for the model.
        bkgs : tuple of tuple of float, optional
            Background polynomial coefficients for each donut.

        Returns
        -------
        params : tuple of float
            Packed parameters.
        """
        if bkgs is None:
            bkgs = [()] * self.nstar
        params = []
        params.extend(fluxes)
        params.extend(dxs)
        params.extend(dys)
        params.append(fwhm)
        params.extend(wavefront_params)
        for bkg in bkgs:
            params.extend(bkg)
        return tuple(params)

    def unpack_params(self, params):
        """Unpack parameters from the single tuple used for optimization.

        Parameters
        ----------
        params : sequence of float
            Packed parameters tuple as returned by `pack_params`.

        Returns
        -------
        dict
            Dictionary with keys 'fluxes', 'dxs', 'dys', 'fwhm', 'wavefront_params', 'bkgs'.
        """
        nstar = self.nstar
        fluxes = params[:nstar]
        dxs = params[nstar:2*nstar]
        dys = params[2*nstar:3*nstar]
        fwhm = params[3*nstar]

        # Wavefront parameters
        wavefront_slice = slice(
            3*nstar+1,
            len(params) - nstar * self.nbkg
        )
        wavefront_params = params[wavefront_slice]

        bkgs = []
        for i in range(nstar):
            bkg_slice = slice(
                len(params) - nstar * self.nbkg + i * self.nbkg,
                len(params) - nstar * self.nbkg + (i+1) * self.nbkg
            )
            bkgs.append(
                tuple(params[bkg_slice])
            )

        out = dict(
            fluxes=fluxes,
            dxs=dxs,
            dys=dys,
            fwhm=fwhm,
            wavefront_params=wavefront_params,
            bkgs=bkgs
        )
        return out

    def chi(
        self, params, data, vars
    ):
        """Compute chi = (data - model)/error.

        The error is modeled as sqrt(model + var) for each donut.

        Parameters
        ----------
        params : sequence of float
            Order is: (fluxes, dxs, dys, fwhm, *wavefront_params, *bkgs)
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
        # We don't know the exact parameterization here, since there are subclasses.
        # Use self._format_params to convert to BaseMultiDonutModel.model expected
        # parameters
        mods = self.model(**self.unpack_params(params))
        chis = np.empty((self.nstar, self.npix, self.npix))
        for i, (mod, datum) in enumerate(zip(mods, data)):
            chis[i] = self._chi(datum, mod, vars[i]).reshape(self.npix, self.npix)
        return chis.ravel()

    def jac(
        self, params, data, vars
    ):
        """Compute jacobian d(chi)/d(param).

        Parameters
        ----------
        params : sequence of float
            Order is: (fluxes, dxs, dys, fwhm, *wavefront_params, *bkgs)
        data : array of float.  Shape: (nstar, npix, npix)
            Images against which to compute chi.
        vars : sequence of array (npix, npix) or sequence of float
            Variances of the sky only.  Do not include Poisson contribution of
            the signal, as this will be added from the current model.

        Returns
        -------
        jac : array of float
            Jacobian array d(chi)/d(param).  First dimension is pixels, second
            dimension is param.
        """
        # We don't know the exact parameterization here (see .chi() above).
        # We will make _some_ assumptions though.  The general order is:
        # flux for each star
        # dx for each star
        # dy for each star
        # single FWHM parameter
        # wavefront parameters
        # followed by [bkg1, bkg2, ...] for each star for the background polynomial

        # For computing the jacobian, the flux, dx, dy and bkg parameters are sparse in
        # which star is effected (i.e., only the parameters corresponding to that star
        # will affect its chi)
        # For the FWHM and wavefront parameters, all stars are affected by each
        # parameter.

        nstar = self.nstar
        npix = self.npix
        nbkg = self.nbkg
        nwavefront = len(params) - nbkg*nstar - (3*nstar + 1)

        out = np.zeros((nstar*npix**2, len(params)))
        chi0 = self.chi(params, data, vars)

        # Flux
        dflux = 0.01
        dflux_params = np.array(params)  # Make a copy so we don't perturb the original
        dflux_param_dict = self.unpack_params(dflux_params)
        dflux_param_dict["fluxes"] += dflux
        chi_flux = self.chi(self.pack_params(**dflux_param_dict), data, vars)
        # We manipulated all fluxes at once above.  Need to separate each star's
        # contribution for output here.
        for i in range(nstar):
            s = slice(i*npix**2, (i+1)*npix**2)
            out[s, i] = (chi_flux[s] - chi0[s]) / dflux

        # Repeat for dx
        dx = 0.01
        dx_params = np.array(params)
        dx_param_dict = self.unpack_params(dx_params)
        dx_param_dict["dxs"] += dx
        chi_dx = self.chi(self.pack_params(**dx_param_dict), data, vars)
        for i in range(nstar):
            s = slice(i*npix**2, (i+1)*npix**2)
            out[s, nstar+i] = (chi_dx[s] - chi0[s]) / dx

        # Repeat for dy
        dy = 0.01
        dy_params = np.array(params)
        dy_param_dict = self.unpack_params(dy_params)
        dy_param_dict["dys"] += dy
        chi_dy = self.chi(self.pack_params(**dy_param_dict), data, vars)
        for i in range(nstar):
            s = slice(i*npix**2, (i+1)*npix**2)
            out[s, 2*nstar+i] = (chi_dy[s] - chi0[s]) / dy

        # FWHM
        dfwhm = 0.01
        dfwhm_params = np.array(params)
        dfwhm_param_dict = self.unpack_params(dfwhm_params)
        dfwhm_param_dict["fwhm"] += dfwhm
        chi_fwhm = self.chi(self.pack_params(**dfwhm_param_dict), data, vars)
        # This one is dense, so just insert full chi
        out[:, 3*nstar] = (chi_fwhm - chi0)/dfwhm

        # Wavefront terms
        for i in range(3*nstar+1, 3*nstar+1+nwavefront):
            params1 = np.array(params)
            params1[i] += self.wavefront_step
            chi1 = self.chi(params1, data, vars)
            out[:, i] = (chi1-chi0)/self.wavefront_step

        # Background terms.  These are sparse too
        dbkg = 0.01
        for i in range(nbkg):
            bkg_params = np.array(params)
            bkg_param_dict = self.unpack_params(bkg_params)
            for j in range(nstar):
                bkgj = list(bkg_param_dict["bkgs"][j])
                bkgj[i] += dbkg
                bkg_param_dict["bkgs"][j] = tuple(bkgj)
            chi_bkg = self.chi(self.pack_params(**bkg_param_dict), data, vars)
            for j in range(nstar):
                s = slice(j*npix**2, (j+1)*npix**2)
                out[s, 3*nstar+1+nwavefront+nbkg*j+i] = (chi_bkg[s] - chi0[s]) / dbkg

        return out

    # Simpler version of jac for testing.  Slightly slower.
    def _jac2(self, params, data, vars):
        nstar = self.nstar
        npix = self.npix
        nbkg = self.nbkg
        nwavefront = len(params) - nbkg*nstar - (3*nstar + 1)

        out = np.empty((nstar*npix**2, len(params)))

        step = [0.01]*nstar  # flux
        step += [0.01]*nstar  # dx
        step += [0.01]*nstar  # dy
        step += [0.01]        # fwhm
        step += [self.wavefront_step]*nwavefront  # wavefront terms
        step += [0.01]*nbkg*nstar  # background terms

        chi0 = self.chi(params, data, vars)
        for i, step in enumerate(step):
            params1 = np.array(params)
            params1[i] += step
            chi1 = self.chi(params1, data, vars)
            out[:, i] = (chi1 - chi0)/step

        return out


class DZMultiDonutModel(BaseMultiDonutModel):
    """Multi donut model that uses double Zernike coefficients to parameterize the
    wavefront.

    Parameters
    ----------
    factory : DonutFactory
    dz_terms : sequence of tuple of int
        List of (k, j) indices specifying which double Zernike terms to use.
    bkg_order : int, optional
        Order of the background polynomial to fit.  If -1, no background.
    dz_ref : DoubleZernike
        Double Zernike coefficients to use for constructing Single Zernike
        reference coefficients to use for each modeled donut.  Either this kwarg
        or `z_refs` must be set.
    z_refs : array of float
        Single Zernike reference coefficients for each donut.  First dimension
        is donut, second dimension is pupil Zernike coefficient.
    field_radius : float
        Field radius in radians.  If dz_ref is provided, then this is ignored and
        the field radius will be inferred from dz_ref.
    thxs, thys : float
        Field angles in radians.
    npix : int
        Number of pixels along image edge.  Must be odd.
    seed : int
        Random seed for use when creating noisy donut images with this class.
    """
    def __init__(self, *args, dz_terms=(), **kwargs):
        # Want dz_terms
        self.dz_terms = dz_terms
        self.k_max = max([term[0] for term in dz_terms], default=0)
        self.j_max = max([term[1] for term in dz_terms], default=0)
        super().__init__(*args, **kwargs)

    def _get_z_fits(self, dz_fit):
        """Convert this class's wavefront parameterization into individual donut
        Zernike coefficients.

        Parameters
        ----------
        dz_fit : array of float
            Array of double Zernike coefficients for this model.

        Returns
        -------
        z_fits : list of array of float
            List of single Zernike coefficients for each donut.
        """
        arr = np.zeros((self.k_max+1, self.j_max+1))
        for i, (k, j) in enumerate(self.dz_terms):
            arr[k, j] = dz_fit[i]
        dz = galsim.zernike.DoubleZernike(arr, uv_outer=self.field_radius)
        z_fits = []
        for thx, thy in zip(self.thxs, self.thys):
            z_fits.append(dz.xycoef(thx, thy))
        return z_fits


class DZBasisMultiDonutModel(BaseMultiDonutModel):
    """Multi donut model that uses a sensitivity matrix to convert mode coefficients
    into double Zernike coefficients to parameterize the wavefront.

    Parameters
    ----------
    factory : DonutFactory
    sensitivity : array of float
        Sensitivity matrix that converts mode coefficients into double Zernike
        coefficients.  Dimensions are (nmode, k_max+1, j_max+1)
    bkg_order : int, optional
        Order of the background polynomial to fit.  If -1, no background.
    dz_ref : DoubleZernike
        Double Zernike coefficients to use for constructing Single Zernike
        reference coefficients to use for each modeled donut.  Either this kwarg
        or `z_refs` must be set.
    z_refs : array of float
        Single Zernike reference coefficients for each donut.  First dimension
        is donut, second dimension is pupil Zernike coefficient.
    field_radius : float
        Field radius in radians.  If dz_ref is provided, then this is ignored and
        the field radius will be inferred from dz_ref.
    thxs, thys : float
        Field angles in radians.
    npix : int
        Number of pixels along image edge.  Must be odd.
    seed : int
        Random seed for use when creating noisy donut images with this class.
    """
    def __init__(self, *args, sensitivity=None, **kwargs):
        if sensitivity is None:
            raise ValueError("Must provide sensitivity")
        self.sensitivity = sensitivity
        self.nmode = self.sensitivity.shape[0]
        self.k_max = self.sensitivity.shape[1]
        self.j_max = self.sensitivity.shape[2]
        super().__init__(*args, **kwargs)

    def _get_z_fits(self, mode_coefs):
        """Convert this class's wavefront parameterization into individual donut
        Zernike coefficients.

        Parameters
        ----------
        mode_coefs : array of float
            Array of mode coefficients for this model.

        Returns
        -------
        z_fits : list of array of float
            List of single Zernike coefficients for each donut.
        """
        arr = np.einsum(
            "i,ikj->kj",
            mode_coefs,
            self.sensitivity
        )
        dz = galsim.zernike.DoubleZernike(arr, uv_outer=self.field_radius)
        z_fits = []
        for thx, thy in zip(self.thxs, self.thys):
            z_fits.append(dz.xycoef(thx, thy))
        return z_fits
