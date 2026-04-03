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
from .utils import gq_points

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
    atm_mode : str, optional
        Atmospheric PSF parameterization.  'fwhm' for scalar FWHM (default),
        'ixx' for second moment tensor (Ixx, Ixy, Iyy) in arcsec^2.
    """
    KOLM_CONST = 0.651  # TODO: Compute this more precisely.

    def __init__(self, factory, npix, seed, bkg_order, atm_mode='fwhm'):
        assert npix % 2 == 1, "npix must be odd"
        assert atm_mode in ('fwhm', 'ixx'), "atm_mode must be 'fwhm' or 'ixx'"

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
        self.atm_mode = atm_mode

    @property
    def natm(self):
        """Number of atmospheric parameters (1 for fwhm, 3 for ixx)."""
        return 1 if self.atm_mode == 'fwhm' else 3

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
        seed=577215,
        atm_mode='fwhm',
    ):
        super().__init__(factory, npix, seed, bkg_order, atm_mode=atm_mode)

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
        self, fluxes, dxs, dys, fwhm=None, wavefront_params=None, *,
        Ixx=None, Ixy=None, Iyy=None,
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
        fwhm : float, optional
            Full width half maximum of Kolmogorov kernel (atm_mode='fwhm').
        wavefront_params : sequence of float
            Wavefront parameters for the model.
        Ixx, Ixy, Iyy : float, optional
            Second moment tensor in arcsec^2 (atm_mode='ixx').
        bkgs : tuple of tuple of float, optional
            Background polynomial coefficients for each donut.
        sky_levels : sequence of float, optional
            Sky levels to use when adding Poisson noise to images.

        Returns
        -------
        imgs : array of float
            Model images for each donut.
        """
        if self.atm_mode == 'ixx':
            fwhm = np.sqrt((Ixx + Iyy) / 2) / self.KOLM_CONST
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
        self, *, fluxes, dxs, dys, fwhm=None,
        Ixx=None, Ixy=None, Iyy=None,
        wavefront_params, bkgs=None
    ):
        """Pack parameters into a single tuple for optimization.

        Parameters
        ----------
        fluxes : sequence of float
            Flux levels at which to set images.
        dxs, dys : sequence of float
            Offsets in arcseconds.
        fwhm : float, optional
            Full width half maximum of Kolmogorov kernel (atm_mode='fwhm').
        Ixx, Ixy, Iyy : float, optional
            Second moment tensor in arcsec^2 (atm_mode='ixx').
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
        if self.atm_mode == 'fwhm':
            params.append(fwhm)
        else:
            params.extend([Ixx, Ixy, Iyy])
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
            Dictionary with keys 'fluxes', 'dxs', 'dys', and either 'fwhm'
            (atm_mode='fwhm') or 'Ixx','Ixy','Iyy' (atm_mode='ixx'),
            plus 'wavefront_params' and 'bkgs'.
        """
        nstar = self.nstar
        natm = self.natm
        fluxes = params[:nstar]
        dxs = params[nstar:2*nstar]
        dys = params[2*nstar:3*nstar]

        out = dict(fluxes=fluxes, dxs=dxs, dys=dys)
        if self.atm_mode == 'fwhm':
            out['fwhm'] = params[3*nstar]
        else:
            out['Ixx'] = params[3*nstar]
            out['Ixy'] = params[3*nstar + 1]
            out['Iyy'] = params[3*nstar + 2]

        # Wavefront parameters
        wavefront_slice = slice(
            3*nstar + natm,
            len(params) - nstar * self.nbkg
        )
        out['wavefront_params'] = params[wavefront_slice]

        bkgs = []
        for i in range(nstar):
            bkg_slice = slice(
                len(params) - nstar * self.nbkg + i * self.nbkg,
                len(params) - nstar * self.nbkg + (i+1) * self.nbkg
            )
            bkgs.append(
                tuple(params[bkg_slice])
            )
        out['bkgs'] = bkgs
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
        natm = self.natm
        nwavefront = len(params) - nbkg*nstar - (3*nstar + natm)

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

        # Atmospheric parameters - dense (shared across all stars)
        datm = 0.01
        for k in range(natm):
            params1 = np.array(params)
            params1[3*nstar + k] += datm
            chi1 = self.chi(params1, data, vars)
            out[:, 3*nstar + k] = (chi1 - chi0) / datm

        # Wavefront terms
        for i in range(3*nstar + natm, 3*nstar + natm + nwavefront):
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
                out[s, 3*nstar + natm + nwavefront + nbkg*j + i] = (chi_bkg[s] - chi0[s]) / dbkg

        return out

    def _jac2(self, params, data, vars):
        nstar = self.nstar
        npix = self.npix
        nbkg = self.nbkg
        natm = self.natm
        nwavefront = len(params) - nbkg*nstar - (3*nstar + natm)

        out = np.empty((nstar*npix**2, len(params)))

        step = [0.01]*nstar  # flux
        step += [0.01]*nstar  # dx
        step += [0.01]*nstar  # dy
        step += [0.01]*natm   # atm params
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


class BaseSpotModel:
    """Base class for spot image models.

    Parameters
    ----------
    factory : DonutFactory
    bkg_order : int
        Order of polynomial background model to use.  If -1, no background.
    npix : int
        Number of pixels along image edge.  Must be odd.
    seed : int
        Random seed for use when creating noisy spot images.
    atm_mode : str, optional
        Atmospheric PSF parameterization.  'fwhm' for scalar FWHM (default),
        'ixx' for second moment tensor (Ixx, Ixy, Iyy) in arcsec^2.
    spot_nrad : int, optional
        Number of pupil radii for factory.spots().
    spot_naz : int, optional
        Number of azimuthal points for factory.spots().
    gq_kwargs : dict, optional
        Keyword arguments for gq_points (e.g. nrad, kfold, rmax).
    """
    KOLM_CONST = 0.651

    def __init__(
        self, factory, npix, seed, bkg_order, atm_mode='fwhm',
        spot_nrad=5, spot_naz=None, gq_kwargs=None
    ):
        assert npix % 2 == 1, "npix must be odd"
        assert atm_mode in ('fwhm', 'ixx'), "atm_mode must be 'fwhm' or 'ixx'"

        self.factory = factory
        self.npix = npix
        self.no2 = (npix-1)//2
        self.gsrng = galsim.BaseDeviate(seed)
        self.sky_scale = (
            3600*np.rad2deg(1/factory.focal_length)*factory.pixel_scale
        )
        self.arcsec_to_meters = factory.pixel_scale / self.sky_scale
        self.bkg_order = bkg_order
        self.nbkg = (bkg_order+1)*(bkg_order+2)//2
        self.atm_mode = atm_mode
        self.spot_nrad = spot_nrad
        self.spot_naz = spot_naz
        self.gq_kwargs = gq_kwargs or {}

    @property
    def natm(self):
        """Number of atmospheric parameters (1 for fwhm, 3 for ixx)."""
        return 1 if self.atm_mode == 'fwhm' else 3

    @lru_cache(maxsize=1000)
    def _spot_positions(self, zk, thx, thy):
        """Compute spot positions from factory.

        Parameters
        ----------
        zk : tuple of float
            Aberration coefficients.
        thx, thy : float
            Field angle in radians.

        Returns
        -------
        sx, sy : array of float
            Spot positions in meters.
        sw : array of bool
            Vignetting mask.
        """
        sx, sy, sw = self.factory.spots(
            aberrations=zk,
            thx=thx, thy=thy,
            nrad=self.spot_nrad, naz=self.spot_naz
        )
        return sx, sy, sw

    @lru_cache(maxsize=100)
    def _gq(self, Ixx, Ixy, Iyy):
        """Compute Gaussian quadrature points for atmospheric PSF.

        Parameters
        ----------
        Ixx, Ixy, Iyy : float
            Second moment tensor in arcsec^2.

        Returns
        -------
        gx, gy : array of float
            Quadrature points in arcsec.
        gw : array of float
            Quadrature weights.
        """
        cov = np.array([[Ixx, Ixy], [Ixy, Iyy]])
        return gq_points(cov=cov, **self.gq_kwargs)

    @lru_cache(maxsize=1000)
    def _bkg(self, bkg):
        """Compute background model.

        Parameters
        ----------
        bkg : tuple of float
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
        flux, dx, dy, Ixx, Ixy, Iyy, zk, bkg,
        thx, thy,
        sky_level=None
    ):
        """Compute spot image model.

        Parameters
        ----------
        flux : float
            Flux level.
        dx, dy : float
            Centroid offset in arcseconds.
        Ixx, Ixy, Iyy : float
            Second moment tensor in arcsec^2.
        zk : tuple of float
            Aberration coefficients.
        bkg : tuple of float
            Background polynomial coefficients.
        thx, thy : float
            Field angle in radians.
        sky_level : float, optional
            Sky level for Poisson noise.

        Returns
        -------
        array of float
            Spot image array.
        """
        sx, sy, sw = self._spot_positions(zk, thx, thy)
        gx, gy, gw = self._gq(Ixx, Ixy, Iyy)

        a2m = self.arcsec_to_meters
        x = np.add.outer(sx, gx * a2m) + dx * a2m
        y = np.add.outer(sy, gy * a2m) + dy * a2m
        w = np.multiply.outer(sw, gw)

        no2 = self.no2
        bds = np.linspace(-no2-0.5, no2+0.5, self.npix+1) * self.factory.pixel_scale
        img, *_ = np.histogram2d(
            y.ravel(), x.ravel(),
            bins=[bds, bds],
            weights=w.ravel(),
            density=False
        )

        total = np.sum(img)
        if total > 0:
            img *= flux / total

        if bkg:
            img += self._bkg(bkg)

        if sky_level is not None:
            pn = galsim.PoissonNoise(self.gsrng, sky_level=sky_level)
            gsimg = galsim.Image(img)
            gsimg.addNoise(pn)
            img = gsimg.array

        return img

    def _chi(self, data, model, var):
        """Compute chi = (data - model)/error."""
        result = ((data-model)/np.sqrt(var+model)).ravel()
        return result


class BaseMultiSpotModel(BaseSpotModel):
    """Model multiple spot images simultaneously.

    Parameters
    ----------
    factory : DonutFactory
    bkg_order : int, optional
        Order of the background polynomial to fit.  If -1, no background.
    dz_ref : DoubleZernike
        Double Zernike reference coefficients.  Either this or z_refs must
        be set.
    z_refs : array of float
        Single Zernike reference coefficients for each star.
    field_radius : float
        Field radius in radians.
    thxs, thys : sequence of float
        Field angles in radians.
    wavefront_step : float, optional
        Finite difference step size for wavefront parameters.
    npix : int
        Number of pixels along image edge.  Must be odd.
    seed : int
        Random seed.
    atm_mode : str, optional
        'fwhm' or 'ixx'.
    spot_nrad : int, optional
        Number of pupil radii for factory.spots().
    spot_naz : int, optional
        Number of azimuthal points for factory.spots().
    gq_kwargs : dict, optional
        Keyword arguments for gq_points.
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
        npix=21,
        seed=577215,
        atm_mode='fwhm',
        spot_nrad=5,
        spot_naz=None,
        gq_kwargs=None,
    ):
        super().__init__(
            factory, npix, seed, bkg_order,
            atm_mode=atm_mode,
            spot_nrad=spot_nrad, spot_naz=spot_naz,
            gq_kwargs=gq_kwargs,
        )

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
        self, fluxes, dxs, dys, fwhm=None, wavefront_params=None, *,
        Ixx=None, Ixy=None, Iyy=None,
        bkgs=None, sky_levels=None
    ):
        """Compute spot image models.

        Parameters
        ----------
        fluxes : sequence of float
            Flux levels for each star.
        dxs, dys : sequence of float
            Centroid offsets in arcseconds.
        fwhm : float, optional
            FWHM in arcsec (used when atm_mode='fwhm').
        wavefront_params : sequence of float
            Wavefront parameters for the model.
        Ixx, Ixy, Iyy : float, optional
            Second moment tensor in arcsec^2 (used when atm_mode='ixx').
        bkgs : tuple of tuple of float, optional
            Background polynomial coefficients for each star.
        sky_levels : sequence of float, optional
            Sky levels for Poisson noise.

        Returns
        -------
        imgs : array of float, shape (nstar, npix, npix)
            Model spot images.
        """
        if self.atm_mode == 'fwhm':
            Ixx = Iyy = (fwhm * self.KOLM_CONST) ** 2
            Ixy = 0.0
        z_fits = self._get_z_fits(wavefront_params)
        return self.model_many(
            fluxes, dxs, dys, Ixx, Ixy, Iyy, z_fits,
            bkgs=bkgs, sky_levels=sky_levels
        )

    def model_many(
        self, fluxes, dxs, dys, Ixx, Ixy, Iyy, z_fits, *,
        bkgs=None,
        sky_levels=None
    ):
        """Compute spot image models for all stars.

        Parameters
        ----------
        fluxes : sequence of float
            Flux levels.
        dxs, dys : sequence of float
            Centroid offsets in arcseconds.
        Ixx, Ixy, Iyy : float
            Second moment tensor in arcsec^2.
        z_fits : sequence of tuple of float
            Single Zernike perturbations for each star.
        bkgs : tuple of tuple of float, optional
            Background polynomial coefficients.
        sky_levels : sequence of float, optional
            Sky levels for Poisson noise.

        Returns
        -------
        imgs : array of float, shape (nstar, npix, npix)
            Model spot images.
        """
        nstar = self.nstar
        npix = self.npix
        if bkgs is None:
            bkgs = [()] * nstar
        if sky_levels is None:
            sky_levels = [None] * nstar

        out = np.empty((nstar, npix, npix))

        for i in range(nstar):
            aberrations = np.array(self.z_refs[i])
            aberrations[:len(z_fits[i])] += z_fits[i]
            out[i] = self._model(
                fluxes[i],
                dxs[i], dys[i],
                Ixx, Ixy, Iyy,
                tuple(aberrations),
                bkgs[i],
                thx=self.thxs[i], thy=self.thys[i],
                sky_level=sky_levels[i],
            )
        return out

    def pack_params(
        self, *, fluxes, dxs, dys, fwhm=None,
        Ixx=None, Ixy=None, Iyy=None,
        wavefront_params, bkgs=None
    ):
        """Pack parameters into a single tuple for optimization.

        Parameters
        ----------
        fluxes : sequence of float
        dxs, dys : sequence of float
        fwhm : float, optional
            Used when atm_mode='fwhm'.
        Ixx, Ixy, Iyy : float, optional
            Used when atm_mode='ixx'.
        wavefront_params : sequence of float
        bkgs : tuple of tuple of float, optional

        Returns
        -------
        params : tuple of float
        """
        if bkgs is None:
            bkgs = [()] * self.nstar
        params = []
        params.extend(fluxes)
        params.extend(dxs)
        params.extend(dys)
        if self.atm_mode == 'fwhm':
            params.append(fwhm)
        else:
            params.extend([Ixx, Ixy, Iyy])
        params.extend(wavefront_params)
        for bkg in bkgs:
            params.extend(bkg)
        return tuple(params)

    def unpack_params(self, params):
        """Unpack parameters from optimization tuple.

        Parameters
        ----------
        params : sequence of float

        Returns
        -------
        dict
            Dictionary with keys 'fluxes', 'dxs', 'dys', and either 'fwhm'
            (atm_mode='fwhm') or 'Ixx','Ixy','Iyy' (atm_mode='ixx'),
            plus 'wavefront_params' and 'bkgs'.
        """
        nstar = self.nstar
        natm = self.natm
        fluxes = params[:nstar]
        dxs = params[nstar:2*nstar]
        dys = params[2*nstar:3*nstar]

        out = dict(fluxes=fluxes, dxs=dxs, dys=dys)
        if self.atm_mode == 'fwhm':
            out['fwhm'] = params[3*nstar]
        else:
            out['Ixx'] = params[3*nstar]
            out['Ixy'] = params[3*nstar + 1]
            out['Iyy'] = params[3*nstar + 2]

        wavefront_slice = slice(
            3*nstar + natm,
            len(params) - nstar * self.nbkg
        )
        out['wavefront_params'] = params[wavefront_slice]

        bkgs = []
        for i in range(nstar):
            bkg_slice = slice(
                len(params) - nstar * self.nbkg + i * self.nbkg,
                len(params) - nstar * self.nbkg + (i+1) * self.nbkg
            )
            bkgs.append(
                tuple(params[bkg_slice])
            )
        out['bkgs'] = bkgs
        return out

    def chi(
        self, params, data, vars
    ):
        """Compute chi = (data - model)/error.

        Parameters
        ----------
        params : sequence of float
        data : array of float, shape (nstar, npix, npix)
        vars : sequence of array or float

        Returns
        -------
        chi : array of float
        """
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
        data : array of float, shape (nstar, npix, npix)
        vars : sequence of array or float

        Returns
        -------
        jac : array of float
        """
        nstar = self.nstar
        npix = self.npix
        nbkg = self.nbkg
        natm = self.natm
        nwavefront = len(params) - nbkg*nstar - (3*nstar + natm)

        out = np.zeros((nstar*npix**2, len(params)))
        chi0 = self.chi(params, data, vars)

        # Flux - sparse
        dflux = 0.01
        dflux_params = np.array(params)
        dflux_param_dict = self.unpack_params(dflux_params)
        dflux_param_dict["fluxes"] += dflux
        chi_flux = self.chi(self.pack_params(**dflux_param_dict), data, vars)
        for i in range(nstar):
            s = slice(i*npix**2, (i+1)*npix**2)
            out[s, i] = (chi_flux[s] - chi0[s]) / dflux

        # dx - sparse
        dx = 0.01
        dx_params = np.array(params)
        dx_param_dict = self.unpack_params(dx_params)
        dx_param_dict["dxs"] += dx
        chi_dx = self.chi(self.pack_params(**dx_param_dict), data, vars)
        for i in range(nstar):
            s = slice(i*npix**2, (i+1)*npix**2)
            out[s, nstar+i] = (chi_dx[s] - chi0[s]) / dx

        # dy - sparse
        dy = 0.01
        dy_params = np.array(params)
        dy_param_dict = self.unpack_params(dy_params)
        dy_param_dict["dys"] += dy
        chi_dy = self.chi(self.pack_params(**dy_param_dict), data, vars)
        for i in range(nstar):
            s = slice(i*npix**2, (i+1)*npix**2)
            out[s, 2*nstar+i] = (chi_dy[s] - chi0[s]) / dy

        # Atmospheric parameters - dense (shared across all stars)
        datm = 0.01
        for k in range(natm):
            params1 = np.array(params)
            params1[3*nstar + k] += datm
            chi1 = self.chi(params1, data, vars)
            out[:, 3*nstar + k] = (chi1 - chi0) / datm

        # Wavefront terms
        for i in range(3*nstar + natm, 3*nstar + natm + nwavefront):
            params1 = np.array(params)
            params1[i] += self.wavefront_step
            chi1 = self.chi(params1, data, vars)
            out[:, i] = (chi1-chi0)/self.wavefront_step

        # Background terms - sparse
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
                out[s, 3*nstar + natm + nwavefront + nbkg*j + i] = (chi_bkg[s] - chi0[s]) / dbkg

        return out

    def _jac2(self, params, data, vars):
        nstar = self.nstar
        npix = self.npix
        nbkg = self.nbkg
        natm = self.natm
        nwavefront = len(params) - nbkg*nstar - (3*nstar + natm)

        out = np.empty((nstar*npix**2, len(params)))

        step = [0.01]*nstar  # flux
        step += [0.01]*nstar  # dx
        step += [0.01]*nstar  # dy
        step += [0.01]*natm   # atm params
        step += [self.wavefront_step]*nwavefront
        step += [0.01]*nbkg*nstar

        chi0 = self.chi(params, data, vars)
        for i, step in enumerate(step):
            params1 = np.array(params)
            params1[i] += step
            chi1 = self.chi(params1, data, vars)
            out[:, i] = (chi1 - chi0)/step

        return out


class DZMultiSpotModel(BaseMultiSpotModel):
    """Multi spot model using double Zernike coefficients to parameterize the
    wavefront.

    Parameters
    ----------
    dz_terms : sequence of tuple of int
        List of (k, j) indices specifying which double Zernike terms to use.
    """
    def __init__(self, *args, dz_terms=(), **kwargs):
        self.dz_terms = dz_terms
        self.k_max = max([term[0] for term in dz_terms], default=0)
        self.j_max = max([term[1] for term in dz_terms], default=0)
        super().__init__(*args, **kwargs)

    def _get_z_fits(self, dz_fit):
        """Convert double Zernike coefficients to per-star single Zernikes.

        Parameters
        ----------
        dz_fit : sequence of float
            Double Zernike coefficient values.

        Returns
        -------
        z_fits : list of array of float
        """
        arr = np.zeros((self.k_max+1, self.j_max+1))
        for i, (k, j) in enumerate(self.dz_terms):
            arr[k, j] = dz_fit[i]
        dz = galsim.zernike.DoubleZernike(arr, uv_outer=self.field_radius)
        z_fits = []
        for thx, thy in zip(self.thxs, self.thys):
            z_fits.append(dz.xycoef(thx, thy))
        return z_fits


class DZBasisMultiSpotModel(BaseMultiSpotModel):
    """Multi spot model using a sensitivity matrix to convert mode coefficients
    into double Zernike coefficients to parameterize the wavefront.

    Parameters
    ----------
    sensitivity : array of float
        Sensitivity matrix, shape (nmode, k_max+1, j_max+1).
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
        """Convert mode coefficients to per-star single Zernikes.

        Parameters
        ----------
        mode_coefs : sequence of float
            Mode coefficient values.

        Returns
        -------
        z_fits : list of array of float
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
