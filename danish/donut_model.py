# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC. and
# Stanford University.
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
from scipy.sparse import csc_matrix, hstack


from .factory import pupil_to_focal_basis
from .loss import chi2_loss

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


JAC_METHODS = ('analytic', 'fd')
"""Differentiation methods accepted by `jac` and `jac_sparse`."""


def _check_jac_method(method):
    """Reject an unrecognized `method` rather than silently differencing."""
    if method not in JAC_METHODS:
        raise ValueError(
            f"method must be one of {JAC_METHODS}, got {method!r}"
        )


class BaseDonutModel:
    """Base class for donut models.

    Parameters
    ----------
    factory : DonutFactoryBase
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
    # I think Ixx might formally diverge for a Kolmogorov profile, but we need
    # something to convert between FWHM and Ixx/Iyy for the atmospheric kernel,
    # So we will just use the constant factor here.  This is computed from
    # galsim.Kolmogorov(fwhm=1).calculateMomentRadius().  So we have:
    #   sigma = KOLM_CONST * fwhm
    #   Ixx = Iyy = sigma^2 = (KOLM_CONST * fwhm)^2
    # and
    #   fwhm = sigma / KOLM_CONST = sqrt(Ixx) / KOLM_CONST
    # which we will generalize to
    #   fwhm = sqrt((Ixx + Iyy) / 2) / KOLM_CONST

    KOLM_CONST = 0.651

    def __init__(
        self, factory, npix, seed, bkg_order, atm_mode='fwhm', loss_fn=None,
    ):
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
        self.loss_fn = loss_fn if loss_fn is not None else chi2_loss
        # Triangle meshes for this model's field angles, keyed on (thx, thy).
        # Populated by _precompute_meshes once the subclass knows its angles.
        self._meshes = {}
        # Projection bases parallel to _meshes; see _precompute_projections.
        self._projections = {}

    @property
    def natm(self):
        """Number of atmospheric parameters (1 for fwhm, 3 for ixx)."""
        return 1 if self.atm_mode == 'fwhm' else 3

    def _precompute_meshes(self, thxs, thys):
        """Build this model's triangle meshes once, up front.

        A model's field angles are fixed at construction, so the meshes they
        imply can be built eagerly.

        No-op for factories that do not render from a mesh (the inverse
        factory), in which case `_opt_kernel` falls back to `factory.image`.

        Parameters
        ----------
        thxs, thys : sequence of float
            Field angles in radians.
        """
        if not hasattr(self.factory, '_build_mesh'):
            return
        for thx, thy in zip(thxs, thys):
            if (thx, thy) not in self._meshes:
                self._meshes[(thx, thy)] = self.factory._build_mesh(thx, thy)

    def _zernike_param_basis(self):
        """Construct a basis of per-donut Zernike coefficients, one slice per
        wavefront parameter.

        Each slice holds the Zernike coefficients this model's `_get_z_fits`
        produces for a unit of one wavefront parameter, so that

            z_fits = sum_i params[i] * basis[..., i]

        reproduces `_get_z_fits(params)`.  Equivalently each slice is
        d(Zernike coefficient)/d(parameter).

        Requires that `_get_z_fits` is a linear function of the wavefront
        parameters -- `test_zernike_param_basis_is_linear` pins the
        requirement.

        Returns
        -------
        array of float, shape (nstar, nzk, nwavefront)
            Zernike coefficients per donut for each wavefront parameter.
        """
        cols = []
        unit = np.zeros(self._nwavefront)
        for i in range(self._nwavefront):
            unit[i] = 1.0
            cols.append(np.array(self._get_z_fits(unit)))
            unit[i] = 0.0
        return np.stack(cols, axis=-1)

    def _precompute_projections(self, z_refs, x_offset=None, y_offset=None):
        """Precompute each mesh's vertex projection over the fitted subspace.

        Two bases meet here: `_zernike_param_basis` takes wavefront parameters
        to Zernike coefficients, and `pupil_to_focal_basis` takes Zernike
        coefficients to vertex pixel positions.  Only the coefficients the
        first can reach are worth carrying -- any Noll index absent from the
        parameterization is fixed for the whole fit, so its contribution is
        folded into a constant position and its column dropped.

        What is stored per field angle is that constant plus the projection
        restricted to the reachable coefficients.  A model call projects with
        one matrix product over that subspace, and contracting the same matrix
        with `_zernike_param_basis` gives d(vertex)/d(wavefront parameter) for
        an analytic Jacobian.

        Requires `_precompute_meshes` to have run, and is a no-op without it.

        Parameters
        ----------
        z_refs : array of float
            Reference aberrations per field angle, in meters.  Must be ordered
            to match the angles passed to `_precompute_meshes`.
        x_offset, y_offset : galsim.zernike.Zernike, optional
            Additional focal plane offsets, constant across the fit.
        """
        # A model with no wavefront parameters -- one simulating images rather
        # than fitting them -- has no subspace to build a basis over.
        if not self._meshes or not self._nwavefront:
            return
        factory = self.factory
        dzkdp = self._zernike_param_basis()
        nterm = dzkdp.shape[1]
        # coef[j] labels the jth Noll index, so coef[0] is unused and a
        # length-1 coefficient array carries no Zernike terms at all.
        if nterm < 2:
            return
        # A coefficient is reachable if any parameter moves it, in any donut.
        self._zk_reach = np.nonzero(np.any(dzkdp != 0, axis=(0, 2)))[0]
        self._dzkdp = dzkdp[:, self._zk_reach, :]
        reach = self._zk_reach
        for i, ((thx, thy), z_ref) in enumerate(zip(self._meshes, z_refs)):
            verts = self._meshes[(thx, thy)]["vertices"]
            x0, y0, dxdz, dydz = pupil_to_focal_basis(
                verts[:, 0], verts[:, 1],
                aberrations=z_ref, nterm=nterm,
                R_outer=factory.R_outer, R_inner=factory.R_inner,
                focal_length=factory.focal_length,
                pixel_scale=factory.pixel_scale,
                x_offset=x_offset, y_offset=y_offset,
            )
            # Unreachable leading coefficients keep their reference value, so
            # they join the fixed tail already in x0/y0.
            fixed = np.zeros(nterm)
            fixed[:] = np.asarray(z_ref, dtype=float)[:nterm]
            fixed[reach] = 0.0
            self._projections[(thx, thy)] = (
                x0 + dxdz @ fixed,
                y0 + dydz @ fixed,
                np.ascontiguousarray(dxdz[:, reach]),
                np.ascontiguousarray(dydz[:, reach]),
            )

    @lru_cache(maxsize=1000)
    def _atm_kernel(self, dx, dy, Ixx, Ixy, Iyy):
        """Compute atmospheric kernel.

        Parameters
        ----------
        dx, dy : float
            Offset in arcseconds.
        Ixx, Ixy, Iyy : float, optional
            Second moment tensor in arcsec^2.

        Returns
        -------
        array of float
            Atmospheric kernel array.
        """
        fwhm = np.sqrt((Ixx + Iyy) / 2) / self.KOLM_CONST
        e1 = (Ixx - Iyy) / (Ixx + Iyy)
        e2 = 2*Ixy / (Ixx + Iyy)
        obj = galsim.Kolmogorov(fwhm=fwhm).shear(e1=e1, e2=e2).shift(dx, dy)
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
        if hasattr(self.factory, '_image_from_mesh'):
            # This model's angles are precomputed; fall back to the factory's
            # own memo for any angle we were not constructed with.
            mesh = self._meshes.get((thx, thy))
            if mesh is None:
                mesh = self.factory._get_mesh(thx, thy)
            basis = self._projections.get((thx, thy))
            verts_px = None
            if basis is not None:
                x0, y0, dxdz, dydz = basis
                head = np.asarray(zk)[self._zk_reach]
                verts_px = np.column_stack(
                    [x0 + dxdz @ head, y0 + dydz @ head]
                )
            return self.factory._image_from_mesh(
                mesh,
                aberrations=zk,
                focal_length=self.factory.focal_length,
                pixel_scale=self.factory.pixel_scale,
                npix=self.npix,
                x_offset=x_offset, y_offset=y_offset,
                verts_px=verts_px,
            )
        result = self.factory.image(
            aberrations=zk,
            x_offset=x_offset, y_offset=y_offset,
            thx=thx, thy=thy,
            npix=self.npix,
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
        flux, dx, dy, Ixx, Ixy, Iyy, zk, bkg,
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
        Ixx, Ixy, Iyy : float
            Second moment tensor in arcsec^2.
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
        atm = self._atm_kernel(dx, dy, Ixx, Ixy, Iyy)
        opt = self._opt_kernel(zk, thx, thy, x_offset=x_offset, y_offset=y_offset)
        arr = convolve(opt, atm)
        arr *= flux

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
        result = self.loss_fn(data, model, var).ravel()
        return result


class SingleDonutModel(BaseDonutModel):
    """Model individual donuts using single Zernike offsets to a reference
    single Zernike series.

    Parameters
    ----------
    factory : DonutFactoryBase
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
        loss_fn=None,
    ):
        super().__init__(
            factory, npix, seed, bkg_order, loss_fn=loss_fn,
        )
        self.z_ref = z_ref
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_terms = z_terms
        self.thx = thx
        self.thy = thy
        if thx is not None and thy is not None:
            self._precompute_meshes([thx], [thy])

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
        Ixx = Iyy = (fwhm * self.KOLM_CONST) ** 2
        Ixy = 0.0
        zk = np.array(self.z_ref)
        for i, term in enumerate(self.z_terms):
            zk[term] += z_fit[i]
        return self._model(
            flux, dx, dy, Ixx, Ixy, Iyy,
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

        flux = params[0]
        step = [
            1e-3 * flux,  # flux
            0.01,         # dx
            0.01,         # dy
            0.01          # fwhm
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
    factory : DonutFactoryBase
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
        loss_fn=None,
    ):
        super().__init__(
            factory, npix, seed, bkg_order, atm_mode=atm_mode, loss_fn=loss_fn,
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
        self._precompute_meshes(thxs, thys)
        self._precompute_projections(self.z_refs)

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
        """Compute models for all donuts.

        Parameters
        ----------
        fluxes : float
            Flux levels at which to set images.
        dxs, dys : float
            Offsets in arcseconds.
        Ixx, Ixy, Iyy : float
            Second moment tensor in arcsec^2.
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

    def _can_use_analytic(self):
        """Whether the analytic wavefront Jacobian is available.

        It needs the mesh renderer's projection bases, which only the triangle
        factory provides and only when the fit varies something.
        """
        return bool(self._projections)

    def _wavefront_jac_analytic(self, params, data, vars):
        """Analytic d(chi)/d(wavefront parameter), one column per parameter.

        The wavefront reaches chi through the optical kernel alone, so

            d(chi)/dp = d(chi)/d(model) * flux * convolve(d(opt)/dp, atm)

        with d(opt)/dp coming from one clip pass over the mesh rather than a
        re-render per parameter.  The convolution is linear, so it applies to
        each derivative image just as it does to the kernel itself, and the
        loss' own dependence on the model enters through d(chi)/d(model).

        Parameters
        ----------
        params : sequence of float
            Packed parameters, as for `jac`.
        data : array of float.  Shape: (nstar, npix, npix)
            Images against which chi is computed.
        vars : sequence of array or sequence of float
            Sky variances, as for `jac`.

        Returns
        -------
        array of float, shape (nstar*npix**2, nwavefront)
            Jacobian columns for the wavefront parameters.
        """
        npix = self.npix
        pdict = self.unpack_params(params)
        nwavefront = self._dzkdp.shape[2]
        out = np.empty((self.nstar*npix**2, nwavefront))

        if self.atm_mode == 'fwhm':
            Ixx = Iyy = (pdict['fwhm'] * self.KOLM_CONST) ** 2
            Ixy = 0.0
        else:
            Ixx, Ixy, Iyy = pdict['Ixx'], pdict['Ixy'], pdict['Iyy']
        z_fits = self._get_z_fits(pdict['wavefront_params'])

        for i in range(self.nstar):
            thx, thy = self.thxs[i], self.thys[i]
            x0, y0, dxdz, dydz = self._projections[(thx, thy)]
            zk = np.array(self.z_refs[i])
            zk[:len(z_fits[i])] += z_fits[i]
            head = zk[self._zk_reach]
            verts_px = np.column_stack([x0 + dxdz @ head, y0 + dydz @ head])
            dverts = np.stack(
                [dxdz @ self._dzkdp[i], dydz @ self._dzkdp[i]], axis=1
            )
            dopt = self.factory._image_derivs_from_mesh(
                self._meshes[(thx, thy)], verts_px, dverts, npix=npix
            )

            atm = self._atm_kernel(pdict['dxs'][i], pdict['dys'][i], Ixx, Ixy, Iyy)
            model = self._model(
                pdict['fluxes'][i], pdict['dxs'][i], pdict['dys'][i],
                Ixx, Ixy, Iyy, tuple(zk), tuple(pdict['bkgs'][i]),
                thx=thx, thy=thy,
            )
            dchi_dmodel = self._dchi_dmodel(data[i], model, vars[i])

            s = slice(i*npix**2, (i+1)*npix**2)
            for p in range(nwavefront):
                dmodel = pdict['fluxes'][i] * convolve(dopt[p], atm)
                out[s, p] = (dchi_dmodel * dmodel).ravel()
        return out

    def _dchi_dmodel(self, datum, model, var):
        """d(chi)/d(model), elementwise.

        The loss divides by an error that is itself a function of the model, so
        this is not simply -1/sqrt(var + model).  Derived by differentiating
        `danish.loss.systematic_loss`; `alpha` is recovered from the loss
        rather than stored, so a custom loss falls back to a difference
        quotient.

        Parameters
        ----------
        datum : array of float
            Observed image.
        model : array of float
            Model image.
        var : float or array of float
            Sky variance.

        Returns
        -------
        array of float
            d(chi)/d(model) at each pixel.
        """
        alpha = getattr(self.loss_fn, 'alpha', None)
        if alpha is None:
            # Unknown loss: perturb it rather than assume a functional form.
            h = 1e-6 * max(np.abs(model).max(), 1.0)
            return (
                self.loss_fn(datum, model + h, var)
                - self.loss_fn(datum, model - h, var)
            ) / (2*h)
        denom = var + model + (alpha*model)**2
        return (
            -1.0/np.sqrt(denom)
            - (datum - model)*(1.0 + 2*alpha**2*model)/(2*denom**1.5)
        )

    def jac(
        self, params, data, vars, method='analytic'
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
        method : str, optional
            'analytic' (default) differentiates the wavefront columns in
            closed form, which is both faster and more accurate than the
            alternative -- a one-sided difference at `wavefront_step` carries
            a few percent error.  'fd' forces the difference quotient for
            every column.  Falls back to 'fd' where no analytic path exists,
            i.e. for any factory that does not render from a mesh.

        Returns
        -------
        jac : array of float
            Jacobian array d(chi)/d(param).  First dimension is pixels, second
            dimension is param.

        Raises
        ------
        ValueError
            If `method` is neither 'analytic' nor 'fd'.
        """
        _check_jac_method(method)
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

        # Flux — perturb all stars at once; divide by per-star step in the loop
        dflux_eps = 1e-3
        param_dict = self.unpack_params(params)
        fluxes = np.array(param_dict["fluxes"])
        dflux_param_dict = dict(param_dict)
        dflux_param_dict["fluxes"] = fluxes * (1 + dflux_eps)
        chi_flux = self.chi(self.pack_params(**dflux_param_dict), data, vars)
        for i in range(nstar):
            s = slice(i*npix**2, (i+1)*npix**2)
            out[s, i] = (chi_flux[s] - chi0[s]) / (dflux_eps * fluxes[i])

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
        wf0 = 3*nstar + natm
        if method == 'analytic' and self._can_use_analytic():
            out[:, wf0:wf0+nwavefront] = self._wavefront_jac_analytic(
                params, data, vars
            )
        else:
            for i in range(wf0, wf0 + nwavefront):
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

    def jac_sparse(
        self, params, data, vars, format='csr', method='analytic'
    ):
        """Compute jacobian d(chi)/d(param) as a sparse matrix.

        Same scheme as `jac` -- same `method`, same steps, same perturbed
        `chi` evaluations in the same order -- so the two agree bit-for-bit.
        Only the storage differs; see `.jac` for the parameter ordering this
        relies on and for what `method` selects.

        Each star's `chi` residuals depend on that star's flux, dx, dy and
        background coefficients, on the shared atmospheric and wavefront
        parameters, and on nothing else.  Every row block therefore has the
        same `3 + natm + nwavefront + nbkg` nonzero columns out of
        `len(params)`, so the density is that ratio: 1.0 at nstar=1, falling to
        0.14 by nstar=40.

        Note `scipy.optimize.least_squares` will not accept a sparse jacobian
        with `tr_solver='exact'` or `method='lm'`.  Use `tr_solver='lsmr'`, or
        leave `tr_solver` as None, which selects lsmr for a sparse jacobian.

        Parameters
        ----------
        params : sequence of float
            Order is: (fluxes, dxs, dys, fwhm, *wavefront_params, *bkgs)
        data : array of float.  Shape: (nstar, npix, npix)
            Images against which to compute chi.
        vars : sequence of array (npix, npix) or sequence of float
            Variances of the sky only.  Do not include Poisson contribution of
            the signal, as this will be added from the current model.
        format : str, optional
            Sparse format to return, 'csr' (default) or 'csc'.
        method : str, optional
            Differentiation method, as for `jac`.

        Raises
        ------
        ValueError
            If `method` is neither 'analytic' nor 'fd'.

        Returns
        -------
        jac : scipy.sparse matrix
            Jacobian d(chi)/d(param), shape (nstar*npix**2, len(params)), in
            `format`.  Rows are pixels, columns are parameters, as for `jac`.
        """
        _check_jac_method(method)
        nstar = self.nstar
        npix = self.npix
        nbkg = self.nbkg
        natm = self.natm
        nwavefront = len(params) - nbkg*nstar - (3*nstar + natm)
        P = npix**2
        M = nstar*P

        chi0 = self.chi(params, data, vars)

        # star_cols[k, i] is the single column that parameter k of star i
        # contributes, restricted to star i's row block.  k = 0, 1, 2 is
        # flux, dx, dy.
        star_cols = np.empty((3, nstar, P))

        # Flux — perturb all stars at once; divide by per-star step in the loop
        dflux_eps = 1e-3
        param_dict = self.unpack_params(params)
        fluxes = np.array(param_dict["fluxes"])
        dflux_param_dict = dict(param_dict)
        dflux_param_dict["fluxes"] = fluxes * (1 + dflux_eps)
        chi_flux = self.chi(self.pack_params(**dflux_param_dict), data, vars)
        for i in range(nstar):
            s = slice(i*P, (i+1)*P)
            star_cols[0, i] = (chi_flux[s] - chi0[s]) / (dflux_eps * fluxes[i])

        # Repeat for dx
        dx = 0.01
        dx_params = np.array(params)
        dx_param_dict = self.unpack_params(dx_params)
        dx_param_dict["dxs"] += dx
        chi_dx = self.chi(self.pack_params(**dx_param_dict), data, vars)
        for i in range(nstar):
            s = slice(i*P, (i+1)*P)
            star_cols[1, i] = (chi_dx[s] - chi0[s]) / dx

        # Repeat for dy
        dy = 0.01
        dy_params = np.array(params)
        dy_param_dict = self.unpack_params(dy_params)
        dy_param_dict["dys"] += dy
        chi_dy = self.chi(self.pack_params(**dy_param_dict), data, vars)
        for i in range(nstar):
            s = slice(i*P, (i+1)*P)
            star_cols[2, i] = (chi_dy[s] - chi0[s]) / dy

        # Atmospheric and wavefront parameters — dense, shared across all stars
        dense = np.empty((M, natm + nwavefront))

        datm = 0.01
        for k in range(natm):
            params1 = np.array(params)
            params1[3*nstar + k] += datm
            chi1 = self.chi(params1, data, vars)
            dense[:, k] = (chi1 - chi0) / datm

        if method == 'analytic' and self._can_use_analytic():
            dense[:, natm:] = self._wavefront_jac_analytic(params, data, vars)
        else:
            for k in range(nwavefront):
                params1 = np.array(params)
                params1[3*nstar + natm + k] += self.wavefront_step
                chi1 = self.chi(params1, data, vars)
                dense[:, natm + k] = (chi1-chi0)/self.wavefront_step

        # Background terms.  These are sparse too
        bkg_cols = np.empty((nstar, nbkg, P))
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
                s = slice(j*P, (j+1)*P)
                bkg_cols[j, i] = (chi_bkg[s] - chi0[s]) / dbkg

        # Assemble.  The per-star blocks are already in CSC form: column i
        # holds rows i*P through (i+1)*P, contiguous and ascending, so the
        # indices and indptr are plain aranges rather than anything we have to
        # build.  hstack then fixes the column order to match pack_params, and
        # converts to `format`.
        indices = np.arange(M)
        indptr = np.arange(0, M + P, P)
        blocks = [
            csc_matrix((star_cols[k].ravel(), indices, indptr), shape=(M, nstar))
            for k in range(3)
        ]
        blocks.append(csc_matrix(dense))
        if nbkg > 0:
            # Column nbkg*j + i belongs to star j, so it holds rows j*P
            # through (j+1)*P.  Same aranges, repeated nbkg times per star.
            rows = np.repeat(np.arange(nstar), nbkg*P).reshape(nstar, nbkg, P)
            rows = (rows*P + np.arange(P)).ravel()
            blocks.append(
                csc_matrix(
                    (bkg_cols.ravel(), rows, np.arange(0, nstar*nbkg*P + P, P)),
                    shape=(M, nstar*nbkg)
                )
            )

        return hstack(blocks, format=format)

    def _jac2(self, params, data, vars):
        nstar = self.nstar
        npix = self.npix
        nbkg = self.nbkg
        natm = self.natm
        nwavefront = len(params) - nbkg*nstar - (3*nstar + natm)

        out = np.empty((nstar*npix**2, len(params)))

        chi0 = self.chi(params, data, vars)

        # Flux — use same f*(1+eps) formula as jac to ensure bit-identical results
        dflux_eps = 1e-3
        param_dict = self.unpack_params(params)
        fluxes = np.array(param_dict["fluxes"])
        for i in range(nstar):
            params1_dict = dict(param_dict)
            params1_dict["fluxes"] = np.array(fluxes)
            params1_dict["fluxes"][i] = fluxes[i] * (1 + dflux_eps)
            chi1 = self.chi(self.pack_params(**params1_dict), data, vars)
            out[:, i] = (chi1 - chi0) / (dflux_eps * fluxes[i])

        # All other params — direct additive perturbation
        step = [0.01]*nstar  # dx
        step += [0.01]*nstar  # dy
        step += [0.01]*natm   # atm params
        step += [self.wavefront_step]*nwavefront  # wavefront terms
        step += [0.01]*nbkg*nstar  # background terms

        for j, s in enumerate(step):
            i = nstar + j
            params1 = np.array(params)
            params1[i] += s
            chi1 = self.chi(params1, data, vars)
            out[:, i] = (chi1 - chi0) / s

        return out


class DZMultiDonutModel(BaseMultiDonutModel):
    """Multi donut model that uses double Zernike coefficients to parameterize the
    wavefront.

    Parameters
    ----------
    factory : DonutFactoryBase
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
        self._nwavefront = len(dz_terms)
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
    factory : DonutFactoryBase
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
        self._nwavefront = self.nmode
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
