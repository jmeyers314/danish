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


class DoubleZernike:
    def __init__(self, coefs, field_radius=1.0):
        """
        Parameters
        ----------
        coefs : array of float
            Double Zernike coefficients.
        field_radius : float
            Outer field radius to use to normalize coefficients.
        """
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
            fiefield_radiusldRad=self.field_radius
        )

    @property
    def jmax(self):
        return self.coefs.shape[1]-1

    @property
    def kmax(self):
        return self.coefs.shape[0]-1


class MultiDonutModel:
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
        Single Zernike reference coefficients for each donut.  First index is
        donut, second index is pupil Zernike coefficient.
    field_radius : float
        Field radius in radians.  Ignored if dz_ref is provided.
    dz_terms : sequence of (int, int)
        Which double Zernike coefficients to include in the fit.
    thxs, thys : float
        Field angle in radians.
    N : int
        Size of image
    seed : int
        Random seed for use when creating noisy donut images with this class.
    """
    def __init__(
        self,
        factory,
        dz_ref=None,
        z_refs=None,
        field_radius=None,
        dz_terms=(),
        thxs=None, thys=None,
        N=90,
        seed=577215
    ):
        self.factory = factory
        # arcseconds per pixel
        # As with SingleDonutModel, we only use the scale here for the
        # atmospheric part, and the kernel is isotropic in pixel coordinates,
        # ignoring distortion.
        self.sky_scale = (
            3600*np.rad2deg(1/factory.focal_length)*factory.pixel_scale
        )

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
        self.N = N
        self.gsrng = galsim.BaseDeviate(seed)
        self.nstar = len(thxs)
        self.jmax_fit = max((dz_term[1] for dz_term in dz_terms), default=0)
        self.kmax_fit = max((dz_term[0] for dz_term in dz_terms), default=0)
        self.npix = 2*self.N+1

    @lru_cache(maxsize=1000)
    def _atm1(self, dx, dy, fwhm):
        fwhm = np.clip(fwhm, 0.1, 2.0)
        obj = galsim.Kolmogorov(fwhm=fwhm).shift(dx, dy)
        return obj.drawImage(nx=self.N, ny=self.N, scale=self.sky_scale).array

    @lru_cache(maxsize=1000)
    def _opt1(self, aberrations, thx, thy):
        return self.factory.image(
            aberrations=aberrations, thx=thx, thy=thy, N=self.N
        )

    def _model1(
        self,
        dx, dy, fwhm, aberrations,
        thx, thy,
        sky_level=None, flux=None
    ):
        atm = self._atm1(dx, dy, fwhm)
        opt = self._opt1(tuple(aberrations), thx, thy)
        arr = cv2.filter2D(opt, -1, atm, borderType=cv2.BORDER_CONSTANT)
        img = galsim.Image(arr)  # Does this make a copy?
        if flux is not None:
            img.array[:] *= flux/np.sum(img.array)
        if sky_level is not None:
            pn = galsim.PoissonNoise(self.gsrng, sky_level=sky_level)
            img.addNoise(pn)
        return img.array

    def _dz(self, dz_fit):
        dzarr = np.zeros((self.kmax_fit+1, self.jmax_fit+1))
        for i, zterm in enumerate(self.dz_terms):
            dzarr[zterm] = dz_fit[i]
        dz = DoubleZernike(dzarr, field_radius=self.field_radius)
        return dz

    def model(
        self, dxs, dys, fwhm, dz_fit, sky_levels=None, fluxes=None
    ):
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
                sky_levels[i], fluxes[i]
            )
        return out

    def unpack_params(self, params):
        dxs = params[:self.nstar]
        dys = params[self.nstar:2*self.nstar]
        fwhm = params[2*self.nstar]
        dz_fit = params[2*self.nstar+1:]
        return dxs, dys, fwhm, dz_fit

    def chi(
        self, params, data, sky_levels=None
    ):
        dxs, dys, fwhm, dz_fit = self.unpack_params(params)
        mods = self.model(dxs, dys, fwhm, dz_fit)
        chis = np.empty((self.nstar, self.npix, self.npix))
        for i, (mod, datum) in enumerate(zip(mods, data)):
            mod *= np.sum(datum)/np.sum(mod)
            chis[i] = (datum-mod)/np.sqrt(sky_levels[i] + mod)
        return chis.ravel()

    def _chi1(self, dx, dy, fwhm, aberrations, thx, thy, datum, sky_level):
        mod1 = self._model1(dx, dy, fwhm, aberrations, thx, thy)
        mod1 *= np.sum(datum)/np.sum(mod1)
        return ((datum-mod1)/np.sqrt(sky_level+mod1)).ravel()

    def jac(
        self, params, data, sky_levels=None
    ):
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
                thx, thy, data[i], sky_levels[i]
            )
            cx = self._chi1(
                dxs[i]+0.01, dys[i], fwhm,
                aberrations,
                thx, thy, data[i], sky_levels[i]
            )
            cy = self._chi1(
                dxs[i], dys[i]+0.01, fwhm,
                aberrations,
                thx, thy, data[i], sky_levels[i]
            )

            out[s, i] = (cx-c0)/0.01
            out[s, i+nstar] = (cy-c0)/0.01
            chi0[s] = c0

        # FWHM
        params1 = np.array(params)
        params1[2*nstar] += 0.001
        chi1 = self.chi(params1, data, sky_levels)
        out[:, 2*nstar] = (chi1-chi0)/0.001

        # DZ terms
        for i in range(2*nstar+1, len(params)):
            params1 = np.array(params)
            params1[i] += 1e-8
            chi1 = self.chi(params1, data, sky_levels)
            out[:, i] = (chi1-chi0)/1e-8

        return out
