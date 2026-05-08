# Fitting details

This page describes the parameterization and fitting machinery built on top
of the [optical model](index.md).

## Wavefront parameterization

### Single-donut fits

For a single donut, the wavefront is a standard annular Zernike series
$W(u,v) = \sum_j a_j Z_j(u/R, v/R)$, and the fitted parameters are the
individual Zernike coefficients $a_j$.

### Multi-sensor fits: double Zernikes

When fitting multiple donuts simultaneously across the focal plane, the
wavefront varies with field angle $(\theta_x, \theta_y)$.  This field
dependence is modelled as a *double Zernike* (DZ) expansion,

$$
W(u,v;\theta_x,\theta_y) = \sum_{kj} c_{kj}\, Z_k(\theta)\, Z_j(\rho),
$$

where $Z_k(\theta)$ is a Zernike polynomial in field coordinates and
$Z_j(\rho)$ is one in pupil coordinates.  `DZMultiDonutModel` fits the
coefficients $c_{kj}$ directly.

### Sensitivity matrix parameterization

`DZBasisMultiDonutModel` instead fits amplitudes of a user-supplied
sensitivity matrix $S_{mkj}$ (e.g., derived from a detailed optical model),
which expresses the wavefront as a linear combination of double Zernike
basis functions:

$$
W = \sum_m b_m \sum_{kj} S_{mkj}\, Z_k(\theta)\, Z_j(\rho).
$$

Figure 2 illustrates how the accuracy of the TA wavefront approach improves
as the Zernike order of the decomposition increases, using quiver plots of
the residual deviation from real batoid raytracing.

<!-- TODO: insert Figure 2 (docs/figures/ta_convergence.png) once the
     figure script is complete. -->

## Atmospheric PSF

The optical donut image is convolved with a Kolmogorov atmospheric kernel
parameterized by FWHM (or second-moment tensor $I_{xx}, I_{xy}, I_{yy}$).
The convolution is performed in pixel space via `cv2.filter2D` (or
`scipy.signal.fftconvolve` as a fallback).

## Loss function

The default loss is the standard chi residual

$$
\chi_i = \frac{d_i - m_i}{\sqrt{\sigma_i^2 + m_i}},
$$

where $\sigma_i^2$ is the read-noise variance and $m_i$ is the model value
(Poisson shot noise).  The `systematic_loss` variant adds a fractional noise
floor $(\alpha \, m_i)^2$ to the denominator, capping the per-pixel SNR and
down-weighting pixels where unmodeled correlated residuals dominate.
