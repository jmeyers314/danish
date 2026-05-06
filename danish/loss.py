import numpy as np


def systematic_loss(alpha):
    """Return a loss function that adds a fractional systematic noise floor.

    The effective variance is ``var + model + (alpha * model)**2``, where
    ``alpha`` is the fractional systematic uncertainty (e.g. 0.02 = 2%).
    This caps the effective SNR per pixel, preventing unmodeled correlated
    residuals (e.g. from atmospheric fluctuations or mirror figure errors)
    from dominating the fit.

    Parameters
    ----------
    alpha : float
        Fractional systematic uncertainty.  Typical values are 0.01–0.05.

    Returns
    -------
    callable
        Loss function with signature ``(data, model, var) -> array``.
    """
    def _loss(data, model, var):
        return (data - model) / np.sqrt(var + model + (alpha * model)**2)
    return _loss


chi2_loss = systematic_loss(0)
"""Standard chi residual, equivalent to ``systematic_loss(0)``.

Assumes Gaussian read noise plus Poisson shot noise:
``(data - model) / sqrt(var + model)``.  This is the default loss function.
"""
