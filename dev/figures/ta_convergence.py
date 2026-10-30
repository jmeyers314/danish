"""Figure 2: TA wavefront convergence with Zernike order.

Generates docs/figures/ta_convergence.png.  Run from the repo root:

    python docs/figures/ta_convergence.py

Dependencies: batoid, danish, matplotlib, numpy

The figure shows quiver plots of the residual deviation between the
danish TA wavefront approach and real batoid raytracing, for increasing
Zernike orders j_max = 4, 10, 21, 36 (i.e., Zernike orders 2, 3, 4, 5...).
Each panel uses the same underlying aberration; as more Zernike terms are
included in the TA decomposition the arrows shrink, demonstrating convergence
to the true raytracing result.

This figure is placed after the Wavefront parameterization section in
docs/model/index.md because it references the concept of Zernike order.
"""

import numpy as np
import matplotlib.pyplot as plt
import batoid
import danish

# ---------------------------------------------------------------------------
# Telescope setup  (same as wavefront_comparison.py)
# ---------------------------------------------------------------------------
# TODO: load Rubin/LSST telescope from batoid.
# TODO: choose field angle (thx, thy).
# TODO: decide on aberration set — a realistic Rubin wavefront or a
#       constructed one with significant high-order content so that
#       convergence is clearly visible.

# ---------------------------------------------------------------------------
# Zernike orders to compare
# ---------------------------------------------------------------------------
# TODO: choose a set of j_max values, e.g. [4, 10, 21, 36] corresponding
# to Zernike orders 2, 3, 4, 5 (triangular numbers).
# For each j_max, compute TA coefficients truncated to that order:
#   zTA_jmax = batoid.zernikeTA(telescope, thx, thy, jmax=j_max, ...)

# ---------------------------------------------------------------------------
# Ground truth: batoid spot positions on a hexapolar grid
# ---------------------------------------------------------------------------
# TODO: trace a hexapolar pupil grid through batoid and record focal positions
# (x_true, y_true).  This is the reference for all panels.

# ---------------------------------------------------------------------------
# Per-order residuals
# ---------------------------------------------------------------------------
# TODO: for each j_max:
#   1. Call factory.spots(aberrations=zTA_jmax, ...) → (x_danish, y_danish, w)
#   2. Compute residuals: dx = x_danish - x_true, dy = y_danish - y_true
#   3. Store (pupil_u, pupil_v, dx, dy) for the quiver plot.

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
# TODO: grid of panels, one per j_max value.
# Suggested layout: 1 row × N columns (or 2×2 for four orders).
#
# fig, axes = plt.subplots(1, len(j_max_values), figsize=(4*len(j_max_values), 4))
# for ax, j_max, (u, v, dx, dy) in zip(axes, j_max_values, residuals):
#     ax.quiver(u, v, dx, dy)
#     ax.set_aspect("equal")
#     ax.set_title(f"$j_{{\\rm max}} = {j_max}$")
#     ax.set_xlabel("u (pupil)")
#     if ax is axes[0]:
#         ax.set_ylabel("v (pupil)")
# fig.suptitle("Deviation from real raytracing vs. Zernike order")
# fig.savefig("docs/figures/ta_convergence.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    raise NotImplementedError("Figure not yet implemented — see TODOs above")
