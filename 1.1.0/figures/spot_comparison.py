"""Figure 1: spot-diagram comparison — batoid vs TA wavefront vs OPD wavefront.

Three columns share a single field angle.  Each column shows the
focal-plane positions of a hexapolar grid of rays:

  Left   — ground truth from batoid raytracing
  Center — danish TA wavefront  (eq. 1 applied to batoid.zernikeTA())
  Right  — OPD wavefront        (eq. 1 applied to batoid.zernike())

A good illustration requires a case where the TA and OPD predictions
diverge visibly; high-order aberrations or a strongly off-axis field
angle tend to show the largest differences.

Generates docs/figures/spot_comparison_light.png and
               docs/figures/spot_comparison_dark.png.
Run from the repo root:

    python docs/figures/spot_comparison.py
    python docs/figures/spot_comparison.py --quiver   # residual quiver mode
"""

import argparse
import numpy as np
import batoid
import danish
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Telescope and field-angle configuration
# ---------------------------------------------------------------------------
telescope = batoid.Optic.fromYaml("LSST_r.yaml")
telescope = telescope.withLocallyShiftedOptic("Detector", [0.0, 0.0, -1.5e-3])
wavelength = 620e-9

thx, thy = 0.0, 1.75   # degrees
jmax = 66

# ---------------------------------------------------------------------------
# Command-line arguments (parsed early so nrad is available at module level)
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--nrad", type=int, default=7, metavar="N",
    help="Radial ring count for hexapolar grid (default: 7)",
)
_args = _parser.parse_args()

nrad = _args.nrad
naz  = int(nrad * 2 * np.pi / (1 - 0.612))

# Start with a consistent set of pupil coordinates
px, py = danish.hexapolar(
    outer=4.18,
    inner=4.18*0.612,
    nrad=nrad,
    naz=naz,
)

# ---------------------------------------------------------------------------
# Zernike / wavefront coefficients
# ---------------------------------------------------------------------------

def get_zernike_ta(telescope):
    """TA Zernike coefficients via batoid.zernikeTA()."""
    return batoid.zernikeTA(
        telescope,
        np.deg2rad(thx), np.deg2rad(thy),
        wavelength,
        jmax=jmax,
        eps=0.612,
        focal_length=10.312,
        reference="chief",
        projection="gnomonic",
        nrad=nrad, naz=naz,
    ) * wavelength

def get_zernike_opd(telescope):
    """OPD Zernike coefficients via batoid.zernike()."""
    return batoid.zernike(
        telescope,
        np.deg2rad(thx), np.deg2rad(thy),
        wavelength,
        jmax=jmax,
        eps=0.612,
        nx=255,
        reference="chief",
        projection="gnomonic",
    ) * wavelength


# ---------------------------------------------------------------------------
# Focal-plane position functions
# (each returns (x, y) in focal-plane metres, centered on the chief ray)
# ---------------------------------------------------------------------------

def _factory():
    R_outer = 4.18
    return danish.DonutFactory(
        R_outer=R_outer, R_inner=R_outer * 0.612,
        focal_length=10.312,
        mask_params=danish.load_mask_params("RubinObsc.yaml"),
    )


def spots_batoid(telescope):
    """Ground-truth focal-plane positions from batoid raytracing.

    Returns
    -------
    x, y : array
        Focal-plane positions in metres relative to the chief-ray hit.
    vignetted : bool array (same length as px/py)
        True where the ray was vignetted.
    """
    rays = batoid.RayVector.fromStop(
        np.array(px), np.array(py),
        optic=telescope,
        theta_x=np.deg2rad(thx),
        theta_y=np.deg2rad(thy),
        wavelength=wavelength,
        projection="gnomonic",
    )
    telescope.trace(rays)
    cr = batoid.RayVector.fromStop(
        0.0, 0.0, optic=telescope,
        theta_x=np.deg2rad(thx),
        theta_y=np.deg2rad(thy),
        wavelength=wavelength,
        projection="gnomonic",
    )
    telescope.trace(cr)
    w = ~rays.vignetted
    return rays.x[w] - cr.x[0], rays.y[w] - cr.y[0], w


def spots_ta(telescope, mask=None):
    """Focal-plane positions from the TA wavefront (danish approach).

    Uses batoid.zernikeTA() and applies  x = -f dW/du,  y = -f dW/dv.

    Parameters
    ----------
    mask : bool array, optional
        If given, return positions only where mask is True (must align with
        the full px/py grid).  Used for quiver alignment with batoid.
    """
    factory = _factory()
    aberrations = get_zernike_ta(telescope)
    x, y, w = factory.spots(
        aberrations=aberrations,
        thx=np.deg2rad(thx), thy=np.deg2rad(thy),
        nrad=nrad, naz=naz,
    )
    if mask is not None:
        return x[mask], y[mask]
    return x[w], y[w]


def spots_opd(telescope, mask=None):
    """Focal-plane positions from the OPD wavefront (comparison only).

    Uses batoid.zernike() — less accurate than the TA approach.

    Parameters
    ----------
    mask : bool array, optional
        If given, return positions only where mask is True (must align with
        the full px/py grid).  Used for quiver alignment with batoid.
    """
    factory = _factory()
    aberrations = get_zernike_opd(telescope)
    x, y, w = factory.spots(
        aberrations=aberrations,
        thx=np.deg2rad(thx), thy=np.deg2rad(thy),
        nrad=nrad, naz=naz,
    )
    if mask is not None:
        return x[mask], y[mask]
    return x[w], y[w]


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
LIGHT = dict(
    bg     = '#FFFFFF',
    batoid = '#2255AA',   # ground truth — blue
    ta     = '#228833',   # TA wavefront — green
    opd    = '#CC2222',   # OPD wavefront — red
    axis   = '#CCCCCC',
    text   = '#333333',
)

DARK = dict(
    bg     = '#2e303e',
    batoid = '#5599ff',
    ta     = '#44cc66',
    opd    = '#ff6666',
    axis   = '#555568',
    text   = '#cccccc',
)

# ---------------------------------------------------------------------------
# Drawing function  (2 rows × 3 cols)
# ---------------------------------------------------------------------------

def draw_figure(c, output_path):
    fig = Figure(figsize=(10, 7), constrained_layout=True)
    fig.patch.set_facecolor(c['bg'])
    axes = fig.subplots(2, 3)

    # --- Compute all data upfront ---
    x_bat, y_bat, vignette_mask = spots_batoid(telescope)

    x_ta,  y_ta  = spots_ta(telescope)
    x_opd, y_opd = spots_opd(telescope)

    x_ta_q,  y_ta_q  = spots_ta(telescope,  mask=vignette_mask)
    x_opd_q, y_opd_q = spots_opd(telescope, mask=vignette_mask)

    dx_ta  = (x_ta_q  - x_bat) * 1e6   # m → µm
    dy_ta  = (y_ta_q  - y_bat) * 1e6
    dx_opd = (x_opd_q - x_bat) * 1e6
    dy_opd = (y_opd_q - y_bat) * 1e6

    rms_ta  = np.sqrt(np.mean(dx_ta**2  + dy_ta**2))
    rms_opd = np.sqrt(np.mean(dx_opd**2 + dy_opd**2))

    # Uniform plot half-range from the scatter data
    all_um = np.concatenate([x_bat, x_ta, x_opd, y_bat, y_ta, y_opd]) * 1e6
    half = np.abs(all_um).max() * 1.1

    # Fixed reference arrow lengths; arrow_scale is derived per panel.
    REF_UM = {'ta': 0.1, 'opd': 1.0}

    def style(ax):
        ax.set_facecolor(c['bg'])
        for spine in ax.spines.values():
            spine.set_edgecolor(c['axis'])
        ax.tick_params(colors=c['text'])

    # --- Top row: scatter plots ---
    for ax, title, color_key, x_um, y_um in [
        (axes[0, 0], 'batoid\n(ground truth)', 'batoid', x_bat * 1e6, y_bat * 1e6),
        (axes[0, 1], 'TA wavefront\n(danish)',  'ta',    x_ta  * 1e6, y_ta  * 1e6),
        (axes[0, 2], 'OPD wavefront',           'opd',   x_opd * 1e6, y_opd * 1e6),
    ]:
        style(ax)
        ax.set_title(title, color=c[color_key], fontsize=10)
        ax.scatter(x_um, y_um, s=5, color=c[color_key], lw=0, zorder=3)

    axes[0, 0].set_ylabel('y  (µm)', color=c['text'], fontsize=9)

    # --- Bottom row: quiver residuals (bottom-left hidden) ---
    axes[1, 0].set_visible(False)

    for ax, method, color_key, dx, dy, rms in [
        (axes[1, 1], 'TA',  'ta',  dx_ta,  dy_ta,  rms_ta),
        (axes[1, 2], 'OPD', 'opd', dx_opd, dy_opd, rms_opd),
    ]:
        style(ax)
        ax.set_title(f'{method} − batoid', color=c[color_key], fontsize=10)
        ax.set_xlabel('x  (µm)', color=c['text'], fontsize=9)
        # Faint background dots at batoid positions so the donut shape is
        # visible even when quiver arrows are very small (e.g. TA panel).
        ax.scatter(x_bat * 1e6, y_bat * 1e6,
                   s=3, color=c[color_key], lw=0, alpha=0.2, zorder=2)
        ref_um = REF_UM[color_key]
        arrow_scale = half * 0.20 / ref_um
        q = ax.quiver(
            x_bat * 1e6, y_bat * 1e6, dx * arrow_scale, dy * arrow_scale,
            color=c[color_key], angles='xy', scale_units='xy', scale=1,
            width=0.003, headwidth=4, headlength=4, zorder=3,
        )
        ax.quiverkey(
            q, 0.11, 0.06, ref_um * arrow_scale, f'{ref_um:g} µm',
            labelpos='E', coordinates='axes',
            color=c[color_key], labelcolor=c['text'],
            fontproperties={'size': 8},
        )
        worst = np.sqrt(dx**2 + dy**2).max()
        ax.text(
            0.97, 0.03, f'RMS = {rms:.2f} µm\nworst = {worst:.2f} µm',
            transform=ax.transAxes, ha='right', va='bottom',
            color=c['text'], fontsize=8,
        )

    axes[1, 1].set_ylabel('y  (µm)', color=c['text'], fontsize=9)

    # --- Set uniform limits and equal aspect on all visible axes ---
    for ax in axes.flat:
        if not ax.get_visible():
            continue
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal', adjustable='box')

    # xlabel on top row (tick labels shared with bottom row via equal limits,
    # but axes are independent so label each)
    for ax in axes[0]:
        ax.set_xlabel('x  (µm)', color=c['text'], fontsize=9)

    fig.savefig(output_path, dpi=150, facecolor=c['bg'])


# ---------------------------------------------------------------------------
# Generate both figures
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    draw_figure(LIGHT, "docs/figures/spot_comparison_light.png")
    draw_figure(DARK,  "docs/figures/spot_comparison_dark.png")
