"""Figure: TA wavefront residual vs. Zernike truncation order for LSST.

Shows the RMS residual between the transverse-aberration (TA) wavefront
prediction and full batoid raytracing as a function of jmax, evaluated at
triangular-number truncation orders from j=28 to j=78.

Generates docs/figures/ta_convergence_light.png and
           docs/figures/ta_convergence_dark.png.
Run from the repo root:

    (set -a && source .env && python docs/figures/ta_convergence.py)
"""

import numpy as np
import batoid
import danish
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Telescope and field-angle configuration  (mirrors spot_comparison.py)
# ---------------------------------------------------------------------------
telescope = batoid.Optic.fromYaml("LSST_r.yaml")
telescope = telescope.withLocallyShiftedOptic("Detector", [0.0, 0.0, -1.5e-3])
wavelength = 620e-9

thx, thy = 0.0, 1.75   # degrees

nrad = 7
naz  = int(nrad * 2 * np.pi / (1 - 0.612))

px, py = danish.hexapolar(
    outer=4.18,
    inner=4.18 * 0.612,
    nrad=nrad,
    naz=naz,
)

# ---------------------------------------------------------------------------
# Triangular numbers between 28 and 78 inclusive
# T(n) = n*(n+1)//2:  T(7)=28, T(8)=36, T(9)=45, T(10)=55, T(11)=66, T(12)=78
# ---------------------------------------------------------------------------
JMAX_VALUES = [28, 36, 45, 55, 66, 78]

# ---------------------------------------------------------------------------
# Ground truth: batoid raytracing
# ---------------------------------------------------------------------------

def spots_batoid():
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


def spots_ta_jmax(jmax, mask):
    """TA spots for a given jmax, filtered to the batoid vignette mask."""
    factory = danish.DonutFactory(
        R_outer=4.18, R_inner=4.18 * 0.612,
        focal_length=10.312,
        mask_params=danish.load_mask_params("RubinObsc.yaml"),
    )
    aberrations = batoid.zernikeTA(
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
    x, y, _ = factory.spots(
        aberrations=aberrations,
        thx=np.deg2rad(thx), thy=np.deg2rad(thy),
        nrad=nrad, naz=naz,
    )
    return x[mask], y[mask]


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
LIGHT = dict(bg='#FFFFFF', line='#2255AA', axis='#CCCCCC', text='#333333')
DARK  = dict(bg='#2e303e', line='#5599ff', axis='#555568', text='#cccccc')


# ---------------------------------------------------------------------------
# Draw figure
# ---------------------------------------------------------------------------

def draw_figure(c, output_path):
    x_bat, y_bat, mask = spots_batoid()

    rms_values = []
    for jmax in JMAX_VALUES:
        x_ta, y_ta = spots_ta_jmax(jmax, mask)
        dx = (x_ta - x_bat) * 1e6   # m → µm
        dy = (y_ta - y_bat) * 1e6
        rms_values.append(np.sqrt(np.mean(dx**2 + dy**2)))

    fig = Figure(figsize=(6, 4), constrained_layout=True)
    fig.patch.set_facecolor(c['bg'])
    ax = fig.subplots()
    ax.set_facecolor(c['bg'])
    for spine in ax.spines.values():
        spine.set_edgecolor(c['axis'])
    ax.tick_params(colors=c['text'])

    ax.plot(JMAX_VALUES, rms_values, color=c['line'], marker='o')
    ax.set_xlabel('$j_{\\rm max}$', color=c['text'])
    ax.set_ylabel('RMS residual  (µm)', color=c['text'])
    ax.set_title('TA vs. batoid: residual vs. Zernike order', color=c['text'])
    ax.set_xticks(JMAX_VALUES)

    fig.savefig(output_path, dpi=150, facecolor=c['bg'])


if __name__ == "__main__":
    draw_figure(LIGHT, "docs/figures/ta_convergence_light.png")
    draw_figure(DARK,  "docs/figures/ta_convergence_dark.png")
