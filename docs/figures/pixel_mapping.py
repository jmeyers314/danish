"""Figure: focal-plane pixels map to pupil-plane quadrilaterals.

The left panel shows a coarse grid of focal-plane pixels colored by
surface brightness.  The right panel shows the same pixels projected
back into the pupil plane: each square pixel becomes an irregular
quadrilateral whose area is proportional to how much pupil light it
collects — directly illustrating why surface brightness equals the
inverse Jacobian determinant.

Generates docs/figures/pixel_mapping_light.png and
           docs/figures/pixel_mapping_dark.png.
Run from the repo root:

    (set -a && source .env && python docs/figures/pixel_mapping.py)
"""

import numpy as np
import batoid
import danish
import galsim
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# ---------------------------------------------------------------------------
# Telescope and field-angle configuration  (mirrors spot_comparison.py)
# ---------------------------------------------------------------------------
telescope = batoid.Optic.fromYaml("LSST_r.yaml")
telescope = telescope.withLocallyShiftedOptic("Detector", [0.0, 0.0, -1.5e-3])
wavelength = 620e-9

thx, thy = 0.0, 0.0   # field center — no off-axis vignetting

R_outer = 4.18
R_inner = R_outer * 0.612
focal_length = 10.312

# Coarse pixels so individual cells are legible in the figure
pixel_scale = 50e-6   # 5× real LSST pixel scale
npix = 37

# ---------------------------------------------------------------------------
# Zernike TA aberrations from batoid, plus extra terms for visual drama
# ---------------------------------------------------------------------------
aberrations = batoid.zernikeTA(
    telescope,
    np.deg2rad(thx), np.deg2rad(thy),
    wavelength,
    jmax=66,
    eps=0.612,
    focal_length=focal_length,
    reference="chief",
    projection="gnomonic",
) * wavelength

# Add aberrations so the quadrilateral sizes vary noticeably
aberrations[6]  += 2.5e-6   # Z6  vertical astigmatism, 2.5 µm
aberrations[8]  += 1.5e-6   # Z8  vertical coma, 1.5 µm

# ---------------------------------------------------------------------------
# Surface-brightness image
# ---------------------------------------------------------------------------
factory = danish.DonutFactory(
    R_outer=R_outer, R_inner=R_inner,
    focal_length=focal_length,
    mask_params=danish.load_mask_params("RubinObsc.yaml"),
    pixel_scale=pixel_scale,
)
image = factory.image(
    aberrations=aberrations,
    thx=np.deg2rad(thx), thy=np.deg2rad(thy),
    npix=npix,
)

# ---------------------------------------------------------------------------
# Build polygon collections
# ---------------------------------------------------------------------------
Z = galsim.zernike.Zernike(aberrations, R_outer=R_outer, R_inner=R_inner)

no2 = (npix - 1) // 2
pix_idx = np.arange(-no2, no2 + 1)

def _focal_corners(ix, iy):
    """Corners of pixel (ix, iy) in focal plane, metres, shape (4, 2)."""
    cx, cy = ix * pixel_scale, iy * pixel_scale
    h = pixel_scale / 2
    return np.array([
        [cx - h, cy - h],
        [cx + h, cy - h],
        [cx + h, cy + h],
        [cx - h, cy + h],
    ])


def _pupil_corners(ix, iy):
    """Pixel corners mapped to pupil plane, shape (4, 2) or None."""
    fc = _focal_corners(ix, iy)
    u, v = danish.factory._focal_to_pupil(
        fc[:, 0], fc[:, 1], Z,
        focal_length=focal_length,
    )
    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        return None
    return np.column_stack([u, v])


focal_patches, pupil_patches, brightness = [], [], []

for j, iy in enumerate(pix_idx):
    for i, ix in enumerate(pix_idx):
        b = image[j, i]
        fc = _focal_corners(ix, iy) * 1e6   # → µm for display
        focal_patches.append(Polygon(fc, closed=True))

        pc = _pupil_corners(ix, iy)
        if pc is None:
            pupil_patches.append(Polygon(np.zeros((4, 2)), closed=True))
        else:
            pupil_patches.append(Polygon(pc, closed=True))
        brightness.append(b)

brightness = np.array(brightness)
vmax = np.percentile(brightness[brightness > 0], 98) if np.any(brightness > 0) else 1.0

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
LIGHT = dict(bg='#FFFFFF', axis='#CCCCCC', text='#333333', ring='#999999')
DARK  = dict(bg='#2e303e', axis='#555568', text='#cccccc', ring='#888899')


# ---------------------------------------------------------------------------
# Draw figure
# ---------------------------------------------------------------------------
def draw_figure(c, output_path):
    fig = Figure(figsize=(10, 5), constrained_layout=True)
    fig.patch.set_facecolor(c['bg'])
    ax_f, ax_p = fig.subplots(1, 2)

    for ax in (ax_f, ax_p):
        for spine in ax.spines.values():
            spine.set_edgecolor(c['axis'])
        ax.tick_params(colors=c['text'])
        ax.set_aspect('equal')
    ax_f.set_facecolor(c['bg'])
    ax_p.set_facecolor('black')   # matches inferno colormap zero value

    ax_f.set_title('Focal plane', color=c['text'])
    ax_p.set_title('Pupil plane', color=c['text'])
    ax_f.set_xlabel('x  (µm)', color=c['text'])
    ax_f.set_ylabel('y  (µm)', color=c['text'])
    ax_p.set_xlabel('u  (m)', color=c['text'])
    ax_p.set_ylabel('v  (m)', color=c['text'])

    kw = dict(cmap='inferno', linewidths=0.3, edgecolors='face')

    fp = PatchCollection(focal_patches, **kw)
    fp.set_array(brightness)
    fp.set_clim(0, vmax)
    ax_f.add_collection(fp)

    pp = PatchCollection(pupil_patches, **kw)
    pp.set_array(brightness)
    pp.set_clim(0, vmax)
    ax_p.add_collection(pp)

    # Annulus circles in pupil plane and projected onto focal plane
    theta = np.linspace(0, 2 * np.pi, 500)
    for r in (R_outer, R_inner):
        u_circ = r * np.cos(theta)
        v_circ = r * np.sin(theta)
        # Pupil panel
        ax_p.plot(u_circ, v_circ, color=c['ring'], lw=1, ls='--')
        # Focal panel: project circles through the wavefront
        x_circ, y_circ = danish.factory._pupil_to_focal(
            u_circ, v_circ, Z, focal_length=focal_length,
        )
        ax_f.plot(x_circ * 1e6, y_circ * 1e6, color=c['ring'], lw=1, ls='--')

    half = (no2 + 1) * pixel_scale * 1e6
    ax_f.set_xlim(-half, half)
    ax_f.set_ylim(-half, half)
    ax_p.set_xlim(-R_outer * 1.2, R_outer * 1.2)
    ax_p.set_ylim(-R_outer * 1.2, R_outer * 1.2)

    # Colorbar
    cb = fig.colorbar(fp, ax=[ax_f, ax_p], shrink=0.7, pad=0.02)
    cb.set_label('surface brightness  (arb.)', color=c['text'])
    cb.ax.yaxis.set_tick_params(color=c['text'])
    for lbl in cb.ax.yaxis.get_ticklabels():
        lbl.set_color(c['text'])

    fig.savefig(output_path, dpi=150, facecolor=c['bg'])


if __name__ == "__main__":
    draw_figure(LIGHT, "docs/figures/pixel_mapping_light.png")
    draw_figure(DARK,  "docs/figures/pixel_mapping_dark.png")
