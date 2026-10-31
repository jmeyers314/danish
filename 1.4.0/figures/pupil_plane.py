"""Cartoon illustrating the pupil plane vs. primary mirror surface.

Generates docs/figures/pupil_plane_light.png and pupil_plane_dark.png.
Run from the repo root:

    python docs/figures/pupil_plane.py
"""

import batoid
import numpy as np
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Telescope geometry (computed once; shared across both figures)
# ---------------------------------------------------------------------------
telescope = batoid.Optic.fromYaml("LSST_r.yaml")

r_outer = 4.18   # M1 outer radius (m)

xs = np.linspace(-r_outer, r_outer, 500)
m1 = telescope["M1"].surface.sag(xs, 0.0)
m1rim = float(m1[0])  # z-height at mirror rim = pupil plane height
mirror_back = float(m1.min()) - 0.12  # back-face of mirror cross-section

# Ray direction (same for both figures)
vx, vz = 1.3, 0.9
ray_extend = 2.0

# Pre-trace rays so batoid is only called once
pupil_positions = np.linspace(-r_outer * 0.97, r_outer * 0.97, 28)
hit_x = []
hit_z = []
for px in pupil_positions:
    rays = batoid.RayVector(
        np.array([px]), np.array([0.0]), np.array([m1rim]),
        np.array([-vx]), np.array([0.0]), np.array([vz]),
    )
    batoid.intersect(telescope["M1"].surface, rays)
    hit_x.append(float(rays.x[0]))
    hit_z.append(float(rays.z[0]))


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
LIGHT = dict(
    bg          = '#FFFFFF',
    mirror_fill = '#BBBBBB',
    mirror_line = '#333333',
    pupil       = '#CC2222',
    left_ray    = '#CC3311',
    right_ray   = '#2255AA',
    axis        = '#888888',
    annotation  = '#333333',
)

# MkDocs Material slate scheme background ≈ #2e303e
DARK = dict(
    bg          = '#2e303e',
    mirror_fill = '#4a4a5c',
    mirror_line = '#cccccc',
    pupil       = '#ff6666',
    left_ray    = '#ff7744',
    right_ray   = '#5599ff',
    axis        = '#888888',
    annotation  = '#cccccc',
)


# ---------------------------------------------------------------------------
# Drawing function
# ---------------------------------------------------------------------------
def draw_figure(c, output_path):
    fig = Figure(figsize=(8, 5), constrained_layout=True)
    fig.patch.set_facecolor(c['bg'])
    ax = fig.subplots()
    ax.set_facecolor(c['bg'])

    # Mirror cross-section: filled body + surface outline
    mirror_xs = np.concatenate([xs, xs[::-1]])
    mirror_zs = np.concatenate([m1, np.full(len(xs), mirror_back)])
    ax.fill(mirror_xs, mirror_zs, color=c['mirror_fill'], lw=0, zorder=2)
    ax.plot(xs, m1, color=c['mirror_line'], lw=1.5, zorder=3)
    ax.plot(
        [xs[0], xs[0], xs[-1], xs[-1]],
        [m1rim, mirror_back, mirror_back, m1rim],
        color=c['mirror_line'], lw=1.5, zorder=3,
    )

    # Pupil plane: dashed line at the rim height
    ax.plot([xs[0], xs[-1]], [m1rim, m1rim],
            color=c['pupil'], lw=1.5, ls='--', zorder=6)

    # Rays
    for px, rx, rz in zip(pupil_positions, hit_x, hit_z):
        color = c['right_ray'] if rx >= 0 else c['left_ray']
        ax.plot(
            [px - vx * ray_extend, rx],
            [m1rim + vz * ray_extend, rz],
            color=color, lw=0.9, alpha=0.85, zorder=5,
        )
        ax.plot(
            [px, px], [m1rim - 0.05, m1rim + 0.05],
            color=color, lw=1.2, zorder=7,
        )

    # Optical axis (dotted)
    ax.plot([0, 0], [mirror_back, m1rim + vz * ray_extend],
            color=c['axis'], lw=0.7, ls=':', zorder=1)

    # Annotations
    fsize = 10.5
    ix = int(0.79 * len(xs))
    ref_px = r_outer * 0.62

    ax.annotate(
        'Pupil plane',
        xy=(xs[0] + 0.3, m1rim),
        xytext=(xs[0] - 0.75, m1rim + 0.45),
        arrowprops=dict(arrowstyle='->', color=c['pupil'], lw=1.0),
        color=c['pupil'], fontsize=fsize, ha='right', va='center',
    )
    ax.annotate(
        'Primary mirror',
        xy=(xs[ix], m1[ix]),
        xytext=(xs[ix] - 0.2, m1[ix] - 0.18),
        arrowprops=dict(arrowstyle='->', color=c['annotation'], lw=1.0),
        color=c['annotation'], fontsize=fsize, ha='left',
    )
    ax.annotate(
        'Incoming\nlight',
        xy=(ref_px - vx * 0.8, m1rim + vz * 0.8),
        xytext=(ref_px + 0.5, m1rim + vz * ray_extend - 0.1),
        arrowprops=dict(arrowstyle='->', color=c['right_ray'], lw=1.0),
        color=c['right_ray'], fontsize=fsize, ha='left',
    )

    ax.set_xlim(xs[0] - 2.8, xs[-1] + 2.8)
    ax.set_ylim(mirror_back - 0.15, m1rim + vz * ray_extend + 0.5)
    ax.set_axis_off()

    fig.savefig(output_path, dpi=150, facecolor=c['bg'])


# ---------------------------------------------------------------------------
# Generate both figures
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    draw_figure(LIGHT, "docs/figures/pupil_plane_light.png")
    draw_figure(DARK,  "docs/figures/pupil_plane_dark.png")
