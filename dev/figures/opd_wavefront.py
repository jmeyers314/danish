"""Figure illustrating the OPD wavefront definition.

Horizontal layout with a non-zero field angle:
  - Beam travels left → right
  - Pupil plane on the left (x = 0), focal plane on the right (x = L)
  - Ideal image point P₀ is off-axis on the focal plane
  - Chief ray runs from the pupil center (0, 0) to P₀ at the field angle,
    NOT along the optical axis
  - Reference sphere: arc centerd on P₀, radius R (schematic scale)
  - Wavefront W: irregular (amplitude-tapered oscillation), lagging and
    leading the sphere; matches the sphere exactly at the chief-ray angle
  - OPD Φ(u, v): signed gap between W and sphere along a marginal ray
  - Transverse aberration: focal-plane displacement of each ray from P₀,
    proportional to the wavefront gradient dW/du

Generates docs/figures/opd_wavefront_light.png and opd_wavefront_dark.png.
Run from the repo root:

    python docs/figures/opd_wavefront.py
"""

import numpy as np
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
L    = 8.0   # x-position of focal plane
y0   = 1.5   # y-offset of ideal image point P₀ (off-axis)
R    = 5.0   # reference sphere radius (schematic; chosen so arc sits mid-beam)
A    = 0.28  # OPD amplitude (exaggerated for visibility)
u_ext = 2.5  # pupil half-height

P0 = np.array([L, y0])

# Angle (rad) from P₀ looking toward pupil point (0, u)
def phi(u):
    return np.arctan2(u - y0, 0.0 - L)

phi_c = phi(0.0)   # chief-ray angle (P₀ → pupil center)

# Angular limits of the arc: cover the full pupil with a little padding.
# Wrap differences to (−π, π] to handle the ±π branch cut — phi(+u_ext)
# lands just below +π while phi_c lands just below −π.
def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi

phi_lo  = phi(+u_ext)   # toward upper pupil edge
phi_hi  = phi(-u_ext)   # toward lower pupil edge
half    = max(abs(_wrap(phi_lo - phi_c)), abs(_wrap(phi_hi - phi_c))) * 1.15
phi_arc = np.linspace(phi_c - half, phi_c + half, 300)

# Reference sphere arc
xs_sph = P0[0] + R * np.cos(phi_arc)
ys_sph = P0[1] + R * np.sin(phi_arc)

# OPD profile: one oscillation with a linearly tapered amplitude (so it looks
# less like a pure sine), zero at the chief-ray angle.
def opd(phi_val):
    t = _wrap(phi_val - phi_c) / half   # normalized angle in [-1, 1]
    return A * np.sin(2.0 * np.pi * t) * (1.0 + 0.35 * t)

opd_arc = opd(phi_arc)
xs_wf = P0[0] + (R + opd_arc) * np.cos(phi_arc)
ys_wf = P0[1] + (R + opd_arc) * np.sin(phi_arc)

# Highlighted marginal ray: pick t = +1/6 so OPD = A (maximum, lagging)
phi_hl = phi_c + half / 6.0
u_hl   = y0 - L * np.tan(phi_hl)   # pupil y-coordinate of this ray

sp_hl = P0 + R                * np.array([np.cos(phi_hl), np.sin(phi_hl)])
wf_hl = P0 + (R + opd(phi_hl)) * np.array([np.cos(phi_hl), np.sin(phi_hl)])
pu_hl = np.array([0.0, u_hl])

# Background rays (equally spaced across the pupil)
u_bg = np.linspace(-u_ext, u_ext, 27)

# Transverse aberration: proportional to the OPD wavefront gradient dW/du.
# Subtracting the pupil-mean gradient zeros the tip/tilt component so the
# cluster of focal-plane hits stays centerd on P₀.
def _dW_du(u, _eps=1e-4):
    """Derivative of the OPD w.r.t. pupil coordinate u (numerical)."""
    return (opd(phi(u + _eps)) - opd(phi(u - _eps))) / (2.0 * _eps)

_dW_mean = np.mean([_dW_du(u) for u in np.linspace(-u_ext, u_ext, 400)])
ta_scale = 0.55   # constant of proportionality; tuned for visual clarity

def ta_fp(u):
    """Schematic transverse aberration ∝ OPD wavefront gradient."""
    return ta_scale * (_dW_du(u) - _dW_mean)

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
LIGHT = dict(
    bg         = '#FFFFFF',
    sphere     = '#2255AA',
    wavefront  = '#CC2222',
    focal_pt   = '#000000',
    fplane     = '#888888',
    chief      = '#444444',
    ray_hl     = '#555555',
    ray_bg     = '#CCCCCC',
    axis       = '#AAAAAA',
    opd        = '#CC6600',
    ta         = '#007744',
    text       = '#333333',
    pupil      = '#888888',
)

DARK = dict(
    bg         = '#2e303e',
    sphere     = '#5599ff',
    wavefront  = '#ff6666',
    focal_pt   = '#eeeeee',
    fplane     = '#888888',
    chief      = '#bbbbbb',
    ray_hl     = '#999999',
    ray_bg     = '#454558',
    axis       = '#6a6c80',
    opd        = '#ffaa44',
    ta         = '#44cc88',
    text       = '#cccccc',
    pupil      = '#888888',
)

# ---------------------------------------------------------------------------
# Drawing function
# ---------------------------------------------------------------------------
def draw_figure(c, output_path):
    fig = Figure(figsize=(11, 7), constrained_layout=True)
    fig.patch.set_facecolor(c['bg'])
    ax = fig.subplots()
    ax.set_facecolor(c['bg'])

    # Optical axis (dotted horizontal)
    ax.plot([-0.5, L + 0.8], [0, 0],
            color=c['axis'], lw=0.8, ls=':', zorder=1)

    # Pupil plane (solid vertical bar)
    ax.plot([0, 0], [-u_ext * 1.15, u_ext * 1.15],
            color=c['pupil'], lw=2.0, zorder=4)

    # Focal plane (solid vertical bar)
    ax.plot([L, L], [-3.2, 3.2],
            color=c['fplane'], lw=2.0, zorder=4)

    # Background rays (pupil → focal plane, displaced by transverse aberration)
    for u in u_bg:
        y_fp = y0 + ta_fp(u)
        ax.plot([0, L], [u, y_fp], color=c['ray_bg'], lw=0.7, zorder=3)
        ax.plot(L, y_fp, '.', ms=2.5, color=c['ray_bg'], zorder=7)

    # Reference sphere arc
    ax.plot(xs_sph, ys_sph, color=c['sphere'], lw=2.0, zorder=5)

    # Wavefront arc
    ax.plot(xs_wf, ys_wf, color=c['wavefront'], lw=2.0, zorder=5)

    # Chief ray (more prominent than background rays)
    ax.plot([0, L], [0, y0], color=c['chief'], lw=1.4, zorder=6)

    # Highlighted marginal ray
    ax.plot([0, L], [u_hl, y0], color=c['ray_hl'], lw=1.0,
            ls='--', zorder=6)

    # Ideal image point P₀
    ax.plot(*P0, 'o', ms=5, color=c['focal_pt'], zorder=10)

    # Pupil center marker (chief ray origin)
    ax.plot(0, 0, 'o', ms=4, color=c['chief'], zorder=10)

    # Pupil point (u, v) marker
    ax.plot(0, u_hl, 's', ms=4, color=c['ray_hl'], zorder=10)

    # OPD double-headed arrow between sphere and wavefront
    ax.annotate('', xy=sp_hl, xytext=wf_hl,
                arrowprops=dict(arrowstyle='<->', color=c['opd'], lw=1.5),
                zorder=9)

    # OPD label: perpendicular offset from the midpoint
    mid     = (sp_hl + wf_hl) / 2.0
    ray_dir = (P0 - pu_hl) / np.linalg.norm(P0 - pu_hl)
    perp    = np.array([-ray_dir[1], ray_dir[0]])   # 90° CCW
    ax.text(*(mid + 0.35 * perp + np.array([0.30, -0.42])), r'$\Phi(u,v)$',
            color=c['opd'], fontsize=11, ha='left', va='center')

    # ---- Annotations ----
    fs = 10.5

    # P₀
    ax.annotate(r'$P_0$',
                xy=P0, xytext=(P0[0] + 0.5, P0[1] + 0.5),
                arrowprops=dict(arrowstyle='->', color=c['text'], lw=0.9),
                color=c['text'], fontsize=fs + 2, ha='left', va='bottom')

    # Transverse aberration: arrow from P₀ to the most-displaced background ray
    _ta_vals = [ta_fp(u) for u in u_bg]
    _i_ann   = int(np.argmax(np.abs(_ta_vals)))
    y_ta     = y0 + _ta_vals[_i_ann]
    ax.annotate('', xy=(L + 0.15, y0), xytext=(L + 0.15, y_ta),
                arrowprops=dict(arrowstyle='<->', color=c['ta'], lw=1.5),
                zorder=9)
    ax.text(L + 0.25, (y0 + y_ta) / 2, 'Transverse\naberration',
            color=c['ta'], fontsize=fs - 1, ha='left', va='center')

    # Focal plane
    ax.text(L + 0.12, 3.0, 'Focal\nplane',
            color=c['fplane'], fontsize=fs - 1, va='top', ha='left')

    # Pupil plane
    ax.text(-0.12, u_ext * 1.12, 'Pupil\nplane',
            color=c['pupil'], fontsize=fs - 1, va='top', ha='right')

    # Optical axis
    ax.text(L + 0.8, 0.12, 'Optical axis',
            color=c['axis'], fontsize=fs - 2, va='bottom', ha='right')

    # Reference sphere label (lower part of arc)
    i_slab = int(0.78 * len(phi_arc))
    ax.annotate('Reference\nsphere',
                xy=(xs_sph[i_slab], ys_sph[i_slab]),
                xytext=(xs_sph[i_slab] + 0.7, ys_sph[i_slab] - 0.6),
                arrowprops=dict(arrowstyle='->', color=c['sphere'], lw=0.9),
                color=c['sphere'], fontsize=fs, ha='left')

    # Wavefront label (upper part of arc)
    i_wlab = int(0.2 * len(phi_arc))
    ax.annotate(r'Wavefront $W$',
                xy=(xs_wf[i_wlab], ys_wf[i_wlab]),
                xytext=(xs_wf[i_wlab] - 0.5, ys_wf[i_wlab] + 0.7),
                arrowprops=dict(arrowstyle='->', color=c['wavefront'], lw=0.9),
                color=c['wavefront'], fontsize=fs, ha='center')

    # Pupil point (u, v)
    ax.text(-0.18, u_hl, r'$(u,v)$',
            color=c['text'], fontsize=fs, ha='right', va='center')

    # Chief ray label (along the ray, near midpoint)
    cx, cy = L * 0.45, y0 * 0.45
    ax.text(cx - 1.0, cy + 0.2 - 0.2, 'Chief ray',
            color=c['chief'], fontsize=fs - 1, ha='center', va='bottom',
            style='italic')

    ax.set_xlim(-1.8, L + 1.8)
    ax.set_ylim(-3.5, 3.8)
    ax.set_aspect('equal')
    ax.set_axis_off()

    fig.savefig(output_path, dpi=150, facecolor=c['bg'])


# ---------------------------------------------------------------------------
# Generate both figures
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    draw_figure(LIGHT, "docs/figures/opd_wavefront_light.png")
    draw_figure(DARK,  "docs/figures/opd_wavefront_dark.png")
