"""Figure: TA vs OPD sensitivity matrix comparison.

For each of the ten rigid-body degrees of freedom of the Rubin telescope
(M2: despace, decenter x/y, tilt x/y; camera: despace, decenter x/y,
tilt x/y), computes the finite-difference Zernike sensitivity
dW/d(dof) with both the TA and OPD wavefront definitions and renders
them as a table.  Each cell shows "OPD (TA)" in nm per µm or µrad.
Agreement confirms that Rubin's OPD-based sensitivity matrix is
interchangeable with a TA-based one.

Generates docs/figures/sensitivity_comparison_light.png and
           docs/figures/sensitivity_comparison_dark.png.
Run from the repo root:

    (set -a && source .env && python docs/figures/sensitivity_comparison.py)
"""

import numpy as np
import batoid

# ---------------------------------------------------------------------------
# Telescope and field-angle configuration  (mirrors spot_comparison.py)
# ---------------------------------------------------------------------------
telescope = batoid.Optic.fromYaml("LSST_r.yaml")
wavelength = 620e-9
thx, thy = 0.0, np.deg2rad(1.67)

JMAX = 66
EPS  = 0.612
NX   = 255
NRAD = 7
NAZ  = int(NRAD * 2 * np.pi / (1 - EPS))

# ---------------------------------------------------------------------------
# Nominal wavefronts
# ---------------------------------------------------------------------------
def zta(tel):
    return batoid.zernikeTA(
        tel, thx, thy, wavelength,
        jmax=JMAX, eps=EPS, focal_length=10.312,
        reference="chief", projection="gnomonic",
        nrad=NRAD, naz=NAZ,
    ) * wavelength

def zopd(tel):
    return batoid.zernike(
        tel, thx, thy, wavelength,
        jmax=JMAX, eps=EPS, nx=NX,
        reference="chief", projection="gnomonic",
    ) * wavelength

nom_ta  = zta(telescope)
nom_opd = zopd(telescope)

# ---------------------------------------------------------------------------
# Degrees of freedom
# ---------------------------------------------------------------------------
D_SHIFT = 1e-6   # 1 µm for decenter / despace
D_TILT  = 1e-6   # 1 µrad for tilt

DOFS = [
    ("M2\ndespace",    lambda d: telescope.withLocallyShiftedOptic("M2", [0, 0, d]),      D_SHIFT, False),
    ("M2\ndecenter x", lambda d: telescope.withLocallyShiftedOptic("M2", [d, 0, 0]),      D_SHIFT, False),
    ("M2\ndecenter y", lambda d: telescope.withLocallyShiftedOptic("M2", [0, d, 0]),      D_SHIFT, False),
    ("M2\ntilt x",     lambda d: telescope.withLocallyRotatedOptic("M2", batoid.RotX(d)), D_TILT,  True),
    ("M2\ntilt y",     lambda d: telescope.withLocallyRotatedOptic("M2", batoid.RotY(d)), D_TILT,  True),
    ("Camera\ndespace",    lambda d: telescope.withLocallyShiftedOptic("LSSTCamera", [0, 0, d]),      D_SHIFT, False),
    ("Camera\ndecenter x", lambda d: telescope.withLocallyShiftedOptic("LSSTCamera", [d, 0, 0]),      D_SHIFT, False),
    ("Camera\ndecenter y", lambda d: telescope.withLocallyShiftedOptic("LSSTCamera", [0, d, 0]),      D_SHIFT, False),
    ("Camera\ntilt x",     lambda d: telescope.withLocallyRotatedOptic("LSSTCamera", batoid.RotX(d)), D_TILT,  True),
    ("Camera\ntilt y",     lambda d: telescope.withLocallyRotatedOptic("LSSTCamera", batoid.RotY(d)), D_TILT,  True),
]

# ---------------------------------------------------------------------------
# Compute sensitivities (finite difference)
# ---------------------------------------------------------------------------
sensitivities = []   # list of (label, s_ta, s_opd, is_tilt)
for label, perturb, delta, is_tilt in DOFS:
    tel_p = perturb(delta)
    s_ta  = (zta(tel_p)  - nom_ta)  / delta
    s_opd = (zopd(tel_p) - nom_opd) / delta
    sensitivities.append((label, s_ta, s_opd, is_tilt))

j_idx = np.arange(4, JMAX + 1)


# ---------------------------------------------------------------------------
# Render markdown table
# ---------------------------------------------------------------------------
THRESHOLD   = 0.5                          # nm/unit — hide near-zero entries
URAD_PER_ARCSEC = np.deg2rad(1/3600) / 1e-6  # ≈ 4.848


def make_markdown_table():
    scale = 1e3  # raw (m/m or m/rad) → nm/µm or nm/µrad

    row_labels = [label.replace('\n', ' ') for label, _, _, _ in sensitivities]
    all_ta  = np.array([s_ta[j_idx]  * scale for _, s_ta,  _, _ in sensitivities])
    all_opd = np.array([s_opd[j_idx] * scale for _, _,  s_opd, _ in sensitivities])

    # Convert tilt rows from nm/µrad → nm/arcsec
    tilt_rows = [i for i, (_, _, _, is_tilt) in enumerate(sensitivities) if is_tilt]
    all_ta[tilt_rows]  *= URAD_PER_ARCSEC
    all_opd[tilt_rows] *= URAD_PER_ARCSEC

    # Keep only Zernike terms where any DOF has significant sensitivity
    max_abs = np.maximum(np.abs(all_ta), np.abs(all_opd)).max(axis=0)
    sig    = max_abs > THRESHOLD
    sig_j  = j_idx[sig]
    sig_ta  = all_ta[:, sig]
    sig_opd = all_opd[:, sig]

    col_labels = [f"Z{j}" for j in sig_j]
    n_dof, n_col = sig_ta.shape

    header = "| DOF | " + " | ".join(col_labels) + " |"
    sep    = "| --- |" + "".join(" ---: |" for _ in col_labels)
    rows   = []
    for i in range(n_dof):
        cells = " | ".join(
            f"{sig_opd[i,k]:.2f} ({sig_ta[i,k]:.2f})" for k in range(n_col)
        )
        rows.append(f"| {row_labels[i]} | {cells} |")

    lines = [header, sep] + rows
    return "\n".join(lines)


if __name__ == "__main__":
    print(make_markdown_table())
