"""fit_mask_edges.py

Read an edges.asdf file produced by generate_mask_model.py and fit polynomial
obscuration models c(θ), R(θ) to each edge in a single nonlinear
least-squares pass.  Output is a YAML file in RubinObsc.yaml format
compatible with danish.load_mask_params().
"""

import sys

import asdf
import numpy as np
import yaml
from scipy.optimize import least_squares


# Rubin raytrace surface order — used to sort output YAML sections.
# Surfaces not listed here are appended at the end in their natural order.
_SURFACE_ORDER = [
    "M1", "M2", "CameraBody", "M3",
    "L1_entrance", "L1_exit",
    "L2_entrance", "L2_exit",
    "Filter_entrance", "Filter_exit",
    "L3_entrance", "L3_exit",
]

# Nominal outer pupil radius — M1 outer edge in pupil-plane metres.
_R_PUPIL_OUTER = 4.18

# Scale factor: M1 outer radius 4.18 m maps to 65 px (half of 130 px donut)
_PX_PER_MM = 65.0 / (4.18 * 1e3)


def fit_edge(xPupil, yPupil, thetas, azimuth_deg, deg):
    """Fit polynomial center(θ) and radius(θ) to edge-point data.

    Parameters
    ----------
    xPupil, yPupil : ndarray, shape (n_theta, n_pts)
        Interpolated pupil-plane coordinates of the edge circle.
    thetas : ndarray, shape (n_theta,)
        Field angles in degrees.
    azimuth_deg : float
        Azimuth angle of the sweep in degrees.  The circle center is
        constrained to lie along this direction.
    deg : int
        Polynomial degree for both center(θ) and radius(θ).

    Returns
    -------
    c_coeffs : ndarray, shape (deg+1,)  or None if no data
    r_coeffs : ndarray, shape (deg+1,)  or None if no data
    thetaMin, thetaMax : float
    rms_mm : float
    theta_pts : ndarray  — per-theta field angles where fit data exists
    center_pts : ndarray — per-theta median center estimate (azimuth frame)
    radius_pts : ndarray — per-theta median radius estimate
    """
    az = np.deg2rad(azimuth_deg)
    # Rotate to azimuth frame so center displacement is purely along x′
    xp = xPupil * np.cos(az) + yPupil * np.sin(az)   # (n_theta, n_pts)
    yp = -xPupil * np.sin(az) + yPupil * np.cos(az)

    n_theta, n_pts = xPupil.shape
    theta_rep = np.broadcast_to(thetas[:, None], (n_theta, n_pts))

    valid = np.isfinite(xp) & np.isfinite(yp)

    xp_v = xp[valid]
    yp_v = yp[valid]
    theta_v = theta_rep[valid]

    # Per-theta circle estimates for diagnostic plots.
    # Rearrange (x-c)²+y²=R² → r²=2c·x+(R²-c²), a linear regression of r²
    # on x. This gives unbiased (c,R) estimates even for partial arc crescents,
    # unlike median(x) or median(hypot) which are biased for off-centre arcs.
    theta_has_data = valid.any(axis=1)
    theta_pts_list, center_pts_list, radius_pts_list = [], [], []
    for i in np.where(theta_has_data)[0]:
        xp_i = xp[i, valid[i]]
        yp_i = yp[i, valid[i]]
        r2   = xp_i**2 + yp_i**2
        A    = np.stack([xp_i, np.ones_like(xp_i)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, r2, rcond=None)
        c_i = a / 2.0
        R_i = float(np.sqrt(max(0.0, b + c_i**2)))
        theta_pts_list.append(float(thetas[i]))
        center_pts_list.append(float(c_i))
        radius_pts_list.append(R_i)
    theta_pts  = np.array(theta_pts_list)
    center_pts = np.array(center_pts_list)
    radius_pts = np.array(radius_pts_list)

    _no_data = (None, None, np.nan, np.nan, np.nan,
                np.array([]), np.array([]), np.array([]))
    if xp_v.size == 0:
        return _no_data

    r0 = float(np.mean(np.hypot(xp_v, yp_v)))
    p0 = np.zeros(2 * (deg + 1))
    p0[-1] = r0  # constant term of radius poly (descending → last element)

    def residuals(params):
        c = np.polyval(params[:deg + 1], theta_v)
        R = np.polyval(params[deg + 1:], theta_v)
        return (xp_v - c) ** 2 + yp_v ** 2 - R ** 2

    result = least_squares(residuals, p0, method="lm")
    c_coeffs = result.x[:deg + 1]
    r_coeffs = result.x[deg + 1:]

    c_fit = np.polyval(c_coeffs, theta_v)
    R_fit = np.polyval(r_coeffs, theta_v)
    radial_resid = np.sqrt((xp_v - c_fit) ** 2 + yp_v ** 2) - R_fit
    rms_mm = float(np.sqrt(np.mean(radial_resid ** 2))) * 1e3

    idx = np.where(theta_has_data)[0]
    thetaMin = float(thetas[idx[0]])
    thetaMax = float(thetas[idx[-1]])

    return c_coeffs, r_coeffs, thetaMin, thetaMax, rms_mm, theta_pts, center_pts, radius_pts


def clipping_thetaMin(c_coeffs, r_coeffs, thetaMin_data, thetaMax, dtheta=0.005):
    """Return the first θ at which a clear=True edge actually clips the pupil.

    A clear=True (outer boundary) edge clips the pupil when any point on the
    outer pupil ring falls outside the fitted circle, i.e. when

        R(θ) < |c(θ)| + R_PUPIL_OUTER

    The search starts 5 * dtheta below thetaMin_data so the polynomial can
    find the true zero-crossing even when min_arc_points has trimmed a few
    theta bins from the onset region.  Tying the margin to the actual step
    size prevents wild extrapolation when arc data is geometrically absent
    (e.g. M2.outer whose arc lives outside pupil_max at low theta).

    Returns the updated thetaMin, or thetaMax+1 if the edge never clips.
    """
    search_start = max(0.0, thetaMin_data - 5 * dtheta)
    theta_fine = np.linspace(search_start, thetaMax, 10_000)
    R = np.polyval(r_coeffs, theta_fine)
    c = np.polyval(c_coeffs, theta_fine)

    clips = R < np.abs(c) + _R_PUPIL_OUTER
    if clips.any():
        return float(theta_fine[np.argmax(clips)])
    return float(thetaMax) + 1.0   # edge never clips — disable it


def _fmt_coeffs(coeffs):
    """Format coefficient list for YAML at full float64 precision."""
    return "[" + ", ".join(f"{v:.17e}" for v in coeffs) + "]"


def save_diagnostics(path, diag_data):
    """Save per-edge center(θ) and radius(θ) diagnostic plots to a PDF.

    Parameters
    ----------
    path : str
        Output PDF path.
    diag_data : dict
        Keyed by surface name (in raytrace order).  Each value is a list of
        dicts with keys: edge_name, clear, thetaMin, thetaMax, c_coeffs,
        r_coeffs, theta_pts, center_pts, radius_pts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for surf, edges in diag_data.items():
            if not edges:
                continue
            n_edges = len(edges)
            fig, axes = plt.subplots(n_edges, 2,
                                     figsize=(10, 3.5 * n_edges),
                                     squeeze=False)
            fig.suptitle(surf, fontsize=12, fontweight="bold")

            for row, e in enumerate(edges):
                tmin, tmax = e["thetaMin"], e["thetaMax"]
                th_fine = np.linspace(
                    min(tmin, e["theta_pts"][0]) if e["theta_pts"].size else tmin,
                    tmax, 500)
                label_str = (f"{e['edge_name']}  clear={e['clear']}  "
                             f"θ=[{tmin:.3f}, {tmax:.3f}]  "
                             f"rms={e['rms_mm']:.3f} mm")

                for col, (qty, pts, coeffs, ylabel) in enumerate([
                    ("center", e["center_pts"], e["c_coeffs"], "center (m)"),
                    ("radius", e["radius_pts"], e["r_coeffs"], "radius (m)"),
                ]):
                    ax = axes[row, col]
                    if e["theta_pts"].size:
                        ax.scatter(e["theta_pts"], pts, s=6, c="steelblue",
                                   zorder=3, label="data (per-θ circle fit)")
                    ax.plot(th_fine, np.polyval(coeffs, th_fine),
                            "r-", lw=1.5, label="polynomial")
                    ax.axvline(tmin, color="0.4", ls="--", lw=0.8,
                               label=f"thetaMin={tmin:.3f}°")
                    ax.axvline(tmax, color="0.6", ls=":",  lw=0.8)
                    ax.set_xlabel("θ (deg)")
                    ax.set_ylabel(ylabel)
                    ax.set_title(f"{label_str} — {qty}", fontsize=8)
                    ax.legend(fontsize=7)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote diagnostics to {path}", file=sys.stderr)


def main(args):
    with asdf.open(args.input) as af:
        meta = af["meta"]
        thetas = np.array(af["thetas"])
        edges_tree = af["edges"]
        azimuth_deg = float(meta["azimuth_deg"])

        dtheta = float(np.median(np.diff(thetas))) if len(thetas) > 1 else 0.005

        results  = {}   # fitted data for YAML output
        diag_data = {}  # raw + fitted data for diagnostic plots

        for surf, edges in edges_tree.items():
            results[surf]   = {}
            diag_data[surf] = []
            n_edges = len(edges)
            for edge_name, edata in edges.items():
                # Translate legacy "edge0"/"edge1" names from older ASdfs.
                if edge_name == "edge0":
                    edge_name = "outer" if n_edges == 1 else "inner"
                elif edge_name == "edge1":
                    edge_name = "outer"
                xPupil = np.array(edata["xPupil"])  # (n_theta, n_pts)
                yPupil = np.array(edata["yPupil"])
                clear  = bool(edata.get("clear", False))
                (c_coeffs, r_coeffs, thetaMin, thetaMax, rms_mm,
                 theta_pts, center_pts, radius_pts) = fit_edge(
                    xPupil, yPupil, thetas, azimuth_deg, args.deg,
                )
                label = f"{surf}/{edge_name}"
                if c_coeffs is None:
                    print(f"  {label:<22}  no valid data, skipping",
                          file=sys.stderr)
                    continue
                # For clear=True (outer boundary) edges, recompute thetaMin
                # from the polynomial — the data-derived value is the first
                # theta at which arc points appear, not when clipping begins.
                if clear:
                    thetaMin = clipping_thetaMin(c_coeffs, r_coeffs,
                                                 thetaMin, thetaMax, dtheta)
                if thetaMin > thetaMax:
                    print(f"  {label:<22}  never clips pupil, skipping",
                          file=sys.stderr)
                    continue
                # Round thetaMin down, thetaMax up to 3 decimal places
                thetaMin = float(np.floor(thetaMin * 1000) / 1000)
                thetaMax = float(np.ceil(thetaMax * 1000) / 1000)
                results[surf][edge_name] = {
                    "clear": clear,
                    "thetaMin": thetaMin,
                    "thetaMax": thetaMax,
                    "center": c_coeffs,
                    "radius": r_coeffs,
                    "rms_mm": rms_mm,
                }
                diag_data[surf].append({
                    "edge_name":  edge_name,
                    "clear":      clear,
                    "thetaMin":   thetaMin,
                    "thetaMax":   thetaMax,
                    "c_coeffs":   c_coeffs,
                    "r_coeffs":   r_coeffs,
                    "rms_mm":     rms_mm,
                    "theta_pts":  theta_pts,
                    "center_pts": center_pts,
                    "radius_pts": radius_pts,
                })
                rms_px = rms_mm * _PX_PER_MM
                print(
                    f"  {label:<22}  clear={str(clear):<5}  "
                    f"θ=[{thetaMin:5.3f}, {thetaMax:5.3f}]  "
                    f"rms={rms_mm:7.4f} mm  ({rms_px:.4f} px)",
                    file=sys.stderr,
                )

    # Write YAML manually to control float precision — surfaces in raytrace order
    def _surf_key(name):
        try:
            return _SURFACE_ORDER.index(name)
        except ValueError:
            return len(_SURFACE_ORDER)

    lines = []
    for surf, edges in sorted(results.items(), key=lambda kv: _surf_key(kv[0])):
        if not edges:
            continue
        lines.append(f"{surf}:")
        for edge_name, d in edges.items():
            lines.append(f"  {edge_name}:")
            lines.append(f"    clear: {str(d['clear']).lower()}")
            lines.append(f"    thetaMin: {d['thetaMin']:.3f}")
            lines.append(f"    thetaMax: {d['thetaMax']:.3f}")
            lines.append(f"    center: {_fmt_coeffs(d['center'])}")
            lines.append(f"    radius: {_fmt_coeffs(d['radius'])}")

    # Append Spider_3D section from the reference RubinObsc.yaml
    import danish
    ref_path = danish.datadir + "/RubinObsc.yaml"
    with open(ref_path) as f:
        ref = yaml.safe_load(f)
    if "Spider_3D" in ref:
        lines.append(yaml.dump({"Spider_3D": ref["Spider_3D"]},
                               default_flow_style=None).rstrip())

    yaml_text = "\n".join(lines) + "\n"
    with open(args.output, "w") as f:
        f.write(yaml_text)
    print(f"Wrote {args.output}", file=sys.stderr)

    if args.diagnostics:
        # Sort diag_data in raytrace order
        diag_data_sorted = {
            surf: diag_data[surf]
            for surf in sorted(diag_data, key=_surf_key)
            if diag_data[surf]
        }
        save_diagnostics(args.diagnostics, diag_data_sorted)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit polynomial obscuration model from edges.asdf."
    )
    parser.add_argument(
        "input", type=str,
        help="Input ASDF file from generate_mask_model.py",
    )
    parser.add_argument(
        "--output", type=str, default="RubinObsc_fitted.yaml",
        help="Output YAML file (default: RubinObsc_fitted.yaml)",
    )
    parser.add_argument(
        "--deg", type=int, default=3,
        help="Polynomial degree for center(θ) and radius(θ) (default: 3)",
    )
    parser.add_argument(
        "--diagnostics", type=str, default=None,
        help="Save center/radius polynomial diagnostic plots to this PDF",
    )
    args = parser.parse_args()
    main(args)
