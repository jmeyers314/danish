"""animate_pupil.py

Animate the illuminated pupil for two obscuration YAML files side by side,
sweeping field angle from 0 to thetaMax.

For each frame the pupil throughput is computed directly from the polynomial
model using the same subpixel `_enclosed_fraction` and `_strut_masked_fraction`
routines used by DonutFactory:
  - clear=True edges:  f = min(f, enclosed_fraction)    # keep inside
  - clear=False edges: f = min(f, 1 - enclosed_fraction) # keep outside
  - Spider_3D vanes:   f = min(f, 1 - strut_fraction)    # vane blocks light

The overlay panel colours by agreement:
  white  — both fully illuminated
  red    — yaml1 > yaml2
  blue   — yaml2 > yaml1
  black  — both dark

Usage:
    python animate_pupil.py FILE1 FILE2
        [--output movie.mp4]     default: pupil_compare.mp4
        [--nframes N]            number of theta steps (default: 100)
        [--dtheta-deg F]         theta step size in degrees (alternative to --nframes)
        [--npix N]               pixel grid size (default: 800)
        [--azimuth-deg DEG]      field-angle sweep direction (default: 45)
        [--spider-angle DEG]     spider rotator angle in degrees; omit to skip
        [--fps N]                frames per second (default: 15)
        [--dpi N]                output resolution in dpi (default: 200)
        [--thetaMax DEG]         override maximum field angle
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

import matplotlib
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import yaml

from danish.factory import (
    _project_spider_vane,
    _enclosed_fraction,
    _strut_masked_fraction,
)

R_OUTER = 4.18    # m  — M1 outer radius / nominal pupil boundary


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_throughput(data, theta, cos_az, sin_az, xx, yy, step, spider_angle=None):
    """Return float throughput array [0, 1] for a uniform pupil grid.

    Parameters
    ----------
    data : dict
        Loaded YAML mask params.
    theta : float
        Field angle in degrees.
    cos_az, sin_az : float
        Cosine and sine of the sweep azimuth.
    xx, yy : ndarray, shape (npix, npix)
        Pupil coordinate grids in meters.
    step : float
        Grid spacing in meters (= dudx = dvdy Jacobian entry).
    spider_angle : float or None
        Spider rotator angle in degrees; if None, Spider_3D is skipped.
    """
    n = xx.size
    u = np.ascontiguousarray(xx.ravel(), dtype=float)
    v = np.ascontiguousarray(yy.ravel(), dtype=float)
    dudx = np.full(n, step)
    dudy = np.zeros(n)
    dvdx = np.zeros(n)
    dvdy = np.full(n, step)

    f = np.ones(n)
    thr_deg = float(theta)

    for key, val in data.items():
        if key == "Spider_3D":
            if spider_angle is None:
                continue
            thx = np.deg2rad(theta) * cos_az
            thy = np.deg2rad(theta) * sin_az
            for vane in val:
                p1x, p1y, sth1, cth1, p2x, p2y, sth2, cth2 = _project_spider_vane(
                    vane["r0"], vane["v0"], vane["width"], vane["length"],
                    vane["angle"] + spider_angle,  # both in degrees
                    thx, thy,                      # radians
                )
                enc = _strut_masked_fraction(
                    u, v, u, v, vane["length"],
                    p1x, p1y, sth1, cth1,
                    p2x, p2y, sth2, cth2,
                    dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy,
                )
                f = np.minimum(f, 1 - enc)
        else:
            for edge, params in val.items():
                th_min = params.get("thetaMin", 0.0)
                th_max = params.get("thetaMax", 2.0)
                if thr_deg < th_min or thr_deg > th_max:
                    continue
                radius = np.polyval(params["radius"], thr_deg)
                center = np.polyval(params["center"], thr_deg)
                cx = center * cos_az
                cy = center * sin_az
                enc = _enclosed_fraction(
                    u, v, u, v, cx, cy, radius,
                    dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy,
                )
                if params["clear"]:
                    f = np.minimum(f, enc)
                else:
                    f = np.minimum(f, 1 - enc)

    return f.reshape(xx.shape)


def throughputs_to_rgb(f1, f2):
    """Encode two throughput maps as an RGB overlay image.

    white  — both = 1
    red    — f1 > f2
    blue   — f2 > f1
    black  — both = 0
    """
    avg = 0.5 * (f1 + f2)
    d = f1 - f2
    r = np.clip(avg + np.maximum(0,  d), 0, 1)
    g = np.clip(avg - np.abs(d),        0, 1)
    b = np.clip(avg + np.maximum(0, -d), 0, 1)
    return np.stack([r, g, b], axis=-1)


def label_from_path(p):
    return Path(p).stem


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file1", help="First YAML file")
    parser.add_argument("file2", help="Second YAML file")
    parser.add_argument("--output", default="pupil_compare.mp4")
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument("--nframes", type=int, default=None,
                             help="Number of theta steps (default: 100)")
    frame_group.add_argument("--dtheta-deg", type=float, default=None,
                             help="Theta step size in degrees (alternative to --nframes)")
    parser.add_argument("--npix", type=int, default=800)
    parser.add_argument("--azimuth-deg", type=float, default=45.0,
                        help="Field-angle sweep direction in degrees (default: 45)")
    parser.add_argument("--spider-angle", type=float, default=None,
                        help="Spider rotator angle in degrees (omit to skip spiders)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--thetaMax", type=float, default=None)
    parser.add_argument("--thetaMin", type=float, default=0.0)
    args = parser.parse_args()

    data1 = load(args.file1)
    data2 = load(args.file2)
    label1 = label_from_path(args.file1)
    label2 = label_from_path(args.file2)

    # Determine theta range
    if args.thetaMax is not None:
        theta_max = args.thetaMax
    else:
        theta_max = 0.0
        for data in (data1, data2):
            for key, val in data.items():
                if key == "Spider_3D":
                    continue
                for params in val.values():
                    theta_max = max(theta_max, params.get("thetaMax", 0.0))

    theta_min = args.thetaMin
    if args.dtheta_deg is not None:
        nframes = round((theta_max - theta_min) / args.dtheta_deg) + 1
    else:
        nframes = args.nframes if args.nframes is not None else 100
    thetas = np.linspace(theta_min, theta_max, nframes)

    # Pupil coordinate grid
    lim = R_OUTER * 1.05
    coords = np.linspace(-lim, lim, args.npix)
    step = coords[1] - coords[0]
    xx, yy = np.meshgrid(coords, coords)

    az = np.deg2rad(args.azimuth_deg)
    cos_az, sin_az = np.cos(az), np.sin(az)

    # --- Set up figure: [yaml1 | yaml2 | overlay] ---
    fig = Figure(figsize=(12, 4.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    fig.patch.set_facecolor("0.15")
    for ax in axes:
        ax.set_facecolor("0.12")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    extent = [-lim, lim, -lim, lim]

    # Initialise with first frame
    f1 = compute_throughput(data1, thetas[0], cos_az, sin_az, xx, yy, step,
                            args.spider_angle)
    f2 = compute_throughput(data2, thetas[0], cos_az, sin_az, xx, yy, step,
                            args.spider_angle)

    cmap_pupil = matplotlib.colormaps["gray"]
    im1 = axes[0].imshow(f1, origin="lower", extent=extent,
                         cmap=cmap_pupil, vmin=0, vmax=1, interpolation="nearest")
    im2 = axes[1].imshow(f2, origin="lower", extent=extent,
                         cmap=cmap_pupil, vmin=0, vmax=1, interpolation="nearest")
    im3 = axes[2].imshow(throughputs_to_rgb(f1, f2), origin="lower", extent=extent,
                         interpolation="nearest")

    axes[0].set_title(label1, color="white", fontsize=8)
    axes[1].set_title(label2, color="white", fontsize=8)
    axes[2].set_title("overlay  (red=1 only, blue=2 only)", color="white", fontsize=8)

    az_str = f"az = {args.azimuth_deg:.0f}°"
    theta_label = fig.text(0.5, 0.01, f"θ = {thetas[0]:.3f}°  ({az_str})",
                           ha="center", color="white", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out = args.output
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800)
    with writer.saving(fig, out, dpi=args.dpi):
        for theta in tqdm(thetas, desc="animate"):
            f1 = compute_throughput(data1, theta, cos_az, sin_az, xx, yy, step,
                                    args.spider_angle)
            f2 = compute_throughput(data2, theta, cos_az, sin_az, xx, yy, step,
                                    args.spider_angle)
            im1.set_data(f1)
            im2.set_data(f2)
            im3.set_data(throughputs_to_rgb(f1, f2))
            theta_label.set_text(f"θ = {theta:.3f}°  ({az_str})")
            writer.grab_frame()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
