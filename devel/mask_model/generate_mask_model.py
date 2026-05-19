"""generate_mask_model.py"""

import functools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import asdf
import astropy.units as u
import batoid
import danish
import numpy as np
from astropy.coordinates import Angle
from matplotlib.animation import FFMpegWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from StarSharp.models.fiducial import default_raytraced_model
from scipy.spatial import cKDTree
from tqdm import tqdm

# Per-worker globals set by _init_worker so the telescope is built once per
# process rather than pickled and sent with every task.
_telescope = None
_wavelength = None


def build_telescope(
    version: str, band: str, rtp_deg: float, camera_piston_mm: float = 0.0
) -> Tuple[batoid.CompoundOptic, float]:
    """Load v3.14 optic with RTP lookup corrections and CameraBody baffle.

    Uses StarSharp's default_raytraced_model (which loads the rtp_lookup
    automatically) and then calls build_telescope() to obtain the perturbed
    batoid optic.
    """

    model = default_raytraced_model(
        version=version,
        band=band,
        rtp=Angle(rtp_deg, unit=u.deg),
    )
    piston = camera_piston_mm * u.mm if camera_piston_mm != 0.0 else None
    tel = model.build_telescope(camera_piston=piston)

    # Insert CameraBody baffle — same as batoid/vignetting_sources.py
    l1_z = tel["L1_entrance"].coordSys.origin[2]
    camera_body = batoid.Baffle(
        batoid.Plane(),
        name="CameraBody",
        obscuration=batoid.ObscCircle(0.80469),
        coordSys=batoid.CoordSys(origin=[0, 0, l1_z + 0.1045]),
        parent="LSSTCamera",
    )
    tel = tel.withInsertedOptic(before="M3", item=camera_body)
    return tel, model.wavelength.to_value(u.m)


def _init_worker(version, band, rtp_deg, camera_piston_mm):
    """Build the telescope once per worker process."""
    global _telescope, _wavelength
    _telescope, _wavelength = build_telescope(version, band, rtp_deg, camera_piston_mm)


def _process_theta(theta, azimuth_deg, nrad, retrace_tol, write_movie):
    """Process one field angle.  Runs in a worker process.

    Returns a dict with:
      'theta'        : float
      'edge_data'    : {(surf, irad): {'r', 'clear', 'xPupil', 'yPupil'}}
      'retrace_data' : {(surf, irad): np.ndarray of passing retrace errors}
      'frame_surfs'  : list of per-surface dicts for movie rendering, or None
    """
    telescope = _telescope
    wavelength = _wavelength

    thx = np.deg2rad(theta) * np.cos(np.deg2rad(azimuth_deg))
    thy = np.deg2rad(theta) * np.sin(np.deg2rad(azimuth_deg))
    rays = batoid.RayVector.asPolar(
        telescope,
        theta_x=thx,
        theta_y=thy,
        wavelength=wavelength,
        nrad=nrad,
        naz=int(2 * np.pi * nrad / 0.61),
        outer=4.18 * 1.02,  # slightly beyond M1 edge so outer obscuration circles are bracketed
        inner=0.0,
    )
    tf = telescope.traceFull(rays.copy())
    epRays = rays.toCoordSys(telescope.stopSurface.coordSys)
    telescope.stopSurface.surface.intersect(epRays)

    edge_data = {}    # (surf, irad) -> {r, clear, xPupil, yPupil}
    retrace_data = {} # (surf, irad) -> ndarray of passing retrace errors
    frame_surfs = [] if write_movie else None

    was_vignetted = np.zeros(len(rays), dtype=bool)
    for surf, out in tf.items():
        if surf in ["Detector"]:
            continue
        out_rays = out["out"]
        now_vignetted = out_rays.vignetted
        not_vignetted = ~now_vignetted
        newly_vignetted = now_vignetted & ~was_vignetted

        obsc = telescope[surf].obscuration
        negated = isinstance(obsc, batoid.ObscNegation)
        if negated:
            obsc = obsc.original
        radii = []
        clears = []
        if isinstance(obsc, batoid.ObscCircle):
            radii.append(obsc.radius)
            clears.append(negated)  # negated circle → clear outer
        elif isinstance(obsc, batoid.ObscAnnulus):
            radii.append(obsc.inner)
            radii.append(obsc.outer)
            clears.append(not negated)   # inner: False for negated (M1/M2/M3)
            clears.append(negated)       # outer: True for negated (M1/M2/M3)

        # Build interpolator: optic-surface coords → pupil coords.
        # out_rays are in global frame; xOptic/yOptic are in local surface
        # frame, but for Rubin surfaces (z-translated only) x,y are the same.
        valid = np.isfinite(out_rays.x) & np.isfinite(out_rays.y)
        src = np.stack([out_rays.x[valid], out_rays.y[valid]], axis=1)
        tree = cKDTree(src)
        ep_src = np.stack([epRays.x[valid], epRays.y[valid]], axis=1)
        ep_tree = cKDTree(ep_src)

        surf_edges = []
        for irad, (r, clear) in enumerate(zip(radii, clears)):
            xOptic = r * np.cos(np.linspace(0, 2 * np.pi, 1000))
            yOptic = r * np.sin(np.linspace(0, 2 * np.pi, 1000))
            query = np.stack([xOptic, yOptic], axis=1)
            dists, nn_idx = tree.query(query, k=100)
            # Keep only neighbors within 10% of the edge radius in
            # optic-local space — replaces the old inconsistent
            # pupil-space median test.  10% ensures at least ~2 radial
            # ring spacings (outer/nrad) for all Rubin surfaces.
            keep = dists < 0.1 * r            # (n_query, k)
            px_nn = epRays.x[valid][nn_idx]   # (n_query, k)
            py_nn = epRays.y[valid][nn_idx]
            xPupil = np.full(len(query), np.nan)
            yPupil = np.full(len(query), np.nan)
            # Weighted linear regression centered on each query point.
            # Intercept (beta[:,2]) gives value at query point regardless
            # of whether neighbors are one-sided (unlike IDW).
            w = np.where(keep, 1.0 / np.maximum(dists, 1e-9) ** 2, 0.0)
            w_sum = w.sum(axis=1)
            valid_q = w_sum > 0
            dx = src[nn_idx, 0] - query[:, 0:1]   # (n_query, k)
            dy = src[nn_idx, 1] - query[:, 1:2]
            A = np.stack([dx, dy, np.ones_like(dx)], axis=-1)  # (n_query, k, 3)
            A_w = A * w[:, :, None]
            M    = np.einsum('nkj,nkl->njl', A_w, A)           # (n_query, 3, 3)
            rhs_x = np.einsum('nkj,nk->nj', A_w, px_nn)        # (n_query, 3)
            rhs_y = np.einsum('nkj,nk->nj', A_w, py_nn)
            vq = valid_q
            Mpinv = np.linalg.pinv(M[vq])   # SVD-based; stable for near-singular M
            beta_x = (Mpinv @ rhs_x[vq, :, None])[:, :, 0]
            beta_y = (Mpinv @ rhs_y[vq, :, None])[:, :, 0]
            xPupil[vq] = beta_x[:, 2]
            yPupil[vq] = beta_y[:, 2]
            # Retrace validation: for each inferred EP point, use WLS in
            # EP space to estimate the corresponding local surface coords,
            # then check the round-trip error.  Mirrors the forward WLS
            # (src→ep) but in reverse (ep_src→src).  Points whose error
            # exceeds retrace_tol * r, or that have no EP-space neighbors
            # within 0.1*r, are discarded.
            valid_j = np.where(np.isfinite(xPupil) & np.isfinite(yPupil))[0]
            if valid_j.size > 0:
                ep_query = np.stack([xPupil[valid_j], yPupil[valid_j]], axis=1)
                ret_dists, ret_nn = ep_tree.query(ep_query, k=10)
                ret_keep = ret_dists < 0.1 * r
                ret_w    = np.where(ret_keep,
                                    1.0 / np.maximum(ret_dists, 1e-9) ** 2,
                                    0.0)
                ret_w_sum = ret_w.sum(axis=1)
                ret_valid = ret_w_sum > 0
                # WLS regression: EP-space offsets → local surface coords
                dx_ep = ep_src[ret_nn, 0] - ep_query[:, 0:1]   # (n, k)
                dy_ep = ep_src[ret_nn, 1] - ep_query[:, 1:2]
                A_ret   = np.stack([dx_ep, dy_ep,
                                    np.ones_like(dx_ep)], axis=-1)  # (n, k, 3)
                A_ret_w = A_ret * ret_w[:, :, None]
                M_ret   = np.einsum('nkj,nkl->njl', A_ret_w, A_ret)
                rhs_sx  = np.einsum('nkj,nk->nj', A_ret_w, src[ret_nn, 0])
                rhs_sy  = np.einsum('nkj,nk->nj', A_ret_w, src[ret_nn, 1])
                sx_retrace = np.full(len(valid_j), np.nan)
                sy_retrace = np.full(len(valid_j), np.nan)
                if ret_valid.any():
                    M_pinv  = np.linalg.pinv(M_ret[ret_valid])
                    beta_sx = (M_pinv @ rhs_sx[ret_valid, :, None])[:, :, 0]
                    beta_sy = (M_pinv @ rhs_sy[ret_valid, :, None])[:, :, 0]
                    sx_retrace[ret_valid] = beta_sx[:, 2]
                    sy_retrace[ret_valid] = beta_sy[:, 2]
                retrace_err = np.hypot(sx_retrace - xOptic[valid_j],
                                       sy_retrace - yOptic[valid_j])
                bad = ~np.isfinite(retrace_err) | (retrace_err > retrace_tol * r)
                xPupil[valid_j[bad]] = np.nan
                yPupil[valid_j[bad]] = np.nan
                passing_err = retrace_err[~bad]
                if passing_err.size > 0:
                    retrace_data[(surf, irad)] = passing_err

            edge_data[(surf, irad)] = {
                "r": r, "clear": clear,
                "xPupil": xPupil, "yPupil": yPupil,
            }
            surf_edges.append((xPupil, yPupil))

        if write_movie:
            frame_surfs.append({
                "name": surf,
                "was_vig": (epRays.x[was_vignetted].copy(), epRays.y[was_vignetted].copy()),
                "new_vig": (epRays.x[newly_vignetted].copy(), epRays.y[newly_vignetted].copy()),
                "not_vig": (epRays.x[not_vignetted].copy(),  epRays.y[not_vignetted].copy()),
                "edges": surf_edges,
            })

        was_vignetted |= now_vignetted

    return {
        "theta":        theta,
        "edge_data":    edge_data,
        "retrace_data": retrace_data,
        "frame_surfs":  frame_surfs,
    }


def main(args):
    thetas = np.arange(
        args.theta_min_deg, args.theta_max_deg + args.dtheta_deg, args.dtheta_deg
    )
    write_movie = args.output != "/dev/null"

    worker_fn = functools.partial(
        _process_theta,
        azimuth_deg=args.azimuth_deg,
        nrad=args.nrad,
        retrace_tol=args.retrace_tol,
        write_movie=write_movie,
    )
    init_args = (args.version, args.band, args.rtp_deg, args.camera_piston_mm)
    n_jobs = min(args.jobs, len(thetas))

    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_init_worker,
        initargs=init_args,
    ) as pool:
        results = list(tqdm(
            pool.map(worker_fn, thetas),
            total=len(thetas),
            desc="trace",
            position=args.tqdm_position,
            leave=args.tqdm_position == 0,
        ))

    # Merge per-theta results into edge_accum and retrace_stats.
    edge_accum = {}   # (surf, irad) -> {"r", "clear", "xPupil": [...], "yPupil": [...]}
    retrace_stats = {}  # (surf, irad) -> list of per-point error arrays
    for result in results:
        for key, edata in result["edge_data"].items():
            if key not in edge_accum:
                edge_accum[key] = {"r": edata["r"], "clear": edata["clear"],
                                   "xPupil": [], "yPupil": []}
            edge_accum[key]["xPupil"].append(edata["xPupil"])
            edge_accum[key]["yPupil"].append(edata["yPupil"])
        for key, errs in result["retrace_data"].items():
            retrace_stats.setdefault(key, []).append(errs)

    # Render movie sequentially from stored frame data.
    if write_movie:
        fig = Figure(figsize=(16, 12), constrained_layout=True)
        FigureCanvasAgg(fig)
        axs = fig.subplots(3, 4)
        writer = FFMpegWriter(fps=args.fps)
        with writer.saving(fig, args.output, dpi=100):
            for result in tqdm(results, desc="movie",
                           position=args.tqdm_position,
                           leave=args.tqdm_position == 0):
                theta = result["theta"]
                for ax in axs.ravel():
                    ax.cla()
                    ax.set_facecolor("#333333")
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(-4.5, 4.5)
                    ax.set_ylim(-4.5, 4.5)
                for iax, sd in enumerate(result["frame_surfs"]):
                    ax = axs.ravel()[iax]
                    ax.set_title(sd["name"], fontsize=8)
                    for xp, yp in sd["edges"]:
                        ax.plot(xp, yp, "y-", lw=1)
                    ax.scatter(*sd["was_vig"], color="k", s=1)
                    ax.scatter(*sd["new_vig"], color="r", s=1)
                    ax.scatter(*sd["not_vig"], color="b", s=1)
                fig.suptitle(
                    f"v{args.version}  band={args.band}  rtp={args.rtp_deg:.1f}°"
                    f"  piston={args.camera_piston_mm:.2f} mm"
                    f"  az={args.azimuth_deg:.1f}°  θ={theta:.3f}°",
                    fontsize=10,
                )
                writer.grab_frame()

    # Report per-edge retrace RMS for passing points.
    if retrace_stats:
        _px = 4.18 / 65  # metres per pixel at donut scale
        lines = ["Retrace RMS (passing points):"]
        for (surf, irad), errs_list in sorted(retrace_stats.items()):
            all_errs = np.concatenate(errs_list)
            rms_m = float(np.sqrt(np.mean(all_errs**2)))
            n_edges = sum(1 for (s, _) in retrace_stats if s == surf)
            ename = ("inner" if irad == 0 else "outer") if n_edges == 2 else "outer"
            lines.append(f"  {surf}/{ename:<22}  rms={rms_m*1e3:6.2f} mm  "
                         f"({rms_m/_px:.3f} px)")
        text = "\n".join(lines)
        if args.retrace_log:
            with open(args.retrace_log, "w") as fh:
                fh.write(text + "\n")
        else:
            print(text, file=sys.stderr)

    # Build and write the intermediate edge-points ASDF file.
    # Name edges "inner"/"outer" for two-edge surfaces (mirrors), "outer" for one-edge.
    n_edges_per_surf = {}
    for (surf, irad) in edge_accum:
        n_edges_per_surf[surf] = max(n_edges_per_surf.get(surf, 0), irad + 1)

    edges_tree = {}
    for (surf, irad), data in edge_accum.items():
        if surf not in edges_tree:
            edges_tree[surf] = {}
        edge_name = ("inner" if irad == 0 else "outer") if n_edges_per_surf[surf] == 2 else "outer"
        edges_tree[surf][edge_name] = {
            "r": float(data["r"]),
            "clear": bool(data["clear"]),
            "xPupil": np.array(data["xPupil"]),  # (n_thetas, n_circle_pts)
            "yPupil": np.array(data["yPupil"]),
        }
    af = asdf.AsdfFile({
        "meta": {
            "version": args.version,
            "band": args.band,
            "rtp_deg": args.rtp_deg,
            "camera_piston_mm": args.camera_piston_mm,
            "azimuth_deg": args.azimuth_deg,
        },
        "thetas": thetas,
        "edges": edges_tree,
    })
    af.write_to(args.edges_output)
    print(f"Wrote edge points to {args.edges_output}", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate mask model for a v3.14 optic."
    )
    parser.add_argument(
        "--version", type=str, default="3.14", help="Optic version (default: 3.14)"
    )
    parser.add_argument(
        "--band", type=str, default="r", help="Band to use for the model (default: r)"
    )
    parser.add_argument(
        "--rtp-deg", type=float, default=0.0, help="RTP angle in degrees (default: 0.0)"
    )
    parser.add_argument(
        "--camera-piston-mm",
        type=float,
        default=0.0,
        help="Camera piston in mm (default: 0.0)",
    )
    parser.add_argument(
        "--azimuth-deg",
        type=float,
        default=0.0,
        help="Azimuth angle in degrees (default: 0.0)",
    )
    parser.add_argument(
        "--theta-min-deg",
        type=float,
        default=0.0,
        help="Theta min angle in degrees (default: 0.0)",
    )
    parser.add_argument(
        "--theta-max-deg",
        type=float,
        default=2.0,
        help="Theta max angle in degrees (default: 2.0)",
    )
    parser.add_argument(
        "--dtheta-deg",
        type=float,
        default=0.01,
        help="Delta theta angle in degrees (default: 0.01)",
    )
    parser.add_argument(
        "--nrad", type=int, default=200, help="Number of rays per theta (default: 200)"
    )
    parser.add_argument(
        "--output", type=str, default="output.mp4", help="Output movie file (default: output.mp4)"
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second (default: 10)"
    )
    parser.add_argument(
        "--edges-output", type=str, default="edges.asdf",
        help="Output ASDF file for interpolated edge points (default: edges.asdf)",
    )
    parser.add_argument(
        "--retrace-tol", type=float, default=0.01, dest="retrace_tol",
        help="Retrace tolerance as fraction of edge radius; points with "
             "round-trip error > tol*r are discarded (default: 0.01)",
    )
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 4,
        help="Number of parallel worker processes (default: cpu count)",
    )
    parser.add_argument(
        "--tqdm-position", type=int, default=0, dest="tqdm_position",
        help="tqdm bar position (0=standalone, 1=nested under outer bar)",
    )
    parser.add_argument(
        "--retrace-log", type=str, default=None, dest="retrace_log",
        help="Write retrace RMS summary to this file instead of stderr",
    )
    args = parser.parse_args()
    main(args)
