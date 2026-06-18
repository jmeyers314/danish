"""mask_sweep.py

Run generate_mask_model.py + fit_mask_edges.py across the standard parameter
space, producing one edges.asdf, fitted YAML, and optional movie per config.

Parameter grid (mirrors run_v3_14_sweep.sh):
  - Azimuth study : band=r, rtp=0, piston=0; az = 45, 135, 225, 315
  - RTP study     : band=r, piston=0; rtp in {-80,-60,-40,-20,40,60,80} (0 covered by azimuth study);
                    ocs_az = ccs_az(45) + rtp
  - Piston study  : band=r, rtp=0, az=45; piston in {-1.5, +1.5} mm
  - Band study    : rtp=0, az=45, piston=0; all bands

Usage (from repo root):
    python devel/mask_model/mask_sweep.py [--outdir PATH] [--dry-run]
                                          [--theta-max-deg F] [--dtheta-deg F]
                                          [--nrad N] [--no-movie]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def stem(version, band, rtp, az, piston):
    rtp_str    = f"{rtp:+d}".replace("+", "p").replace("-", "m")
    az_str     = f"{int(az):+d}".replace("+", "p").replace("-", "m")
    piston_str = f"{piston:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
    return f"v{version}_{band}_rtp{rtp_str}_az{az_str}_p{piston_str}"


def build_runs():
    runs = []

    # Azimuth study
    for az in [45, 135, 225, 315]:
        runs.append(dict(band="r", rtp=0, az=az, piston=0.0))

    # RTP study (az tracks R44 WFS, ccs_az=45)
    for rtp in [-80, -60, -40, -20, 40, 60, 80]:   # 0 already in azimuth study
        runs.append(dict(band="r", rtp=rtp, az=45 + rtp, piston=0.0))

    # Piston study
    for piston in [-1.5, 1.5]:       # 0 already in azimuth study
        runs.append(dict(band="r", rtp=0, az=45, piston=piston))

    # Band study
    for band in ["u", "g", "i", "z", "y"]:   # r already in azimuth study
        runs.append(dict(band=band, rtp=0, az=45, piston=0.0))

    return runs


def _run_cmd(cmd, logfile=None):
    """Run a subprocess.

    stderr passes through to the terminal (so tqdm renders inline) unless
    *logfile* is given, in which case it is redirected to that file.
    """
    if logfile is None:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    else:
        with open(logfile, "w") as fh:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=fh)
    return proc.returncode


def run_one(run, outdir, yamldir, args):
    band, rtp, az, piston = run["band"], run["rtp"], run["az"], run["piston"]
    s = stem(args.version, band, rtp, az, piston)
    edges_out = outdir / f"{s}.asdf"
    movie_out = outdir / f"{s}.mp4"
    yaml_out  = yamldir / f"RubinObsc_{s}.yaml"

    if args.refit:
        # Skip generation; require the .asdf to already exist.
        if not edges_out.exists():
            return s, "SKIP (no asdf)"
        if yaml_out.exists() and not args.overwrite:
            return s, "SKIP"
    else:
        outputs_exist = (
            edges_out.exists()
            and yaml_out.exists()
            and (args.no_movie or movie_out.exists())
        )
        if outputs_exist and not args.overwrite:
            return s, "SKIP"

        # Step 1: generate edge points
        cmd = [
            sys.executable,
            str(REPO_ROOT / "devel" / "mask_model" / "generate_mask_model.py"),
            "--version", args.version,
            "--band", band,
            "--rtp-deg", str(rtp),
            "--camera-piston-mm", str(piston),
            "--azimuth-deg", str(az),
            "--theta-max-deg", str(args.theta_max_deg),
            "--dtheta-deg", str(args.dtheta_deg),
            "--nrad", str(args.nrad),
            "--edges-output", str(edges_out),
            "--output", str(movie_out) if not args.no_movie else "/dev/null",
            "--retrace-tol", str(args.retrace_tol),
            "--tqdm-position", "1",
            "--retrace-log", str(edges_out.with_suffix(".retrace.log")),
        ]
        if _run_cmd(cmd) != 0:
            return s, "FAIL (generate)"

    # Step 2: fit polynomial model → YAML
    log_out = yamldir / f"RubinObsc_{s}.log"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "devel" / "mask_model" / "fit_mask_edges.py"),
        str(edges_out),
        "--output", str(yaml_out),
        "--deg", str(args.deg),
        "--diagnostics", str(yamldir / f"RubinObsc_{s}.pdf"),
    ]
    if _run_cmd(cmd, logfile=log_out) != 0:
        return s, "FAIL (fit)"

    return s, "DONE"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=str, default="3.14",
                        help="StarSharp optic version to trace (default: 3.14)")
    parser.add_argument("--outdir", type=Path,
                        default=REPO_ROOT / "devel" / "mask_model" / "edges",
                        help="Directory for .asdf and .mp4 files (default: devel/mask_model/edges/)")
    parser.add_argument("--yamldir", type=Path,
                        default=REPO_ROOT / "devel" / "mask_model" / "yaml",
                        help="Directory for fitted YAML files (default: devel/mask_model/yaml/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print runs without executing")
    parser.add_argument("--theta-max-deg", type=float, default=2.0)
    parser.add_argument("--dtheta-deg", type=float, default=0.01)
    parser.add_argument("--nrad", type=int, default=100)
    parser.add_argument("--deg", type=int, default=3,
                        help="Polynomial degree for fit_mask_edges (default: 3)")
    parser.add_argument("--no-movie", action="store_true",
                        help="Skip writing the mp4 (faster)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if output ASDF already exists")
    parser.add_argument("--refit", action="store_true",
                        help="Re-run only fit_mask_edges using existing .asdf files")
    parser.add_argument("--retrace-tol", type=float, default=0.01,
                        help="Retrace tolerance as fraction of edge radius (default: 0.01)")
    args = parser.parse_args()

    runs = build_runs()
    print(f"Total runs: {len(runs)}")

    if args.dry_run:
        for run in runs:
            s = stem(args.version, run["band"], run["rtp"], run["az"], run["piston"])
            print(f"  {args.outdir}/{s}.asdf")
            print(f"  {args.yamldir}/RubinObsc_{s}.yaml")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.yamldir.mkdir(parents=True, exist_ok=True)

    with tqdm(runs, position=0, leave=True, unit="run") as pbar:
        for run in pbar:
            s = stem(args.version, run["band"], run["rtp"], run["az"], run["piston"])
            pbar.set_description(s)
            try:
                _, status = run_one(run, args.outdir, args.yamldir, args)
            except Exception as exc:
                status = f"ERROR: {exc}"
            tqdm.write(f"  {s:<45}  {status}")


if __name__ == "__main__":
    main()
