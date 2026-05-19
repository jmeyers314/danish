"""study.py

Run animate_pupil.py for four curated study groups, each comparing a variant
YAML against the relevant reference (band=r, rtp=0, az=45, piston=0).

Studies:
  - Azimuth : az=135, 225, 315 vs reference az=45
  - RTP     : rtp=-80,-60,-40,-20,+40,+60,+80 vs reference rtp=0
  - Piston  : piston=-1.5,+1.5 mm vs reference piston=0
  - Band    : u, g, i, z, y vs reference r

Usage (from repo root):
    python devel/mask_model/study.py [--yamldir PATH] [--outdir PATH]
                                     [--jobs N] [--dry-run]
                                     [--spider-angle DEG]
                                     [--nframes N] [--npix N]
                                     [--fps N] [--dpi N]
"""

import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANIMATE   = REPO_ROOT / "devel" / "mask_model" / "animate_pupil.py"


def stem(band, rtp, az, piston):
    rtp_str    = f"{rtp:+d}".replace("+", "p").replace("-", "m")
    az_str     = f"{int(az):+d}".replace("+", "p").replace("-", "m")
    piston_str = (f"{piston:+.1f}".replace("+", "p")
                                  .replace("-", "m")
                                  .replace(".", "d"))
    return f"v3.14_{band}_rtp{rtp_str}_az{az_str}_p{piston_str}"


def yaml_path(yamldir, band, rtp, az, piston):
    return yamldir / f"RubinObsc_{stem(band, rtp, az, piston)}.yaml"


def build_comparisons(yamldir):
    ref = yaml_path(yamldir, "r", 0, 45, 0.0)
    comps = []

    # Azimuth study: sweep direction matches each variant's azimuth
    for az in [135, 225, 315]:
        az_str = f"{int(az):+d}".replace("+", "p").replace("-", "m")
        comps.append(dict(
            group="azimuth",
            label=f"azimuth_az{az_str}_vs_p45",
            file1=ref,
            file2=yaml_path(yamldir, "r", 0, az, 0.0),
            azimuth_deg=float(az),
        ))

    # RTP study: sweep direction matches variant (ocs_az = 45 + rtp)
    for rtp in [-80, -60, -40, -20, 40, 60, 80]:
        rtp_str = f"{rtp:+d}".replace("+", "p").replace("-", "m")
        comps.append(dict(
            group="rtp",
            label=f"rtp_{rtp_str}_vs_0",
            file1=ref,
            file2=yaml_path(yamldir, "r", rtp, 45 + rtp, 0.0),
            azimuth_deg=float(45 + rtp),
        ))

    # Piston study
    for piston in [-1.5, 1.5]:
        piston_str = f"{piston:+.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
        comps.append(dict(
            group="piston",
            label=f"piston_{piston_str}_vs_0",
            file1=ref,
            file2=yaml_path(yamldir, "r", 0, 45, piston),
            azimuth_deg=45.0,
        ))

    # Band study
    for band in ["u", "g", "i", "z", "y"]:
        comps.append(dict(
            group="band",
            label=f"band_{band}_vs_r",
            file1=ref,
            file2=yaml_path(yamldir, band, 0, 45, 0.0),
            azimuth_deg=45.0,
        ))

    return comps


def run_one(comp, outdir, args):
    label = comp["label"]
    out   = outdir / f"{label}.mp4"

    if out.exists() and not args.overwrite:
        return label, "SKIP"

    for f in (comp["file1"], comp["file2"]):
        if not Path(f).exists():
            return label, f"SKIP (missing {Path(f).name})"

    tqdm.write(f"  start  {label}")

    cmd = [
        sys.executable, str(ANIMATE),
        str(comp["file1"]), str(comp["file2"]),
        "--output", str(out),
        "--azimuth-deg", str(comp["azimuth_deg"]),
        "--nframes", str(args.nframes),
        "--npix",    str(args.npix),
        "--fps",     str(args.fps),
        "--dpi",     str(args.dpi),
    ]
    if args.spider_angle is not None:
        cmd += ["--spider-angle", str(args.spider_angle)]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)

    def _relay():
        for line in proc.stderr:
            tqdm.write(f"  [{label}] {line.rstrip()}")

    t = threading.Thread(target=_relay, daemon=True)
    t.start()
    proc.wait()
    t.join()

    if proc.returncode != 0:
        return label, "FAIL"
    return label, "DONE"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yamldir", type=Path,
                        default=REPO_ROOT / "devel" / "mask_model" / "yaml",
                        help="Directory containing fitted YAML files")
    parser.add_argument("--outdir", type=Path,
                        default=REPO_ROOT / "devel" / "mask_model" / "study",
                        help="Output directory for movies (default: devel/mask_model/study/)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                        help="Parallel workers (default: cpu count)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print comparisons without running")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if output mp4 already exists")
    parser.add_argument("--spider-angle", type=float, default=None,
                        help="Spider rotator angle in degrees (omit to skip spiders)")
    parser.add_argument("--nframes", type=int, default=200,
                        help="Number of theta frames per movie (default: 200)")
    parser.add_argument("--npix", type=int, default=800,
                        help="Pupil grid size in pixels (default: 800)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second (default: 15)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Output DPI (default: 200)")
    args = parser.parse_args()

    comps = build_comparisons(args.yamldir)

    if args.dry_run:
        prev_group = None
        for c in comps:
            if c["group"] != prev_group:
                print(f"\n[{c['group']}]")
                prev_group = c["group"]
            print(f"  {c['label']}")
            print(f"    ref : {c['file1']}")
            print(f"    cmp : {c['file2']}")
            print(f"    az  : {c['azimuth_deg']}°")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Total comparisons: {len(comps)}")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, c, args.outdir, args): c for c in comps}
        with tqdm(total=len(comps), unit="comp", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    _, status = fut.result()
                except Exception as exc:
                    status = f"ERROR: {exc}"
                tqdm.write(f"  done   {c['label']:<45}  {status}")
                pbar.update(1)


if __name__ == "__main__":
    main()
