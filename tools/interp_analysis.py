"""Does 2 Hz box interpolation hurt the vehicles? Vehicle PSNR of test images grouped by the time to the nearest
annotated keyframe (0 ms = the box is a real annotation, 100/200 ms = linearly interpolated box).

inputs: per-image CSVs from tools/per_image_metrics.py and frame tables from tools/frame_times.py
  python tools/interp_analysis.py --dir <interp_check dir> --runs s4_camtime:004_camera s4:004_uniform s3:003_uniform ...
(run name streetgs_2cams_<x>.csv, frames table frames_<y>.csv)
"""
import argparse, csv, math, os
from collections import defaultdict


def load(p):
    return list(csv.DictReader(open(p)))


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--min_px", type=int, default=200, help="ignore images with fewer vehicle pixels")
    a = ap.parse_args()
    bins = lambda dt: "0 ms (keyframe)" if dt < 10 else ("~100 ms" if dt < 150 else "~200 ms")
    order = ["0 ms (keyframe)", "~100 ms", "~200 ms"]
    pooled_delta = defaultdict(list)
    print(f"vehicle PSNR by time to nearest keyframe (images with >= {a.min_px} vehicle px); n = images\n")
    print(f"{'run':24s}" + "".join(f"{b:>20s}" for b in order) + f"{'100ms-key':>11s}{'200ms-key':>11s}{'official':>10s}")
    for spec in a.runs:
        run, frames = spec.split(":")
        rows = load(os.path.join(a.dir, f"streetgs_2cams_{run}.csv"))
        dt = {int(r["frame_idx"]): float(r["dt_key_ms"]) for r in load(os.path.join(a.dir, f"frames_{frames}.csv"))}
        g = defaultdict(list)
        for r in rows:
            if int(r["vehicle_px"]) >= a.min_px and not math.isnan(float(r["vehicle_psnr"])):
                g[bins(dt[int(r["frame_idx"])])].append(float(r["vehicle_psnr"]))
        official = mean([float(r["vehicle_psnr"]) for r in rows if int(r["vehicle_px"]) > 0])
        m = {b: mean(g[b]) for b in order}
        d1, d2 = m["~100 ms"] - m[order[0]], m["~200 ms"] - m[order[0]]
        pooled_delta["~100 ms"].append(d1); pooled_delta["~200 ms"].append(d2)
        print(f"{run:24s}" + "".join(f"{m[b]:>13.2f} (n={len(g[b]):2d})" for b in order) + f"{d1:>+11.2f}{d2:>+11.2f}{official:>10.2f}")
    print(f"\nmean over runs of (bin - keyframe): ~100 ms {mean(pooled_delta['~100 ms']):+.2f} dB, "
          f"~200 ms {mean(pooled_delta['~200 ms']):+.2f} dB "
          f"(negative = interpolated boxes are worse)")


if __name__ == "__main__":
    main()
