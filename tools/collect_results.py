"""Collect test metrics of all runs + simple per-scene statistics into results/results.csv and a markdown table.

Statistics come from the processed scene folders:
  ego speed          from lidar_pose (10 Hz)
  moving objects     tracks whose median speed > 5 km/h (from the interpolated boxes)
  vehicle pixels     mean share of image covered by fine vehicle masks (front cameras, every 4th frame)

usage (from the drivestudio root, any python with numpy + pillow):
  python tools/collect_results.py --runs ~/gs/out/drivestudio/truckscenes --data data/truckscenes/processed_10Hz/mini \
      --scene_json <TRUCKSCENES_ROOT>/v1.2-mini/scene.json --out results
"""
import argparse, csv, glob, json, os
import numpy as np
from PIL import Image


def scene_stats(d):
    P = [np.loadtxt(f) for f in sorted(glob.glob(f"{d}/lidar_pose/*.txt"))]
    xyz = np.array([p[:3, 3] for p in P])
    dist = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()); dur = (len(P) - 1) / 10
    info = json.load(open(f"{d}/instances/instances_info.json"))
    sp = []
    for o in info.values():
        fa = o["frame_annotations"]; f = np.array(fa["frame_idx"])
        if len(f) < 3: continue
        T = np.array(fa["obj_to_world"])[:, :3, 3]
        sp.append(np.median(np.linalg.norm(np.diff(T, axis=0), axis=1) / (np.diff(f) / 10)) * 3.6)
    sp = np.array(sp); mov = sp[sp > 5]
    fr = [np.mean(np.array(Image.open(f)) > 0)
          for c in "01" for f in sorted(glob.glob(f"{d}/fine_dynamic_masks/vehicle/*_{c}.png"))[::4]]
    return dict(ego_kmh=dist / dur * 3.6, dist_m=dist, objects=len(info), moving=len(mov),
                moving_kmh=float(np.median(mov)) if len(mov) else 0.0, vehicle_px_pct=float(np.mean(fr)) * 100 if fr else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--scene_json", required=True); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    scenes = json.load(open(os.path.expanduser(a.scene_json)))
    rows = []
    for run in sorted(glob.glob(os.path.expanduser(f"{a.runs}/streetgs_2cams_s*"))):
        m = sorted(glob.glob(f"{run}/metrics/images_test_*.json"))
        if not m: continue
        s = int(run.rsplit("_s", 1)[1]); mt = json.load(open(m[-1]))
        g = lambda k: mt.get(f"image_metrics/test/{k}", -1)
        tags = dict(t.split(".", 1) for t in scenes[s].get("description", "").split(";") if "." in t)
        r = dict(scene=s, area=tags.get("area", ""), weather=tags.get("weather", ""), lighting=tags.get("lighting", ""),
                 psnr=g("psnr"), ssim=g("ssim"), lpips=g("lpips"), vehicle_psnr=g("vehicle_psnr"), human_psnr=g("human_psnr"))
        r.update(scene_stats(os.path.expanduser(f"{a.data}/{s:03d}")))
        rows.append(r)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    hdr = "| scene | area | weather | ego km/h | moving objects (median km/h) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | vehicle PSNR | human PSNR |"
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    fmt = lambda v: "–" if v is None or v < 0 else f"{v:.2f}"
    for r in rows:
        lines.append(f"| {r['scene']} | {r['area']} | {r['weather']} | {r['ego_kmh']:.0f} | {r['moving']}/{r['objects']} "
                     f"({r['moving_kmh']:.0f}) | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['lpips']:.3f} | "
                     f"{fmt(r['vehicle_psnr'])} | {fmt(r['human_psnr'])} |")
    open(f"{a.out}/results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
