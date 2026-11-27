"""Timestamp of every processed frame and its distance to the nearest annotated (2 Hz) keyframe.

Rebuilds the 10 Hz timeline exactly as addon/datasets/truckscenes/truckscenes_preprocess.py does
(--timeline camera | uniform), so frame indices match the processed scene. Only needs the json tables.

  python tools/frame_times.py --root <TRUCKSCENES_ROOT> --scene 4 --timeline camera --out frames_004.csv
columns: frame_idx, t_us, dt_key_ms (time to the nearest keyframe), is_key (dt_key_ms < 10)
"""
import argparse, csv, json

CAM0 = "CAMERA_LEFT_FRONT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--version", default="v1.2-mini")
    ap.add_argument("--scene", type=int, required=True); ap.add_argument("--timeline", default="camera", choices=["camera", "uniform"])
    ap.add_argument("--interpolate_N", type=int, default=4); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = lambda n: json.load(open(f"{a.root}/{a.version}/{n}.json", encoding="utf-8"))
    sensor = {s["token"]: s["channel"] for s in L("sensor")}
    cal = {c["token"]: sensor[c["sensor_token"]] for c in L("calibrated_sensor")}
    samples = {s["token"]: s for s in L("sample")}
    scene = L("scene")[a.scene]
    keys, t = [], scene["first_sample_token"]
    while t: keys.append(samples[t]); t = samples[t]["next"]
    toks = {s["token"] for s in keys}
    kts = [s["timestamp"] for s in keys]
    if a.timeline == "camera":
        cam = sorted(x["timestamp"] for x in L("sample_data") if x["sample_token"] in toks and cal[x["calibrated_sensor_token"]] == CAM0)
        ts = [x for x in cam if kts[0] - 50_000 <= x <= kts[-1] + 50_000]
    else:
        N = a.interpolate_N
        ts = [int(kts[i] + j * (kts[i + 1] - kts[i]) / (N + 1)) for i in range(len(kts) - 1) for j in range(N + 1)] + [kts[-1]]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_idx", "t_us", "dt_key_ms", "is_key"])
        for fi, x in enumerate(ts):
            d = min(abs(x - k) for k in kts) / 1e3
            w.writerow([fi, x, f"{d:.1f}", int(d < 10)])
    print(f"scene {a.scene} ({a.timeline}): {len(ts)} frames, {len(kts)} keyframes -> {a.out}")


if __name__ == "__main__":
    main()
