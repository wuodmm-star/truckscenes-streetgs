"""MAN TruckScenes -> drivestudio "processed" scene folder (same layout as datasets/nuscenes/nuscenes_preprocess.py
in interpolated mode), so Street Gaussians / OmniRe can train on it.

  <out>/<idx>/images/{frame:03d}_{cam}.jpg       copied 1980x943 originals (TruckScenes images are undistorted)
  <out>/<idx>/extrinsics/{frame:03d}_{cam}.txt   cam-to-world 4x4 (OpenCV camera axes)
  <out>/<idx>/intrinsics/{cam}.txt               fx fy cx cy k1 k2 p1 p2 k3
  <out>/<idx>/lidar/{frame:03d}.bin              float32 (N,4) x y z intensity, in the EGO frame of that frame
  <out>/<idx>/lidar_pose/{frame:03d}.txt         ego-to-world 4x4 for that frame
  <out>/<idx>/instances/{instances_info,frame_instances}.json
  <out>/<idx>/dynamic_masks/{all,human,vehicle}/{frame:03d}_{cam}.png
  <out>/<idx>/sky_masks/                         (filled later by datasets/tools/extract_masks.py)
  <out>/<idx>/check/                             sanity-check images (LiDAR + boxes projected into the cameras)

Differences from the nuScenes processor, and why:
  * 6 LiDARs instead of 1 (LIDAR_TOP_FRONT/LEFT/RIGHT, LEFT, RIGHT, REAR). For each frame, the sweep of every LiDAR
    closest in time is taken to the world with ITS OWN ego pose, then into the ego frame of the frame's reference
    sweep (LIDAR_TOP_FRONT). The loader treats each file as rays from one origin; with several LiDARs the origin is
    the ego origin instead of the true sensor -- point positions (what Gaussians are initialised from and what the
    depth loss uses) are exact, only the ray origin is approximate.
  * Points on our own tractor and trailer are removed (box in the ego frame; the trailer swings when turning).
  * Extra TruckScenes classes: vehicle.other / vehicle.train are rigid vehicles; vehicle.ego_trailer (our own
    trailer) is NOT an object to model and is skipped.
  * Camera ids: 0 LEFT_FRONT, 1 RIGHT_FRONT, 2 LEFT_BACK, 3 RIGHT_BACK. drivestudio anchors the world at camera 0
    of the first frame, so camera 0 must exist in every frame.

Runs in any Python env that has truckscenes-devkit (pip install truckscenes-devkit; used only for its
compressed-PCD reader); output is plain files.
usage (from the drivestudio root):
  python datasets/truckscenes/truckscenes_preprocess.py --root <TRUCKSCENES_ROOT> \
      --out data/truckscenes/processed_10Hz/mini --scenes 2,3
"""
import argparse, json, os, shutil
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot, Slerp
from truckscenes.utils.data_classes import LidarPointCloud

CAMS = ["CAMERA_LEFT_FRONT", "CAMERA_RIGHT_FRONT", "CAMERA_LEFT_BACK", "CAMERA_RIGHT_BACK"]
LIDARS = ["LIDAR_TOP_FRONT", "LIDAR_TOP_LEFT", "LIDAR_TOP_RIGHT", "LIDAR_LEFT", "LIDAR_RIGHT", "LIDAR_REAR"]
HUMAN = ["animal", "human.pedestrian.adult", "human.pedestrian.child", "human.pedestrian.construction_worker",
         "human.pedestrian.personal_mobility", "human.pedestrian.police_officer", "human.pedestrian.stroller",
         "human.pedestrian.wheelchair", "vehicle.bicycle", "vehicle.motorcycle"]
VEHICLE = ["vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.car", "vehicle.construction",
           "vehicle.emergency.ambulance", "vehicle.emergency.police", "vehicle.trailer", "vehicle.truck",
           "vehicle.other", "vehicle.train"]
DYNAMIC = HUMAN + VEHICLE


def se3(q_wxyz, t):
    T = np.eye(4); T[:3, :3] = Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix(); T[:3, 3] = t
    return T


def box_corners(o2w, lwh):  # 8 world corners; object frame x = length (forward), y = width, z = height
    l, w, h = lwh
    c = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
    return c @ o2w[:3, :3].T + o2w[:3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="v1.2-mini")
    ap.add_argument("--scenes", default="2", help="comma-separated indices into scene.json (folder name = index)")
    ap.add_argument("--timeline", default="camera", choices=["camera", "uniform"],
                    help="camera: frame times = real capture times of camera 0; uniform: evenly spaced (v1 behaviour)")
    ap.add_argument("--interpolate_N", type=int, default=4, help="uniform timeline: frames inserted between 2 Hz keyframes")
    ap.add_argument("--max_range", type=float, default=120.0)
    a = ap.parse_args()
    root, outroot = os.path.expanduser(a.root), os.path.expanduser(a.out)
    L = lambda n: json.load(open(f"{root}/{a.version}/{n}.json", encoding="utf-8"))
    sensor = {s["token"]: s for s in L("sensor")}
    cal = {c["token"]: dict(c, channel=sensor[c["sensor_token"]]["channel"]) for c in L("calibrated_sensor")}
    pose = {p["token"]: p for p in L("ego_pose")}
    samples = {s["token"]: s for s in L("sample")}
    cat = {c["token"]: c["name"] for c in L("category")}
    inst_cat = {i["token"]: cat[i["category_token"]] for i in L("instance")}
    all_sd, all_ann, scenes = L("sample_data"), L("sample_annotation"), L("scene")

    for sidx in [int(s) for s in a.scenes.split(",")]:
        scene = scenes[sidx]; out = f"{outroot}/{sidx:03d}"
        print(f"== scene {sidx}: {scene['name']}  ({scene.get('description', '')})", flush=True)
        for d in ["images", "sky_masks", "extrinsics", "intrinsics", "lidar", "lidar_pose", "instances", "check",
                  "dynamic_masks/all", "dynamic_masks/human", "dynamic_masks/vehicle"]:
            os.makedirs(f"{out}/{d}", exist_ok=True)
        keys, t = [], scene["first_sample_token"]
        while t: keys.append(samples[t]); t = samples[t]["next"]
        toks = {s["token"] for s in keys}
        sds = {}
        for x in all_sd:
            if x["sample_token"] in toks:
                sds.setdefault(cal[x["calibrated_sensor_token"]]["channel"], []).append(x)
        for v in sds.values(): v.sort(key=lambda x: x["timestamp"])

        # ---- 10 Hz timeline
        #  camera  (default): the real capture times of camera 0. The cameras are hardware-synchronised (< 3 ms
        #          apart) and run at 10 Hz, so images, camera poses AND interpolated boxes all refer to the same instant.
        #  uniform (v1): keyframe timestamps with N evenly spaced ones in between. Images are then picked as the
        #          closest capture, which can be tens of ms (up to one frame) away from the instant the boxes are
        #          interpolated at -> fast vehicles are rendered at a shifted position.
        kts = [s["timestamp"] for s in keys]; N = a.interpolate_N
        if a.timeline == "camera":
            ts = sorted(x["timestamp"] for x in sds[CAMS[0]] if kts[0] - 50_000 <= x["timestamp"] <= kts[-1] + 50_000)
        else:
            ts = [int(kts[i] + j * (kts[i + 1] - kts[i]) / (N + 1)) for i in range(len(kts) - 1) for j in range(N + 1)] + [kts[-1]]
        ts = np.array(ts, np.int64); F = len(ts)
        closest = lambda ch: [sds[ch][int(np.argmin(np.abs(np.array([x["timestamp"] for x in sds[ch]]) - t)))] for t in ts]
        world_T_ego = lambda x: se3(pose[x["ego_pose_token"]]["rotation"], pose[x["ego_pose_token"]]["translation"])
        ego_T_sensor = lambda x: se3(cal[x["calibrated_sensor_token"]]["rotation"], cal[x["calibrated_sensor_token"]]["translation"])

        # ---- cameras
        c2w, Ks, img_dt = {}, {}, []
        for ci, ch in enumerate(CAMS):
            sel = closest(ch); c2w[ci] = []
            for fi, x in enumerate(sel):
                shutil.copyfile(f"{root}/{x['filename']}", f"{out}/images/{fi:03d}_{ci}.jpg")
                T = world_T_ego(x) @ ego_T_sensor(x); c2w[ci].append(T)
                np.savetxt(f"{out}/extrinsics/{fi:03d}_{ci}.txt", T)
                img_dt.append(abs(x["timestamp"] - ts[fi]) / 1e3)
            K = np.array(cal[sel[0]["calibrated_sensor_token"]]["camera_intrinsic"]); Ks[ci] = K
            np.savetxt(f"{out}/intrinsics/{ci}.txt", [K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0, 0, 0, 0, 0])
        W, H = sds[CAMS[0]][0]["width"], sds[CAMS[0]][0]["height"]
        print(f"{F} frames x {len(CAMS)} cams, {W}x{H}; image-to-timeline offset mean {np.mean(img_dt):.1f} ms, max {np.max(img_dt):.1f} ms", flush=True)

        # ---- LiDAR: 6 sensors merged into the ego frame of the frame's LIDAR_TOP_FRONT sweep
        per_lidar = {ch: closest(ch) for ch in LIDARS}
        npts = []
        for fi in range(F):
            ref = per_lidar["LIDAR_TOP_FRONT"][fi]; ego_inv = np.linalg.inv(world_T_ego(ref)); chunks = []
            for ch in LIDARS:
                x = per_lidar[ch][fi]
                pc = LidarPointCloud.from_file(f"{root}/{x['filename']}").points  # (4+, n) sensor frame
                p, inten = pc[:3].T.astype(np.float64), pc[3]
                Te = ego_T_sensor(x); pe = p @ Te[:3, :3].T + Te[:3, 3]  # this sweep's ego frame
                own = (pe[:, 0] > -18) & (pe[:, 0] < 7) & (np.abs(pe[:, 1]) < 2.6) & (pe[:, 2] < 4.5)  # tractor + trailer
                keep = ~own & (np.linalg.norm(pe[:, :2], axis=1) < a.max_range)
                T = ego_inv @ world_T_ego(x)  # sweep ego -> reference ego (motion between the sensors' timestamps)
                pr = pe[keep] @ T[:3, :3].T + T[:3, 3]
                chunks.append(np.c_[pr, inten[keep]])
            pts = np.concatenate(chunks).astype(np.float32); npts.append(len(pts))
            pts.tofile(f"{out}/lidar/{fi:03d}.bin")
            np.savetxt(f"{out}/lidar_pose/{fi:03d}.txt", world_T_ego(ref))
        print(f"LiDAR: {np.mean(npts):,.0f} points per frame (6 sensors, own truck removed)", flush=True)

        # ---- objects: 2 Hz keyframe boxes -> tracks interpolated at every frame's timestamp
        #      (position: linear, rotation: Slerp, size: linear; a frame up to 50 ms outside the annotated span
        #       gets the nearest annotated box)
        by_inst = {}
        for b in all_ann:
            if b["sample_token"] in toks and inst_cat[b["instance_token"]] in DYNAMIC:
                by_inst.setdefault(b["instance_token"], []).append((samples[b["sample_token"]]["timestamp"], b))
        skipped = sorted({inst_cat[b["instance_token"]] for b in all_ann if b["sample_token"] in toks} - set(DYNAMIC))
        instances_info, frame_instances = {}, {fi: [] for fi in range(F)}
        for oid, (tok, v) in enumerate(by_inst.items()):
            v.sort(key=lambda z: z[0])
            kt = np.array([t for t, _ in v], np.float64)
            Rs = Rot.from_quat([[b["rotation"][1], b["rotation"][2], b["rotation"][3], b["rotation"][0]] for _, b in v])
            tr = np.array([b["translation"] for _, b in v]); lwh = np.array([[b["size"][1], b["size"][0], b["size"][2]] for _, b in v])
            frames = [f for f in range(F) if kt[0] - 50_000 <= ts[f] <= kt[-1] + 50_000]
            if len(kt) == 1: frames = [int(np.argmin(np.abs(ts - kt[0])))]  # single annotation: its nearest frame
            tq = np.clip(ts[frames].astype(np.float64), kt[0], kt[-1])
            rot = Slerp(kt, Rs)(tq) if len(kt) > 1 else Rot.from_quat(np.repeat(Rs.as_quat(), len(frames), 0))
            o2w = []
            for i, f in enumerate(frames):
                T = np.eye(4); T[:3, :3] = rot[i].as_matrix(); T[:3, 3] = [np.interp(tq[i], kt, tr[:, j]) for j in range(3)]
                o2w.append(T.tolist())
            sizes = [[float(np.interp(tq[i], kt, lwh[:, j])) for j in range(3)] for i in range(len(frames))]
            instances_info[oid] = dict(id=tok, class_name=inst_cat[tok],
                                       frame_annotations=dict(frame_idx=frames, obj_to_world=o2w, box_size=sizes))
            for f in frames: frame_instances[f].append(oid)
        json.dump(instances_info, open(f"{out}/instances/instances_info.json", "w"))
        json.dump(frame_instances, open(f"{out}/instances/frame_instances.json", "w"))
        ncls = {}
        for v in instances_info.values(): ncls[v["class_name"]] = ncls.get(v["class_name"], 0) + 1
        print(f"objects: {len(instances_info)} tracks {ncls}; skipped classes {skipped}", flush=True)

        # ---- coarse dynamic masks: 2D bounding rectangle of each fully visible projected box (as nuScenes)
        groups = dict(all=DYNAMIC, human=HUMAN, vehicle=VEHICLE)
        for ci in range(len(CAMS)):
            for fi in range(F):
                w2c = np.linalg.inv(c2w[ci][fi]); masks = {g: np.zeros((H, W), np.uint8) for g in groups}
                for oid in frame_instances[fi]:
                    info = instances_info[oid]; i = fi - info["frame_annotations"]["frame_idx"][0]
                    cw = box_corners(np.array(info["frame_annotations"]["obj_to_world"][i]), info["frame_annotations"]["box_size"][i])
                    cc = cw @ w2c[:3, :3].T + w2c[:3, 3]
                    if not np.all(cc[:, 2] > 0.1): continue
                    uv = cc @ Ks[ci].T; uv = uv[:, :2] / uv[:, 2:]
                    if not (np.all(uv[:, 0] >= 0) and np.all(uv[:, 0] < W) and np.all(uv[:, 1] >= 0) and np.all(uv[:, 1] < H)): continue
                    u0, v0 = uv.min(0).astype(int); u1, v1 = uv.max(0).astype(int)
                    for g, cls in groups.items():
                        if info["class_name"] in cls: masks[g][v0:v1, u0:u1] = 255
                for g in groups:
                    cv2.imwrite(f"{out}/dynamic_masks/{g}/{fi:03d}_{ci}.png", masks[g])

        # ---- sanity check: LiDAR (by depth) + object boxes projected into every camera at 3 moments
        for fi in [0, F // 2, F - 1]:
            pl = np.fromfile(f"{out}/lidar/{fi:03d}.bin", np.float32).reshape(-1, 4)[:, :3]
            Tl = np.loadtxt(f"{out}/lidar_pose/{fi:03d}.txt"); pw = pl @ Tl[:3, :3].T + Tl[:3, 3]
            tiles = []
            for ci in range(len(CAMS)):
                img = cv2.imread(f"{out}/images/{fi:03d}_{ci}.jpg"); w2c = np.linalg.inv(c2w[ci][fi])
                pc = pw @ w2c[:3, :3].T + w2c[:3, 3]; m = (pc[:, 2] > 0.5) & (pc[:, 2] < 80)
                uv = pc[m] @ Ks[ci].T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
                col = cv2.applyColorMap((np.clip(pc[m, 2] / 80, 0, 1) * 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0]
                ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                img[uv[ok, 1], uv[ok, 0]] = col[ok]
                for oid in frame_instances[fi]:
                    info = instances_info[oid]; i = fi - info["frame_annotations"]["frame_idx"][0]
                    cw = box_corners(np.array(info["frame_annotations"]["obj_to_world"][i]), info["frame_annotations"]["box_size"][i])
                    cc = cw @ w2c[:3, :3].T + w2c[:3, 3]
                    if np.any(cc[:, 2] < 0.5): continue
                    uv2 = cc @ Ks[ci].T; uv2 = (uv2[:, :2] / uv2[:, 2:]).astype(int)
                    for e0, e1 in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
                        cv2.line(img, tuple(map(int, uv2[e0])), tuple(map(int, uv2[e1])), (255, 255, 255), 2)
                cv2.putText(img, f"{fi} {CAMS[ci]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                tiles.append(cv2.resize(img, (W // 2, H // 2)))
            cv2.imwrite(f"{out}/check/frame{fi:03d}.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
        print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
