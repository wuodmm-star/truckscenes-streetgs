"""Per-image test metrics of a finished drivestudio run (the official eval only reports means).

Renders the test split exactly like tools/eval.py / models.video_utils.render_images and writes one CSV row per
test image: frame index, camera, full-image PSNR, vehicle-mask PSNR (same definition as drivestudio's
"Vehicle-Only PSNR": PSNR over the pixels of the fine vehicle mask) and the number of vehicle pixels.
The mean of the vehicle_psnr column over images with vehicle pixels reproduces the official number.

run from the drivestudio root (drivestudio env; on WSL also LD_LIBRARY_PATH=/usr/lib/wsl/lib):
  python <this> --ckpt <run>/checkpoint_final.pth --out <run>/per_image_test.csv [--split train]
Note: drivestudio renders a TEST frame with each vehicle pose = mean of the (optimised) poses of the previous and next
frame (models/nodes/rigid.py, in_test_set), TRAIN frames with their own optimised pose.
"""
import argparse, csv, os, sys

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets.driving_dataset import DrivingDataset
from models.video_utils import compute_psnr
from utils.misc import import_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    a = ap.parse_args()
    cfg = OmegaConf.load(os.path.join(os.path.dirname(a.ckpt), "config.yaml"))
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3), device=torch.device("cuda"))
    trainer.resume_from_checkpoint(ckpt_path=a.ckpt, load_only_model=True)
    trainer.set_eval()

    test = dataset.test_image_set if a.split == "test" else dataset.train_image_set; ds = trainer._get_downscale_factor(); rows = []
    with torch.no_grad():
        for i in tqdm(range(len(test)), desc=f"per-image {a.split} metrics"):
            img, cam = test.get_image(i, ds)
            img = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in img.items()}
            cam = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in cam.items()}
            rgb = trainer(img, cam)["rgb"].clamp(0., 1.)
            gt = img["pixels"]
            vm = img["vehicle_masks"].bool() if "vehicle_masks" in img else torch.zeros_like(gt[..., 0], dtype=torch.bool)
            n = int(vm.sum())
            rows.append(dict(i=i, frame_idx=int(img["frame_idx"].flatten()[0]), cam_id=int(cam["cam_id"].flatten()[0]),
                             psnr=compute_psnr(rgb, gt), vehicle_psnr=compute_psnr(rgb[vm], gt[vm]) if n else float("nan"),
                             vehicle_px=n))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    v = np.array([r["vehicle_psnr"] for r in rows if r["vehicle_px"] > 0])
    print(f"{len(rows)} {a.split} images -> {a.out}; mean PSNR {np.mean([r['psnr'] for r in rows]):.4f}, "
          f"mean vehicle PSNR {v.mean():.4f} over {len(v)} images with vehicles")


if __name__ == "__main__":
    main()
