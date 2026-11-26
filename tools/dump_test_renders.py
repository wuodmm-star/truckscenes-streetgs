"""Render drivestudio's test split from a finished run and save the renders losslessly (float32 .npy).

Same setup as tools/eval.py (dataset from the run's config.yaml, trainer.resume_from_checkpoint), then models.
video_utils.render_images(compute_metrics=True) -- so the printed metrics must reproduce the run's own
metrics/images_test_*.json, which checks that nothing differs from the official evaluation.

run from the drivestudio root (drivestudio env; on WSL also LD_LIBRARY_PATH=/usr/lib/wsl/lib):
  python <this> --ckpt <run>/checkpoint_final.pth --out <dir>
"""
import argparse, json, os, sys

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_images
from utils.misc import import_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = OmegaConf.load(os.path.join(os.path.dirname(a.ckpt), "config.yaml"))
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3), device=torch.device("cuda"))
    trainer.resume_from_checkpoint(ckpt_path=a.ckpt, load_only_model=True)
    trainer.set_eval()
    r = render_images(trainer=trainer, dataset=dataset.test_image_set, compute_metrics=True)
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/rgbs.npy", np.stack(r["rgbs"]).astype(np.float32))
    np.save(f"{a.out}/gt_rgbs.npy", np.stack(r["gt_rgbs"]).astype(np.float32))
    np.save(f"{a.out}/cam_ids.npy", np.array(r["cam_ids"]).reshape(-1))
    m = {k: float(v) for k, v in r.items() if isinstance(v, (float, int)) and not isinstance(v, bool)}
    json.dump(m, open(f"{a.out}/official_metrics.json", "w"), indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
