"""Overlay sky (blue) and fine dynamic (red) masks on a few frames -> one preview jpg."""
import sys, cv2, numpy as np
d, out = sys.argv[1], sys.argv[2]
tiles = []
for f in ["000", "098", "195"]:
    row = []
    for c in [0, 1]:
        img = cv2.imread(f"{d}/images/{f}_{c}.jpg").astype(np.float32)
        sky = cv2.imread(f"{d}/sky_masks/{f}_{c}.png", 0) > 0
        dyn = cv2.imread(f"{d}/fine_dynamic_masks/all/{f}_{c}.png", 0) > 0
        img[sky] = 0.5 * img[sky] + 0.5 * np.array([255, 80, 0])
        img[dyn] = 0.4 * img[dyn] + 0.6 * np.array([0, 0, 255])
        img = img.astype(np.uint8); cv2.putText(img, f"frame {f} cam {c}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        row.append(cv2.resize(img, (990, 471)))
    tiles.append(np.hstack(row))
cv2.imwrite(out, np.vstack(tiles).astype(np.uint8))
