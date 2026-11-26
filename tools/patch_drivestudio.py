"""Register the 'truckscenes' dataset in drivestudio (idempotent)."""
import re
meta = "datasets/dataset_meta.py"
s = open(meta).read()
if '"truckscenes"' not in s:
    cams = ["CAMERA_LEFT_FRONT", "CAMERA_RIGHT_FRONT", "CAMERA_LEFT_BACK", "CAMERA_RIGHT_BACK"]
    entry = '    "truckscenes": {\n' + "".join(
        f'        {i}: {{\n            "camera_name": "{c}",\n            "original_size": (943, 1980),\n            "egocar_visible": False\n        }},\n'
        for i, c in enumerate(cams)) + "    },\n"
    s = s.replace("DATASETS_CONFIG = {\n", "DATASETS_CONFIG = {\n" + entry, 1)
    open(meta, "w").write(s)
vis = "utils/visualization.py"
s = open(vis).read()
if "layout_truckscenes" not in s:
    s = s.replace('    elif dataset_type == "kitti":\n        layout = layout_kitti',
                  '    elif dataset_type == "truckscenes":\n        layout = layout_truckscenes\n'
                  '    elif dataset_type == "kitti":\n        layout = layout_kitti', 1)
    s += '''

def layout_truckscenes(
    imgs: List[np.array], cam_names: List[str]
) -> np.array:
    """Tile TruckScenes cameras.
        #####################################################
        # CAMERA_LEFT_FRONT  #  CAMERA_RIGHT_FRONT  #
        #####################################################
        # CAMERA_LEFT_BACK   #  CAMERA_RIGHT_BACK   #   (only if loaded)
        #####################################################
    """
    pos = {"CAMERA_LEFT_FRONT": (0, 0), "CAMERA_RIGHT_FRONT": (0, 1),
           "CAMERA_LEFT_BACK": (1, 0), "CAMERA_RIGHT_BACK": (1, 1)}
    h, w, ch = imgs[0].shape
    rows = 2 if any(pos[n][0] == 1 for n in cam_names) else 1
    tiled = np.zeros((h * rows, w * 2, ch), dtype=np.float32)
    for img, name in zip(imgs, cam_names):
        r, c = pos[name]
        tiled[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return tiled
'''
    open(vis, "w").write(s)
print("patched")
