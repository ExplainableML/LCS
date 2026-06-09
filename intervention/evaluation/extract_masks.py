import json
from pathlib import Path

import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector

# -------------------------
# CONFIGURATION
# -------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = "/lustre/groups/akata/code/bader/geneval/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
MODEL_CHECKPOINT = "/lustre/groups/akata/code/bader/geneval/obj_detection_folder/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"

CONF_THRESHOLD = 0.0

ROOT = Path(
    "/lustre/groups/akata/code/bader/comp_gen_oct25/LCS_camera_ready/imgs-natural-local-interp-t9"
)

DET_MASK_OUT_ROOT = Path(
    "/lustre/groups/akata/code/bader/comp_gen_oct25/LCS_camera_ready/masks/imgs-natural-local-interp-t9"
)

CLASSNAMES_PATH = "/lustre/groups/akata/code/bader/geneval/evaluation/object_names.txt"
with open(CLASSNAMES_PATH) as f:
    classnames = [line.strip() for line in f]

# -------------------------
# LOAD MODEL
# -------------------------

print("Loading Mask2Former...", flush=True)
object_detector = init_detector(
    MODEL_CONFIG, MODEL_CHECKPOINT, device=DEVICE
)

# -------------------------
# HELPERS
# -------------------------

def parse_class_from_filename(png_path: Path) -> str:
    """
    Expected filename format:
    <anything>-<class>-<anything>.png
    Example: Dark_Azure-fire_hydrant-3.png
    """
    stem = png_path.stem
    parts = stem.split("-")

    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {png_path.name}")

    # Convert underscores to spaces to match COCO class names
    return parts[1].replace("_", " ")

# -------------------------
# MAIN LOOP
# -------------------------

for idx, png_path in enumerate(ROOT.glob("*.png")):
    try:
        target_class = parse_class_from_filename(png_path)
    except ValueError as e:
        print(e)
        continue

    if target_class not in classnames:
        print(f"Class '{target_class}' not in classnames, skipping {png_path.name}")
        continue

    target_idx = classnames.index(target_class)

    # -------------------------
    # DETECTION
    # -------------------------

    result = inference_detector(object_detector, png_path)

    if not isinstance(result, tuple):
        print(f"No segmentation output for {png_path.name}, skipping...")
        continue

    bbox_result, segm_result = result

    cls_boxes = bbox_result[target_idx]
    cls_masks = segm_result[target_idx]

    best_score = -1
    best_det_mask = None

    for i, box in enumerate(cls_boxes):
        score = box[4]
        if score >= CONF_THRESHOLD and score > best_score:
            best_score = score
            best_det_mask = cls_masks[i]

    if best_det_mask is None:
        print(f"No valid detection for '{target_class}' in {png_path.name}")
        continue

    # -------------------------
    # SAVE MASK
    # -------------------------

    det_mask_out_path = DET_MASK_OUT_ROOT / png_path.stem
    det_mask_out_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(det_mask_out_path, best_det_mask.astype(np.uint8))

    if idx % 100 == 0:
        print(
            f"Saved detector mask -> {det_mask_out_path}.npy",
            flush=True,
        )
