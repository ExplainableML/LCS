import numpy as np
from PIL import Image
from pathlib import Path
from skimage.color import rgb2lab, deltaE_ciede2000
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["walls", "objects"],
)
args = parser.parse_args()

if args.mode == "walls":
    GT_DIR = Path("visualization_walls/vae")
    PRED_DIR = Path("visualization_walls/observations")
else:  # objects
    GT_DIR = Path("visualization_objects/vae")
    PRED_DIR = Path("visualization_objects/observations")

TS = [0, 10, 20, 30, 40, 50]
BLOCK = 16  # 512 -> 32

def load_rgb(path):
    return np.asarray(
        Image.open(path).convert("RGB"),
        dtype=np.float32
    ) / 255.0

def downscale_block_avg(img, block):
    h, w, c = img.shape
    img = img.reshape(
        h // block, block,
        w // block, block,
        c
    )
    return img.mean(axis=(1, 3))


# --- accumulators ---
delta_pred_gt = {t: [] for t in TS}        # pred_t vs gt_t
delta_pred_gt50 = {t: [] for t in TS}     # pred_t vs gt_50
delta_gt_gt50 = {t: [] for t in TS}       # gt_t vs gt_50

delta_avg_pred_gt = {t: [] for t in TS}
delta_avg_pred_gt50 = {t: [] for t in TS}
delta_avg_gt_gt50 = {t: [] for t in TS}


for prompt_dir in GT_DIR.iterdir():
    if not prompt_dir.is_dir():
        continue

    pred_prompt_dir = PRED_DIR / prompt_dir.name
    if not pred_prompt_dir.exists():
        continue

    gt50_path = prompt_dir / "t50.png"
    if not gt50_path.exists():
        continue

    gt50 = load_rgb(gt50_path)
    gt50_32 = downscale_block_avg(gt50, BLOCK)
    gt50_lab = rgb2lab(gt50_32)

    gt50_avg_lab = rgb2lab(
        gt50.mean(axis=(0, 1), keepdims=True)
    ).squeeze()

    for t in TS:
        name = f"t{t}.png"
        gt_path = prompt_dir / name
        pred_path = pred_prompt_dir / name

        if not gt_path.exists() or not pred_path.exists():
            continue

        gt = load_rgb(gt_path)
        pred = load_rgb(pred_path)

        gt_32 = downscale_block_avg(gt, BLOCK)
        gt_lab = rgb2lab(gt_32)
        pred_lab = rgb2lab(pred)

        # --- per-pixel ΔE2000 ---
        delta_pred_gt_t = deltaE_ciede2000(pred_lab, gt_lab)
        delta_pred_gt50_t = deltaE_ciede2000(pred_lab, gt50_lab)
        delta_gt_gt50_t = deltaE_ciede2000(gt_lab, gt50_lab)

        delta_pred_gt[t].append(delta_pred_gt_t.mean())
        delta_pred_gt50[t].append(delta_pred_gt50_t.mean())
        delta_gt_gt50[t].append(delta_gt_gt50_t.mean())

        # --- global average color ΔE2000 ---
        gt_avg_lab = rgb2lab(
            gt.mean(axis=(0, 1), keepdims=True)
        ).squeeze()

        pred_avg_lab = rgb2lab(
            pred.mean(axis=(0, 1), keepdims=True)
        ).squeeze()

        delta_avg_pred_gt[t].append(
            deltaE_ciede2000(
                pred_avg_lab[None, None, :],
                gt_avg_lab[None, None, :]
            )[0, 0]
        )

        delta_avg_pred_gt50[t].append(
            deltaE_ciede2000(
                pred_avg_lab[None, None, :],
                gt50_avg_lab[None, None, :]
            )[0, 0]
        )

        delta_avg_gt_gt50[t].append(
            deltaE_ciede2000(
                gt_avg_lab[None, None, :],
                gt50_avg_lab[None, None, :]
            )[0, 0]
        )


print("\nΔE2000 (per-pixel, pred_t vs gt_50):")
for t in TS:
    if delta_pred_gt50[t]:
        print(f"t{t:>2}: {np.mean(delta_pred_gt50[t]):.0f}")

print("\nΔE2000 (per-pixel, gt_t vs gt_50):")
for t in TS:
    if delta_gt_gt50[t]:
        print(f"t{t:>2}: {np.mean(delta_gt_gt50[t]):.0f}")

print("\nΔE2000 (global avg color, pred_t vs gt_50):")
for t in TS:
    if delta_avg_pred_gt50[t]:
        print(f"t{t:>2}: {np.mean(delta_avg_pred_gt50[t]):.0f}")

print("\nΔE2000 (global avg color, gt_t vs gt_50):")
for t in TS:
    if delta_avg_gt_gt50[t]:
        print(f"t{t:>2}: {np.mean(delta_avg_gt_gt50[t]):.0f}")


# --- stable expected ΔE2000 between all gt_50 images ---
gt50_paths = [p / "t50.png" for p in GT_DIR.iterdir() if (p / "t50.png").exists()]
num_gt50 = len(gt50_paths)

if num_gt50 >= 2:
    gt50_32_list = []
    gt50_avg_lab_list = []

    for path in gt50_paths:
        img = load_rgb(path)
        img_32 = downscale_block_avg(img, BLOCK)
        gt50_32_list.append(rgb2lab(img_32))
        gt50_avg_lab_list.append(
            rgb2lab(img.mean(axis=(0, 1), keepdims=True)).squeeze()
        )

    delta_pixel_sum = 0.0
    delta_avg_sum = 0.0
    num_pairs = 0

    for i in range(num_gt50):
        for j in range(i + 1, num_gt50):
            delta_pixel = deltaE_ciede2000(
                gt50_32_list[i],
                gt50_32_list[j]
            )
            delta_pixel_sum += delta_pixel.mean()

            delta_avg = deltaE_ciede2000(
                gt50_avg_lab_list[i][None, None, :],
                gt50_avg_lab_list[j][None, None, :]
            )[0, 0]

            delta_avg_sum += delta_avg
            num_pairs += 1

    print("\nStable expected ΔE2000 between all gt_50 images:")
    print(f"Per-pixel mean ΔE2000: {delta_pixel_sum / num_pairs:.2f}")
    print(f"Global average ΔE2000: {delta_avg_sum / num_pairs:.2f}")
else:
    print("\nNot enough gt_50 images to compute stable baseline ΔE2000.")
