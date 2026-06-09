from PIL import Image
from pathlib import Path

INPUT_DIR = Path("visualization_quali_intervene/observations")
OUTPUT_DIR = Path("visualization_quali_intervene/observations_up")
SCALE = 16

for png_path in INPUT_DIR.rglob("*.png"):
    # Preserve relative folder structure
    relative_path = png_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / relative_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(png_path) as img:
        new_size = (img.width * SCALE, img.height * SCALE)
        img_big = img.resize(new_size, Image.NEAREST)
        img_big.save(out_path)

    print(f"Scaled: {png_path} → {out_path}")
