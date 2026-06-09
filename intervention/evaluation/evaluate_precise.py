import os
from pathlib import Path
import cv2
import numpy as np
import re
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from skimage.color import deltaE_ciede2000
from skimage import color as skcolor

COLORS = {
    'Red': '#FF0000',
    'Orange': '#FF7F00',
    'Yellow': '#FFFF00',
    'Chartreuse': '#7FFF00',
    'Green': '#00FF00',
    'Spring_Green': '#00FF7F',
    'Cyan': '#00FFFF',
    'Azure': '#007FFF',
    'Blue': '#0000FF',
    'Violet': '#7F00FF',
    'Magenta': '#FF00FF',
    'Rose': '#FF007F',
    'Dark_Red': '#7F0000',
    'Dark_Orange': '#7F3F00',
    'Dark_Yellow': '#7F7F00',
    'Dark_Chartreuse': '#3F7F00',
    'Dark_Green': '#007F00',
    'Dark_Spring_Green': '#007F3F',
    'Dark_Cyan': '#007F7F',
    'Dark_Azure': '#003F7F',
    'Dark_Blue': '#00007F',
    'Dark_Violet': '#3F007F',
    'Dark_Magenta': '#7F007F',
    'Dark_Rose': '#7F003F',
    'Light_Red': '#FF7F7F',
    'Light_Orange': '#FFBF7F',
    'Light_Yellow': '#FFFF7F',
    'Light_Chartreuse': '#BFFF7F',
    'Light_Green': '#7FFF7F',
    'Light_Spring_Green': '#7FFFBF',
    'Light_Cyan': '#7FFFFF',
    'Light_Azure': '#7FBFFF',
    'Light_Blue': '#7F7FFF',
    'Light_Violet': '#BF7FFF',
    'Light_Magenta': '#FF7FFF',
    'Light_Rose': '#FF7FBF',
    'Muted_Red': '#BF4040',
    'Muted_Orange': '#BF7F40',
    'Muted_Yellow': '#BFBF40',
    'Muted_Chartreuse': '#7FBF40',
    'Muted_Green': '#40BF40',
    'Muted_Spring_Green': '#40BF7F',
    'Muted_Cyan': '#40BFBF',
    'Muted_Azure': '#407FBF',
    'Muted_Blue': '#4040BF',
    'Muted_Violet': '#7F40BF',
    'Muted_Magenta': '#BF40BF',
    'Muted_Rose': '#BF407F',
    'Black': '#000000',
    'White': '#FFFFFF',
    'Gray': '#808080',
}

BASIC_COLORS = [
    'Red', 'Orange', 'Yellow', 'Chartreuse', 
    'Green', 'Spring_Green', 'Cyan', 'Azure', 
    'Blue', 'Violet', 'Magenta', 'Rose', 'Black',
    'White', 'Gray'
    ]

def hex_to_rgb(hex_color):
    """
    Convert HEX color string to RGB tuple.

    Args:
        hex_color (str): HEX color, e.g. "#804020" or "804020"

    Returns:
        tuple: (R, G, B) each in range 0–255
    """
    # Remove leading '#' if present
    hex_color = hex_color.lstrip('#')

    if len(hex_color) != 6:
        raise ValueError("HEX color must be 6 characters long")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)

def load_image(img_path):
    # --- Load image ---
    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not load image")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks, save_path):
    """
    Overlay masks on an image and save to file instead of displaying.

    Args:
        raw_image: PIL Image or NumPy array.
        masks: List of masks (NumPy arrays).
        save_path: Path to save the output image.
    """
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)
    
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"Annotated image saved to {save_path}", flush=True)
    plt.close()  # Close figure to free memory
    
    del mask
    gc.collect()

def delta_e_ciede2000(image_rgb, mask, rgb_color):
    """
    image_rgb : np.ndarray (H, W, 3), RGB uint8
    mask      : bool array (H, W)
    rgb_color : tuple(int, int, int)

    Returns:
        mean_delta_e : float
    """

    # --- Extract masked RGB pixels ---
    masked_pixels = image_rgb[mask]

    if masked_pixels.size == 0:
        raise ValueError("Mask contains no valid pixels")

    # --- Mean RGB color ---
    mean_rgb = masked_pixels.mean(axis=0).astype(np.float32)

    # --- Convert mean RGB -> Lab ---
    mean_rgb_norm = mean_rgb[None, None, :] / 255.0
    mean_lab = skcolor.rgb2lab(mean_rgb_norm)[0, 0]

    # --- Reference RGB -> Lab ---
    ref_rgb = np.array(rgb_color, dtype=np.float32)[None, None, :] / 255.0
    ref_lab = skcolor.rgb2lab(ref_rgb)[0, 0]

    # --- ΔE CIEDE2000 ---
    delta_e = deltaE_ciede2000(
        mean_lab[None, :],
        ref_lab[None, :]
    )[0]

    return float(delta_e)

def distance_hsl(image_rgb, mask, rgb_color):

    # --- Extract masked RGB pixels ---
    masked_pixels = image_rgb[mask]

    if masked_pixels.size == 0:
        raise ValueError("Mask contains no valid pixels")

    # --- Mean RGB of masked region ---
    mean_rgb = masked_pixels.mean(axis=0).astype(np.uint8)

    # --- Convert mean RGB to HLS ---
    mean_hls = cv2.cvtColor(
        np.uint8([[mean_rgb]]),
        cv2.COLOR_RGB2HLS
    ).astype(np.float32)[0, 0]

    # --- Convert reference RGB to HLS ---
    ref_hls = cv2.cvtColor(
        np.uint8([[rgb_color]]),
        cv2.COLOR_RGB2HLS
    ).astype(np.float32)[0, 0]

    # --- Hue circular distance ---
    hue_diff = abs(mean_hls[0] - ref_hls[0])
    hue_error = min(hue_diff, 180.0 - hue_diff)

    # --- Linear distances ---
    lightness_error = abs(mean_hls[1] - ref_hls[1])
    saturation_error = abs(mean_hls[2] - ref_hls[2])

    return np.array([
        hue_error,
        lightness_error,
        saturation_error
    ])

v = 'imgs-plain-global-interp-t9'
no_mask = False

img_folder = f'/lustre/groups/akata/code/bader/comp_gen_oct25/LCS_camera_ready/{v}'
mask_folder = f'/lustre/groups/akata/code/bader/comp_gen_oct25/LCS_camera_ready/masks/{v}'

save_path =  f'/lustre/groups/akata/code/bader/comp_gen_oct25/LCS_camera_ready/evaluation/results/{v}'


folder = Path(img_folder)

all_imgs_hsl = []
by_color_hsl = [[] for _ in BASIC_COLORS]
dark_imgs_hsl = []
light_imgs_hsl = []
muted_imgs_hsl = []
bright_imgs_hsl = []

all_imgs_2k = []
by_color_2k = [[] for _ in BASIC_COLORS]
dark_imgs_2k = []
light_imgs_2k = []
muted_imgs_2k = []
bright_imgs_2k = []

skipped = 0

for i, file in tqdm(enumerate(folder.iterdir())):
    if file.is_file():
        file_name = str(file).split('/')[-1].split('.')[0]

        color = file_name.split('-')[0]
        img_path = os.path.join(img_folder, file)
        image_rgb = load_image(img_path)
        color_hex = COLORS[color]

        color_rgb = hex_to_rgb(color_hex)

        # the mask will not be there if the target object was not detected: then the color will not be
        # used to calculate average color, but the number of images skipped for this reason will be counted
        try:  
            if no_mask:
                mask = np.ones((512, 512), dtype=bool)
            else:
                mask = np.load(f"{mask_folder}/{file_name}.npy") == 1
        except:
            skipped += 1
            print(f'Skipped {file_name}')
            continue

        mean_hsl = distance_hsl(image_rgb, mask, color_rgb)
        mean_2k = delta_e_ciede2000(image_rgb, mask, color_rgb)

        all_imgs_hsl.append(mean_hsl)
        all_imgs_2k.append(mean_2k)

        for j, c in enumerate(BASIC_COLORS):
            if c in color:
                by_color_hsl[j].append(mean_hsl)
                by_color_2k[j].append(mean_2k)

        if 'Dark' in color:
            dark_imgs_hsl.append(mean_hsl)
            dark_imgs_2k.append(mean_2k)
        elif 'Light' in color:
            light_imgs_hsl.append(mean_hsl)
            light_imgs_2k.append(mean_2k)
        elif 'Muted' in color:
            muted_imgs_hsl.append(mean_hsl)
            muted_imgs_2k.append(mean_2k)
        elif color not in ['Black', 'White', 'Gray']: 
            bright_imgs_hsl.append(mean_hsl)
            bright_imgs_2k.append(mean_2k)

        print(file_name, mean_hsl, mean_2k, flush=True)
    if i % 100 == 0 and len(all_imgs_2k) > 0:
        print(f"Average all: {np.mean(np.stack(all_imgs_2k), axis=0):.2f}", flush=True)

avg_all_hsl = np.mean(np.stack(all_imgs_hsl), axis=0)
avg_all_2k = np.mean(np.stack(all_imgs_2k), axis=0)
avg_by_color_hsl = []
avg_by_color_2k = []
for ind_color_hsl, ind_color_2k in zip(by_color_hsl, by_color_2k):
    avg_by_color_hsl.append(np.mean(np.stack(ind_color_hsl), axis=0))
    avg_by_color_2k.append(np.mean(np.stack(ind_color_2k), axis=0))
try:
    avg_dark_hsl = np.mean(np.stack(dark_imgs_hsl), axis=0)
    avg_light_hsl = np.mean(np.stack(light_imgs_hsl), axis=0)
    avg_muted_hsl = np.mean(np.stack(muted_imgs_hsl), axis=0)
    avg_bright_hsl = np.mean(np.stack(bright_imgs_hsl), axis=0)
    avg_dark_2k = np.mean(np.stack(dark_imgs_2k), axis=0)
    avg_light_2k = np.mean(np.stack(light_imgs_2k), axis=0)
    avg_muted_2k = np.mean(np.stack(muted_imgs_2k), axis=0)
    avg_bright_2k = np.mean(np.stack(bright_imgs_2k), axis=0)
except:
    avg_dark_hsl = float('nan')
    avg_light_hsl = float('nan')
    avg_muted_hsl = float('nan')
    avg_bright_hsl = float('nan')
    avg_dark_2k = float('nan')
    avg_light_2k = float('nan')
    avg_muted_2k = float('nan')
    avg_bright_2k = float('nan')

with open(save_path, "w") as f:
    f.write(f"Average all HSL: {avg_all_hsl}\n")
    for c, avg in zip(BASIC_COLORS, avg_by_color_hsl):
        f.write(f"Average {c} HSL: {avg}\n")
    f.write(f"Average dark HSL: {avg_dark_hsl}\n")
    f.write(f"Average light HSL: {avg_light_hsl}\n")
    f.write(f"Average muted HSL: {avg_muted_hsl}\n")
    f.write(f"Average bright HSL: {avg_bright_hsl}\n")

    f.write(f"Average all 2k: {avg_all_2k:.2f}\n")
    for c, avg in zip(BASIC_COLORS, avg_by_color_2k):
        f.write(f"Average {c} 2K: {avg:.2f}\n")
    f.write(f"Average dark 2k: {avg_dark_2k:.2f}\n")
    f.write(f"Average light 2k: {avg_light_2k:.2f}\n")
    f.write(f"Average muted 2k: {avg_muted_2k:.2f}\n")
    f.write(f"Average bright 2k: {avg_bright_2k:.2f}\n")

    f.write(f"Total Num Imgs: {len(all_imgs_2k)}\n")
    f.write(f"Skipped: {skipped}\n")

