import torch
import os
import colorsys
from PIL import Image
from diffusers import FluxPipeline

FLUX_PATH = "black-forest-labs/FLUX.1-dev"
OUT_DIR = "hsv_latents_8"
NUM_SAMPLES = 512
IMAGE_SIZE = 512

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda"

pipeline = FluxPipeline.from_pretrained(
    FLUX_PATH, torch_dtype=torch.bfloat16
).to(device)

vae = pipeline.vae

# HSV grid resolution (cube root approx)
grid = 8

samples = []

for h in range(grid + 1):
    for s in range(grid):
        for v in range(grid):
            samples.append((
                h / (grid),
                s / (grid - 1),
                v / (grid - 1),
            ))


print(f"Generating {len(samples)} HSV samples")

for idx, (h, s, v) in enumerate(samples):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    rgb = (int(r * 255), int(g * 255), int(b * 255))
    
    # Convert RGB to HEX string
    hexcode = "#{:02X}{:02X}{:02X}".format(*rgb)
    
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), rgb)

    image_tensor = pipeline.image_processor.preprocess(image).to(
        device=device, dtype=torch.bfloat16
    )

    with torch.no_grad():
        dist = vae.encode(image_tensor)
        latents = dist.latent_dist.sample()

        latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor

        latents = pipeline._pack_latents(
            latents,
            1,
            pipeline.transformer.config.in_channels // 4,
            2 * (IMAGE_SIZE // (pipeline.vae_scale_factor * 2)),
            2 * (IMAGE_SIZE // (pipeline.vae_scale_factor * 2)),
        )

    # Save using HEX code as filename
    out_path = os.path.join(OUT_DIR, f"{hexcode}.pt")
    torch.save(latents.cpu(), out_path)

    if idx % 25 == 0:
        print(f"Saved {idx}/{len(samples)}")

print("Done.")