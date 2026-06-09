import torch
from diffusers import FluxPipeline
import os
import argparse
import re
from tqdm import tqdm
import json

from flux_replace_latent import flux_replace_latents
from flux_visualize_attention import FluxAttnProcessor2_0_visualize

FLUX_PATH = "black-forest-labs/FLUX.1-dev"

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

OBJS = [
    ("a close-up photo of a wall", 12),
    ("a close-up photo of a paper sheet", 8),
    ("a photo of a clear sky", 4),
    ("a close-up photo of a plain sweater", 15),
    ("a close-up photo of a concrete floor", 5),
    ("a closeup of a plain rug", 3),
    ("a photo of a clear sky at night", 6),
    ("a close-up photo of sand", 0),
    ("a close-up photo of metal texture", 8),
    ("a close-up photo of wooden texture", 9)
]

SHORTS = {
    "a close-up photo of a wall": "wall",
    "a close-up photo of a paper sheet": "paper",
    "a photo of a clear sky": "sky",
    "a close-up photo of a plain sweater": "sweater",
    "a close-up photo of a concrete floor": "floor",
    "a closeup of a plain rug": "rug",
    "a photo of a clear sky at night": "night",
    "a close-up photo of sand": "sand",
    "a close-up photo of metal texture": "metal",
    "a close-up photo of wooden texture": "wood",
}

BATCH_SIZE = 1   # keep fixed unless you want multiple samples per run

def find_token_ids(tokens, text):
    """
    Given a list of tokens and a target string, return the indices of the tokens
    that concatenate to form the string.
    """
    text = text.replace(" ", "")
    # Normalize tokens by removing the SentencePiece boundary marker
    norm_tokens = [t.replace("▁", "") for t in tokens]
    n = len(norm_tokens)
    for start in range(n):
        if norm_tokens[start] == "":
            continue
        combined = ""
        for end in range(start, n):
            combined += norm_tokens[end]
            if combined == text:
                return list(range(start, end + 1))
            # Stop early if we've already gone too far
            if not text.startswith(combined):
                break

    return []  # no match found

def int_list(arg):
    try:
        return [int(x) for x in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a comma-separated list of integers")

def float_list(arg):
    try:
        return [float(x) for x in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a comma-separated list of floats")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--replace_timestep", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seg_block", type=int, default=18)
    parser.add_argument("--num_imgs", type=int, default=4)
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--split_start", type=int, default=0)
    parser.add_argument("--split_end", type=int, default=10000)
    cmd_args = parser.parse_args()

    pipe = FluxPipeline.from_pretrained(
        FLUX_PATH,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    os.makedirs(cmd_args.out_path, exist_ok=True)

    for prompt, seed in OBJS:
        for i, color in enumerate(COLORS.keys()):
            if i < cmd_args.split_start or i >= cmd_args.split_end:
                continue

            list_colors = [COLORS[color]]

            # tokenize per prompt so token indexing stays correct
            text_inputs = pipe.tokenizer_2(
                prompt,
                padding=False,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            tokens = pipe.tokenizer_2.convert_ids_to_tokens(text_inputs["input_ids"][0])
            num_tokens = len(tokens)

            print(prompt)
            print(tokens)
            list_text_ids = [-1 for _ in range(len(list_colors))]

            for j in range(len(pipe.transformer.transformer_blocks)):
                pipe.transformer.transformer_blocks[j].attn.processor = \
                    FluxAttnProcessor2_0_visualize(
                        block_id=j,
                        token_start=0,
                        token_end=num_tokens,
                        save_path=""
                    )

            print(f"Running: {prompt} | t={cmd_args.replace_timestep} | g=None")
            generator = torch.Generator(device="cuda").manual_seed(seed)
            imgs = flux_replace_latents(
                pipe,
                cmd_args.replace_timestep,
                list_colors,
                cmd_args.out_path,
                cmd_args.seg_block,
                list_text_ids,
                None,
                cmd_args.save_visualizations,
                cmd_args.save_dir,
                prompt,
                num_inference_steps=cmd_args.num_inference_steps,
                width=512,
                height=512,
                num_images_per_prompt=BATCH_SIZE,
                generator=generator
            )
            obj = SHORTS[prompt]
            img_path = os.path.join(cmd_args.out_path, f"{color}-{obj.replace(' ', '_')}.png")
            print(img_path)
            imgs.images[0].save(img_path)
