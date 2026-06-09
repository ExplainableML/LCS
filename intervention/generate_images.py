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
    'black': '#222222',
    'blue': '#0515f5',
    'brown': '#964B00',
    'green': '#05f561',
    'orange': '#f59905',
    'pink': '#ed05f5',
    'purple': '#8d05f5',
    'red': '#f50505',
    'white': '#DDDDDD',
    'yellow': '#f5ed05',
    'tight_qualitative_red': '#d81612',
    'tight_qualitative_orange': '#ea710c',
    'tight_qualitative_yellow': '#f3db1a',
    'tight_qualitative_green': '#3ffe3f',
    'tight_qualitative_blue': '#3f79bf',
    'tight_qualitative_purple': '#9360b4',
    'tight_qualitative_pink': '#f527e0',
    'tight_qualitative_black': '#222222',
    'tight_qualitative_white': '#DDDDDD',
    'hue_qualitative_rm1': "#E60000",
    'hue_qualitative_rm2': "#E6002E",
    'hue_qualitative_rm3': "#E6005C",
    'hue_qualitative_rm4': "#E6008A",
    'hue_qualitative_rm5': "#E600B8",
    'hue_qualitative_rm6': "#E600E6",
    'hue_qualitative_bg1': "#0000CC",
    'hue_qualitative_bg2': "#1A1AE6",
    'hue_qualitative_bg3': "#3333CC",
    'hue_qualitative_bg4': "#4D4DB3",
    'hue_qualitative_bg5': "#666699",
    'hue_qualitative_bg6': "#808080",
    'hue_qualitative_wrb1': "#DDDDDD",
    'hue_qualitative_wrb2': "#F2B6B6",
    'hue_qualitative_wrb3': "#d81612",
    'hue_qualitative_wrb4': "#990000",
    'hue_qualitative_wrb5': "#330000",
    'hue_qualitative_wrb6': "#222222",
}

BATCH_SIZE = 1   # keep fixed unless you want multiple samples per run

def find_token_ids(tokens, text):
    """
    Given a list of tokens and a target string, return the indices of the tokens
    that concatenate to form the string.
    """
    print(tokens, text)
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

def str_list(arg):
    try:
        return [str(x) for x in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a comma-separated list of strings")

def float_list(arg):
    try:
        return [float(x) for x in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a comma-separated list of floats")

def remove_colors(prompt: str, colors=COLORS) -> str:
    color_pattern = re.compile(
        r'\b(' + '|'.join(map(re.escape, colors.keys())) + r')\b',
        flags=re.IGNORECASE
    )

    seen_color = False

    def replacer(match):
        nonlocal seen_color
        color = match.group(0).lower()

        # Keep "orange" if a color has already appeared
        if color == "orange" and seen_color:
            return match.group(0)

        # Otherwise remove the color
        seen_color = True
        return ""

    result = color_pattern.sub(replacer, prompt)
    return re.sub(r'\s+', ' ', result).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--objects", type=str_list, default=[])
    parser.add_argument("--colors", type=str_list, default=[])
    parser.add_argument("--replace_timestep", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seg_block", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_imgs", type=int, default=1)
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--remove_color", action="store_true",help="Remove color from prompt if flag is set")
    parser.add_argument("--global_mod", action="store_true",help="Remove color from prompt if flag is set")
    cmd_args = parser.parse_args()

    pipe = FluxPipeline.from_pretrained(
        FLUX_PATH,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    list_colors = [COLORS[c] for c in cmd_args.colors]

    generator = torch.Generator(device="cuda").manual_seed(cmd_args.seed)

    prompt = cmd_args.prompt

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

    if cmd_args.global_mod:
        list_text_ids = [-1 for _ in range(len(list_colors))]
    else:
        list_text_ids = [find_token_ids(tokens, obj_inst)[0] for obj_inst in cmd_args.objects]
    print(list_colors, list_text_ids)
    if len(list_text_ids) != len(list_colors) or len(list_text_ids) < 1:
        raise ValueError(f"objects or colors wrong: {list_colors}, {list_text_ids}")

    for j in range(len(pipe.transformer.transformer_blocks)):
        pipe.transformer.transformer_blocks[j].attn.processor = \
            FluxAttnProcessor2_0_visualize(
                block_id=j,
                token_start=0,
                token_end=num_tokens,
                save_path=""
            )

    print(f"Running: {prompt} | t={cmd_args.replace_timestep}")
    generator = torch.Generator(device="cuda").manual_seed(cmd_args.seed)
    for j in range(cmd_args.num_imgs):
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

        imgs.images[0].save(cmd_args.out_path)
