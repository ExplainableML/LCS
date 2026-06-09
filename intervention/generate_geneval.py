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
    parser.add_argument("--prompt_path", type=str, default='/lustre/groups/akata/code/bader/comp_gen_oct25/flux_geneval_color_nocolor/evaluation_metadata_colors.jsonl')
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--replace_timestep", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seg_block", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_imgs", type=int, default=4)
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--split_start", type=int, default=0)
    parser.add_argument("--split_end", type=int, default=10000)
    parser.add_argument("--remove_color", action="store_true",help="Remove color from prompt if flag is set")
    parser.add_argument("--global_mod", action="store_true",help="Remove color from prompt if flag is set")
    cmd_args = parser.parse_args()

    pipe = FluxPipeline.from_pretrained(
        FLUX_PATH,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(cmd_args.seed)

    with open(cmd_args.prompt_path) as fp:
        prompt_dicts = [json.loads(line) for line in fp]

    for i, prompt_dict in tqdm(enumerate(prompt_dicts)):
        if i < cmd_args.split_start or i >= cmd_args.split_end:
            continue

        prompt = prompt_dict['prompt']
        if cmd_args.remove_color:
            prompt = remove_colors(prompt)

        out_path1 = os.path.join(cmd_args.out_path, str(i).zfill(5))
        os.makedirs(out_path1, exist_ok=True)

        with open(os.path.join(out_path1, "metadata.jsonl"), "w") as f:
            f.write(json.dumps(prompt_dict) + "\n")
        out_img_path1 = os.path.join(out_path1, 'samples')
        os.makedirs(out_img_path1, exist_ok=True)

        color_str = prompt_dict["include"][0]["color"]
        color = COLORS[color_str]

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
            text_id = -1
        else:
            text_id = find_token_ids(tokens, prompt_dict["include"][0]["class"])[0]

        for j in range(len(pipe.transformer.transformer_blocks)):
            pipe.transformer.transformer_blocks[j].attn.processor = \
                FluxAttnProcessor2_0_visualize(
                    block_id=j,
                    token_start=0,
                    token_end=num_tokens,
                    save_path=""
                )

        print(f"Running: {prompt} | token={text_id} | color={color} | t={cmd_args.replace_timestep} | g=None")
        generator = torch.Generator(device="cuda").manual_seed(cmd_args.seed)
        for j in range(cmd_args.num_imgs):
            imgs = flux_replace_latents(
                pipe,
                cmd_args.replace_timestep,
                [color],
                cmd_args.out_path,
                cmd_args.seg_block,
                [text_id],
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

            print(os.path.join(out_img_path1, f"{str(j).zfill(4)}.png"))
            imgs.images[0].save(os.path.join(out_img_path1, f"{str(j).zfill(4)}.png"))
