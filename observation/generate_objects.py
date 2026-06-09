import os
import json
import torch
import argparse

from diffusers import FluxPipeline
from flux_replace_latent import flux_replace_latents
from flux_visualize_attention import FluxAttnProcessor2_0_visualize

TOKEN = "<hf-token>"
FLUX_PATH = "black-forest-labs/FLUX.1-dev"


def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="images_objects")
    parser.add_argument("--color_name", type=str, default="#FFFFFF")
    parser.add_argument("--seg_block", type=int, default=18)
    parser.add_argument("--text_id", type=int, default=-1)
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--save_dir", type=str, default="visualization_objects/")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--replace_timestep", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Load pipeline once
    pipe = FluxPipeline.from_pretrained(
        FLUX_PATH, torch_dtype=torch.bfloat16
    ).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Iterate over JSONL prompts
    for entry in load_jsonl(args.jsonl_path):
        prompt = entry["prompt"]

        # Create output directory: images/{prompt}/
        prompt_dir = os.path.join(args.out_root, prompt)
        os.makedirs(prompt_dir, exist_ok=True)

        # Tokenize once per prompt (for attention viz)
        text_inputs = pipe.tokenizer_2(
            prompt,
            padding=False,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        tokens = pipe.tokenizer_2.convert_ids_to_tokens(
            text_inputs["input_ids"][0]
        )
        num_tokens = len(tokens)

        # Attach attention visualizers
        for i, block in enumerate(pipe.transformer.transformer_blocks):
            block.attn.processor = FluxAttnProcessor2_0_visualize(
                block_id=i,
                token_start=0,
                token_end=num_tokens,
                save_path=""
            )

        # Generate image(s)
        imgs = flux_replace_latents(
            pipe,
            args.replace_timestep,
            [args.color_name],
            prompt_dir,
            args.seg_block,
            [args.text_id],
            None,
            args.save_visualizations,
            args.save_dir,
            prompt,
            num_inference_steps=args.num_inference_steps,
            width=512,
            height=512,
            num_images_per_prompt=args.batch_size,
            generator=generator
        )

        # Save first image as img.png
        imgs.images[0].save(os.path.join(prompt_dir, "img.png"))

        print(f"Saved: {prompt_dir}/img.png")
