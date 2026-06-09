import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import copy
import os

import colorsys
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
""" 

def hex_to_hsl(hex_code):
    hex_code = hex_code.lstrip("#")
    r, g, b = [int(hex_code[i:i+2], 16)/255.0 for i in (0,2,4)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # note: HLS in Python
    return h, s, l

def proj(a, b):
    return (a @ b) / (a @ a + 1e-8) * a

def hsl_to_lcs(h, s, l, anchors_position, anchors_hex):
    b = anchors_position[6]
    w = anchors_position[7]
    a = w - b

    c_L = b + l * a

    # order anchors 
    hue_anchors_position = anchors_position[:6]
    hue_anchors_hex = anchors_hex[:6]
    hue_anchors_values = np.array([hex_to_hsl(hx)[0] for hx in hue_anchors_hex])
    idx = np.argsort(hue_anchors_values)
    thetas = hue_anchors_values[idx]
    thetas = np.append(thetas, thetas[0] + 1)
    hue_anchors_position = torch.stack([hue_anchors_position[i] for i in idx])
    hue_anchors_position = torch.vstack([hue_anchors_position, hue_anchors_position[0]])

    for i in range(len(thetas) - 1):
        if thetas[i] <= h <= thetas[i + 1]:
            alpha = (h - thetas[i]) / (thetas[i + 1] - thetas[i])

            o = torch.mean(hue_anchors_position)
            ds = (
                (hue_anchors_position[i] - o) / (torch.norm(hue_anchors_position[i] - o) + 1e-8),
                (hue_anchors_position[i + 1] - o) / (torch.norm(hue_anchors_position[i + 1] - o) + 1e-8),
            )
            psi = torch.acos(torch.clamp(ds[0] @ ds[1], -1, 1))

            d_H = torch.sin((1 - alpha) * psi) / torch.sin(psi) * ds[0] + torch.sin(alpha * psi) / torch.sin(psi) * ds[1]
            R = (1 - alpha) * torch.norm(hue_anchors_position[i] - o) + alpha * torch.norm(hue_anchors_position[i + 1] - o)
            c_H = s * R * (1 - abs(2 * l - 1)) * d_H + o

            c = c_H + (c_L - o)

            return c
    else:
        return c_L

def lcs_to_hsl(c, anchors_position, anchors_hex):
    b = anchors_position[6]
    w = anchors_position[7]
    a = w - b

    l = torch.clamp(torch.norm(proj(a, c - b)) / torch.norm(a), 0, 1)
    c_L = b + proj(a, c - b)

    # order anchors
    hue_anchors_position = anchors_position[:6]
    hue_anchors_hex = anchors_hex[:6]
    hue_anchors_values = np.array([hex_to_hsl(hx)[0] for hx in hue_anchors_hex])
    idx = np.argsort(hue_anchors_values)
    thetas = hue_anchors_values[idx]
    thetas = np.append(thetas, thetas[0] + 1)
    hue_anchors_position = torch.stack([hue_anchors_position[i] for i in idx])
    hue_anchors_position = torch.vstack([hue_anchors_position, hue_anchors_position[0]])

    o = torch.mean(hue_anchors_position)
    c_H = c + (o - c_L)

    # selecting closest anchors
    dists = torch.tensor([torch.norm(c_H - h_i) for h_i in hue_anchors_position[:-1]])
    k1, k2 = torch.topk(dists, k=2, largest=False).indices.tolist()
    if k2 < k1:
        k1, k2 = k2, k1
    if k1 == 0 and k2 == len(hue_anchors_position) - 2:
        k1, k2 = len(hue_anchors_position) - 2, len(hue_anchors_position) - 1
    k = k1

    hs = (
        hue_anchors_position[k],
        hue_anchors_position[k + 1],
    )

    alpha_nom = (c_H - o) @ (hs[0] - o) / (torch.norm(c_H - o) * torch.norm(hs[0] - o) + 1e-8)
    alpha_nom = float(torch.acos(torch.clamp(alpha_nom, 0, 1)))
    alpha_denom = (hs[1] - o) @ (hs[0] - o) / (torch.norm(hs[1] - o) * torch.norm(hs[0] - o) + 1e-8)
    alpha_denom = float(torch.acos(torch.clamp(alpha_denom, 0, 1)))
    alpha = alpha_nom / alpha_denom

    h = (thetas[k] + alpha * (thetas[k + 1] - thetas[k])) % 1.0
    R = (1 - alpha) * torch.norm(hue_anchors_position[k] - o) + alpha * torch.norm(hue_anchors_position[k + 1] - o)
    s = torch.norm(c - c_L) / (R * (1 - torch.abs(2 * l - 1)) + 1e-8)

    return float(h), float(s), float(l)

def hsl_to_hex(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    r = max(0, min(255, round(r * 255)))
    g = max(0, min(255, round(g * 255)))
    b = max(0, min(255, round(b * 255)))
    return f"#{r:02X}{g:02X}{b:02X}"

def save_pca_patch_projections(z, canonical_positions, canonical_hex, out_prefix="pca_patches"):
    """
    z : (N, 3) PCA coordinates of patches
    canonical_positions : (8, 3) tensor
    canonical_hex : list of 8 hex strings
    Saves two 2D projection plots with canonical anchors.
    """

    z = np.asarray(z.cpu())
    canon = np.asarray(canonical_positions.cpu())

    canon_rgb = [tuple(int(hx[i:i+2],16)/255 for i in (1,3,5)) for hx in canonical_hex]

    # -------- Plot 1 : PC0 vs PC1 --------
    plt.figure(figsize=(6,6))
    plt.scatter(z[:,0], z[:,1], s=6, c="black", alpha=0.5)

    for i in range(len(canon)):
        plt.scatter(canon[i,0], canon[i,1],
                    s=120, c=[canon_rgb[i]], edgecolors="black", linewidths=0.8, zorder=5)
        plt.text(canon[i,0], canon[i,1], f"{i}",
                 fontsize=9, ha="center", va="center", color="white", weight="bold")

    plt.xlabel("Black ↔ White")
    plt.ylabel("Hue Axis 1")
    plt.title("PCA Patch Projection: PC0 vs PC1")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pc0_pc1.png", dpi=200)
    plt.close()

    # -------- Plot 2 : PC0 vs PC2 --------
    plt.figure(figsize=(6,6))
    plt.scatter(z[:,0], z[:,2], s=6, c="black", alpha=0.5)

    for i in range(len(canon)):
        plt.scatter(canon[i,0], canon[i,2],
                    s=120, c=[canon_rgb[i]], edgecolors="black", linewidths=0.8, zorder=5)
        plt.text(canon[i,0], canon[i,2], f"{i}",
                 fontsize=9, ha="center", va="center", color="white", weight="bold")

    plt.xlabel("Black ↔ White")
    plt.ylabel("Hue Axis 2")
    plt.title("PCA Patch Projection: PC0 vs PC2")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pc0_pc2.png", dpi=200)
    plt.close()



    plt.figure(figsize=(6,6))
    plt.scatter(z[:,1], z[:,2], s=6, c="black", alpha=0.5)

    for i in range(len(canon)):
        plt.scatter(canon[i,1], canon[i,2],
                    s=120, c=[canon_rgb[i]], edgecolors="black", linewidths=0.8, zorder=5)
        plt.text(canon[i,1], canon[i,2], f"{i}",
                 fontsize=9, ha="center", va="center", color="white", weight="bold")

    plt.xlabel("Hue Axis 1")
    plt.ylabel("Hue Axis 2")
    plt.title("PCA Patch Projection: PC1 vs PC2")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pc1_pc2.png", dpi=200)
    plt.close()

def normalize_to_timestep(C, alpha_t, beta_t, alpha_final_t, beta_final_t):
    return (C - alpha_t) / (beta_t + 1e-8) * beta_final_t + alpha_final_t

def denormalize_from_timestep(C_hat, alpha_t, beta_t, alpha_final_t, beta_final_t):
    return (C_hat - alpha_final_t) / (beta_final_t + 1e-8) * beta_t + alpha_t

def steer_type_one(
    latents, B, mu,
    target_hex,
    anchors_position,
    anchors_hex,
    alphas, betas,
    t, final_t
):
    device = latents.device

    # enter LCS
    C = (latents.squeeze(0) - mu) @ B.T     # (L, 3)

    # scale t -> final_t
    alpha_t = alphas[t].to(device)
    beta_t  = betas[t].to(device)
    alpha_final_t = alphas[final_t].to(device)
    beta_final_t  = betas[final_t].to(device)
    C_hat = normalize_to_timestep(C, alpha_t, beta_t, alpha_final_t, beta_final_t)

    # intervene
    y_star = hex_to_hsl(target_hex)
    c_star = hsl_to_lcs(*y_star, anchors_position, anchors_hex)
    C_bar = C_hat.mean(dim=0, keepdim=True)
    C_hat_one = C_hat + (c_star.view(1, -1) - C_bar)

    # scale t -> final_t
    C_one = denormalize_from_timestep(C_hat_one, alpha_t, beta_t, alpha_final_t, beta_final_t)

    # leave LCS
    latents = latents.squeeze(0) + (C_one - C) @ B

    return latents.unsqueeze(0).to(torch.bfloat16)


def steer_type_two(
    latents, B, mu,
    target_hex,
    anchors_position,
    anchors_hex,
    alphas, betas,
    t, final_t
):
    device = latents.device

    # enter LCS
    C = (latents.squeeze(0) - mu) @ B.T

    # scale t -> final_t
    alpha_t = alphas[t].to(device)
    beta_t  = betas[t].to(device)
    alpha_final_t = alphas[final_t].to(device)
    beta_final_t  = betas[final_t].to(device)
    C_hat = normalize_to_timestep(C, alpha_t, beta_t, alpha_final_t, beta_final_t)

    # intervene
    Y = torch.stack([
        torch.tensor(lcs_to_hsl(c_hat, anchors_position, anchors_hex), device=device)
        for c_hat in C_hat
    ])

    angles = Y[:,0] * 2 * torch.pi
    mean_angle = torch.atan2(torch.sin(angles).mean(), torch.cos(angles).mean())
    y_hat = ((mean_angle / (2 * torch.pi)) % 1.0, Y[:,1].mean(), Y[:,2].mean())
    
    y_star = hex_to_hsl(target_hex)

    h_new = (Y[:,0] + ((y_star[0] - y_hat[0] + 0.5) % 1.0) - 0.5) % 1.0
    s_new = torch.clamp(Y[:,1] + y_star[1] - y_hat[1], 0, 1)
    l_new = torch.clamp(Y[:,2] + y_star[2] - y_hat[2], 0, 1)
    Y_two = torch.stack([h_new, s_new, l_new], dim=1)

    C_hat_two = torch.stack([
        hsl_to_lcs(*y_two, anchors_position, anchors_hex) for y_two in Y_two
    ])

    # scale t -> final_t
    C_two = denormalize_from_timestep(C_hat_two, alpha_t, beta_t, alpha_final_t, beta_final_t)

    # leave LCS
    latents = latents.squeeze(0) + (C_two - C) @ B

    return latents.unsqueeze(0).to(torch.bfloat16)


def get_mask(mask_block_id, transformer_blocks, text_id, prompt,
             alpha=0.80, blur_kernel=5, save_dir="masks"):

    all_masks = transformer_blocks[mask_block_id].attn.processor.mean_attn_map
    mask_float = all_masks[text_id]

    HW = mask_float.numel()
    H = W = int(HW ** 0.5)

    assert H * W == HW, "Attention map is not square!"

    mask_float = mask_float.view(H, W)

    # ---- STEP 1: Blur / fill holes FIRST ----
    mask_reshaped = mask_float.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    blurred = F.max_pool2d(
        mask_reshaped,
        kernel_size=blur_kernel,
        stride=1,
        padding=blur_kernel // 2
    )

    blurred = -F.max_pool2d(
        -blurred,
        kernel_size=blur_kernel,
        stride=1,
        padding=blur_kernel // 2
    )

    mask_float = blurred.squeeze(0).squeeze(0)

    # ---- STEP 2: Top-mass thresholding ----
    flat = mask_float.flatten()

    sorted_vals, sorted_idx = torch.sort(flat, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    total_mass = sorted_vals.sum()

    cutoff = torch.searchsorted(cumsum, alpha * total_mass)
    cutoff = int(min(cutoff.item(), flat.numel() - 1))

    binary_flat = torch.zeros_like(flat)
    binary_flat[sorted_idx[:cutoff + 1]] = 1

    mask_bool = binary_flat.view(H, W).bool()

    # ---- STEP 3: reshape to pipeline ----
    mask_bool_reshaped = mask_bool.flatten().unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]

    # ---- STEP 4: Save PNG ----
    os.makedirs(save_dir, exist_ok=True)

    mask_img = mask_bool.float() * 255
    mask_img = mask_img.cpu().numpy().astype("uint8")

    filename = f"{prompt}_mask.png"
    path = os.path.join(save_dir, filename)

    Image.fromarray(mask_img, mode="L").save(path)
    print("Saved mask:", path)

    return mask_bool_reshaped



def steer(
    B, mu, anchors_position, anchors_hex, alphas, betas, gamma, 
    latents, steer_timestep, target_hex,
    mask_block_id, model, text_id, prompt):

    for th, tid in zip(target_hex, text_id):
        if tid >= 0 and steer_timestep > 0:
            mask = get_mask(mask_block_id, model.transformer.transformer_blocks, tid, prompt)
            mask_flat = mask.squeeze(0).squeeze(-1)   # (N,)
            idx = mask_flat.nonzero(as_tuple=True)[0]
            latents_local = latents[:, idx, :]           # (1, K, 64)
        else:
            latents_local = latents

        latents_steered_type_one = steer_type_one(
            latents_local,
            B,
            mu,
            th,
            anchors_position,
            anchors_hex,
            alphas, 
            betas,
            steer_timestep, 
            50
        )

        latents_steered_type_two = steer_type_two(
            latents_local,
            B,
            mu,
            th,
            anchors_position,
            anchors_hex,
            alphas, 
            betas,
            steer_timestep, 
            50
        )
        
        if tid >= 0 and steer_timestep > 0:
            latents[:, idx, :] = gamma * latents_steered_type_one + (1 - gamma) * latents_steered_type_two
        else:
            latents[:, :, :] = gamma * latents_steered_type_one + (1 - gamma) * latents_steered_type_two

    return latents


def latents_to_rgb_image(
    latents, B, mu,
    anchors_position, anchors_hex,
    alphas, betas,
    t, final_t,
    save_dir):
    
    device = latents.device

    C = (latents.squeeze(0) - mu) @ B.T

    alpha_t = alphas[t].to(device)
    beta_t  = betas[t].to(device)

    alpha_final_t = alphas[final_t].to(device)
    beta_final_t  = betas[final_t].to(device)

    C_hat = normalize_to_timestep(C, alpha_t, beta_t, alpha_final_t, beta_final_t)

    save_path = os.path.join(save_dir, "patch_plot")
    os.makedirs(save_path, exist_ok=True)
    save_pca_patch_projections(C_hat, anchors_position, anchors_hex, os.path.join(save_path, f"{t}"))

    rgb = []
    for c_hat in C_hat:
        h, s, l = lcs_to_hsl(c_hat, anchors_position, anchors_hex)
        hx = hsl_to_hex(h, s, l)
        rgb.append([int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)])

    side = int(np.ceil(np.sqrt(len(rgb))))
    img = np.zeros((side, side, 3), dtype=np.uint8)
    for i, c in enumerate(rgb):
        img[i//side, i%side] = c

    return Image.fromarray(img)

def visualize(
    B, mu, anchors_position, anchors_hex, alphas, betas,
    latents, timestep, model, prompt, width, height, save_dir):

    img = latents_to_rgb_image(
        latents,
        B,
        mu,
        anchors_position,
        anchors_hex,
        alphas, 
        betas,
        timestep, 
        50,
        save_dir
    )

    output_dir = os.path.join(save_dir, "observations", f"{prompt}")
    os.makedirs(output_dir, exist_ok=True)
    img.save(os.path.join(output_dir, f"t{timestep}.png"))

    latents_step = model._unpack_latents(latents, height, width, model.vae_scale_factor)
    latents_step = (latents_step / model.vae.config.scaling_factor) + model.vae.config.shift_factor

    image = model.vae.decode(latents_step, return_dict=False)[0]
    image = model.image_processor.postprocess(image, output_type="pil")[0]

    output_dir = os.path.join(save_dir, "vae", f"{prompt}")
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(output_dir, f"t{timestep}.png"))



@torch.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def flux_replace_latents(
    model,
    steer_timestep,
    target_hex,
    out_path,
    mask_block_id,
    text_id,
    gamma,
    save_visualizations,
    save_dir,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    true_cfg_scale: float = 1.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_ip_adapter_image: Optional[PipelineImageInput] = None,
    negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            will be used instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
            not greater than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
        true_cfg_scale (`float`, *optional*, defaults to 1.0):
            True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
            `negative_prompt` is provided.
        height (`int`, *optional*, defaults to model.unet.config.sample_size * model.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
        width (`int`, *optional*, defaults to model.unet.config.sample_size * model.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 3.5):
            Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
            a model to generate images more aligned with `prompt` at the expense of lower image quality.

            Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
            the [paper](https://huggingface.co/papers/2210.03142) to learn more.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will be generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
            Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
            IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
            provided, embeddings are computed from the `ip_adapter_image` input argument.
        negative_ip_adapter_image:
            (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
            Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
            IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
            provided, embeddings are computed from the `ip_adapter_image` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `model.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(model: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

    Examples:

    Returns:
        [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
        is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
        images.
    """

    height = height or model.default_sample_size * model.vae_scale_factor
    width = width or model.default_sample_size * model.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    model.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    model._guidance_scale = guidance_scale
    model._joint_attention_kwargs = joint_attention_kwargs
    model._current_timestep = None
    model._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = model._execution_device

    lora_scale = (
        model.joint_attention_kwargs.get("scale", None) if model.joint_attention_kwargs is not None else None
    )
    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = model.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_text_ids,
        ) = model.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    # 4. Prepare latent variables
    num_channels_latents = model.transformer.config.in_channels // 4
    latents, latent_image_ids = model.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(model.scheduler.config, "use_flow_sigmas") and model.scheduler.config.use_flow_sigmas:
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        model.scheduler.config.get("base_image_seq_len", 256),
        model.scheduler.config.get("max_image_seq_len", 4096),
        model.scheduler.config.get("base_shift", 0.5),
        model.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        model.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * model.scheduler.order, 0)
    model._num_timesteps = len(timesteps)

    # handle guidance
    if model.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
        negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
    ):
        negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        negative_ip_adapter_image = [negative_ip_adapter_image] * model.transformer.encoder_hid_proj.num_ip_adapters

    elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
        negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
    ):
        ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        ip_adapter_image = [ip_adapter_image] * model.transformer.encoder_hid_proj.num_ip_adapters

    if model.joint_attention_kwargs is None:
        model._joint_attention_kwargs = {}

    image_embeds = None
    negative_image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = model.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )
    if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
        negative_image_embeds = model.prepare_ip_adapter_image_embeds(
            negative_ip_adapter_image,
            negative_ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )

    ##########
    B = torch.load(os.path.join("../find_lcs/lcs_params_6", "B.pt")).to(latents.device) # (3,64)
    mu = torch.load(os.path.join("../find_lcs/lcs_params_6", "mu.pt")).to(latents.device) # (64,) 
    anchors_position = torch.load(os.path.join("../find_lcs/lcs_params_6", "anchors_positions.pt")).to(latents.device) # (8,3) 
    anchors_hex = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#000000", "#FFFFFF"] 
    alphas = torch.load(os.path.join("../find_lcs/alphas_and_betas_6/", "alphas.pt")).to(latents.device) # (50, 3)
    betas = torch.load(os.path.join("../find_lcs/alphas_and_betas_6/", "betas.pt")).to(latents.device) # (50, 3)
    one_minus_gammas = [
        0, 
        0.0108, 0.0218, 0.0330, 0.0444, 0.0560, 0.0678, 0.0799, 0.0922, 0.1048, 0.1176, 
        0.1307, 0.1441, 0.1578, 0.1718, 0.1861, 0.2007, 0.2156, 0.2308, 0.2464, 0.2623, 
        0.2786, 0.2953, 0.3124, 0.3299, 0.3478, 0.3662, 0.3850, 0.4043, 0.4241, 0.4444, 
        0.4652, 0.4866, 0.5086, 0.5312, 0.5544, 0.5783, 0.6028, 0.6281, 0.6541, 0.6809, 
        0.7085, 0.7369, 0.7662, 0.7965, 0.8277, 0.8600, 0.8933, 0.9278, 0.9635, 1.000
    ]
    if gamma is None:
        gamma = 1 - one_minus_gammas[steer_timestep]

    if steer_timestep == 0:
        latents = steer(B, mu, anchors_position, anchors_hex, alphas, betas, gamma, 
                        latents, steer_timestep, target_hex,
                        mask_block_id, model, text_id, prompt)

    if save_visualizations:
        visualize(B, mu, anchors_position, anchors_hex, alphas, betas,
                  latents, 0, model, prompt, width, height, save_dir)
    #####

    # 6. Denoising loop
    # We set the index here to remove DtoH sync, helpful especially during compilation.
    # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
    model.scheduler.set_begin_index(0)
    with model.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if model.interrupt:
                continue

            model._current_timestep = t
            if image_embeds is not None:
                model._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            with model.transformer.cache_context("cond"):
                noise_pred = model.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=model.joint_attention_kwargs,
                    return_dict=False,
                )[0]

            if do_true_cfg:
                if negative_image_embeds is not None:
                    model._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

                with model.transformer.cache_context("uncond"):
                    neg_noise_pred = model.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=model.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(model, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % model.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

            ######################
            if steer_timestep == i + 1:
                latents = steer(B, mu, anchors_position, anchors_hex, alphas, betas, gamma, 
                                latents, steer_timestep, target_hex,
                                mask_block_id, model, text_id, prompt)

            if save_visualizations and i + 1 in {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}:
                visualize(B, mu, anchors_position, anchors_hex, alphas, betas,
                          latents, i + 1, model, prompt, width, height, save_dir)
            ######################

    model._current_timestep = None

    if output_type == "latent":
        image = latents
    else:
        latents = model._unpack_latents(latents, height, width, model.vae_scale_factor)
        latents = (latents / model.vae.config.scaling_factor) + model.vae.config.shift_factor
        image = model.vae.decode(latents, return_dict=False)[0]
        image = model.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    model.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)
