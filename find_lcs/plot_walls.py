import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

root_dir = "latents_walls"  # Each subdir = a color
timesteps = list(range(50, -1, -1))  # 50,49,...,0

# -------------------- Load PCA model --------------------
B = torch.load(os.path.join("lcs_params_6", "B.pt"))      # (3,64)
mu = torch.load(os.path.join("lcs_params_6", "mu.pt"))      # (64,)

B = B.numpy()
mu = mu.numpy()

# -------------------- Style --------------------
sns.set_style("white")
sns.set_context("talk")

# -------------------- Gather color names and patch colors --------------------
color_names = []
color_rgb = []

for subdir in sorted(os.listdir(root_dir)):
    subpath = os.path.join(root_dir, subdir)
    if os.path.isdir(subpath):
        color_names.append(subdir)

        # Load img.png and compute mean color
        img_path = os.path.join(subpath, "img.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img) / 255.0  # normalize
            mean_rgb = img_np.mean(axis=(0,1))  # average over H,W
            color_rgb.append(tuple(mean_rgb))
        else:
            color_rgb.append((0,0,0))  # fallback black

color_rgb = np.array(color_rgb)  # shape (num_colors, 3)

# -------------------- Load embeddings function --------------------
def load_embeddings(t):
    embs = []
    for subdir in color_names:
        file_path = os.path.join(root_dir, subdir, f"{t}.pt")
        if os.path.exists(file_path):
            tensor = torch.load(file_path, map_location=torch.device("cpu"))
            tensor = tensor.squeeze(0)
            mean_embedding = tensor.mean(0).to(torch.float32)
            embs.append(mean_embedding.cpu().numpy())
        else:
            embs.append(np.zeros(mu.shape[0]))  # fallback zero vector
    return np.stack(embs)

# -------------------- Canonical points --------------------
canonical_positions = torch.load(os.path.join("lcs_params_6", "anchors_positions.pt"))    # (3,8)
canonical_hex = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#000000", "#FFFFFF"] 

# -------------------- Separate plots per timestep --------------------
out_dir = "plot_trajectory"
os.makedirs(out_dir, exist_ok=True)

for t in timesteps:
    emb = load_embeddings(t)

    # Project using B and mu
    z = (emb - mu) @ B.T  # shape (num_colors, n_components)

    # Project canonical points
    canon_proj = canonical_positions.numpy()

    plt.figure(figsize=(3,3))

    # Color points
    for i, name in enumerate(color_names):
        plt.scatter(z[i,1], z[i,2],
                    color=color_rgb[i],
                    s=80)

    # Canonical points as stars
    for i in range(canon_proj.shape[0] - 2):
        plt.scatter(canon_proj[i,1], canon_proj[i,2],
                    s=120,
                    c=[canonical_hex[i]],
                    edgecolors="black",
                    linewidths=0.5,
                    marker="*",
                    zorder=5)
        plt.text(canon_proj[i,1], canon_proj[i,2], f"{i}",
                 fontsize=9, ha="center", va="center",
                 color="white", weight="bold")

    plt.xlabel("PC2")
    plt.ylabel("PC3")

    sns.despine(trim=False, offset=5)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"color_embeddings_t{t}.png"),
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
