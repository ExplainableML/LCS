import torch
import os
import numpy as np
from sklearn.decomposition import PCA

# -------------------- Config --------------------
root_dir = "hsv_latents_6"          # directory with per-color .pt files
save_dir = "lcs_params_6"           # where to save PCA tensors
os.makedirs(save_dir, exist_ok=True)

# -------------------- Load embeddings --------------------
embeddings = []
color_names = []

for file_name in sorted(os.listdir(root_dir)):
    if file_name.endswith(".pt"):
        hex_code = file_name.replace(".pt", "")
        tensor = torch.load(os.path.join(root_dir, file_name), map_location="cpu")
        tensor = tensor.squeeze(0)
        mean_embedding = tensor.mean(0).float()
        embeddings.append(mean_embedding.numpy())
        color_names.append(hex_code)

embeddings = np.stack(embeddings)  # shape: (num_colors, 64)
emb_tensor = torch.from_numpy(embeddings)

# -------------------- Step 1: Compute mean embedding --------------------
mu = emb_tensor.mean(dim=0)
emb_centered = emb_tensor - mu

# -------------------- Step 2: Fit PCA --------------------
pca = PCA(n_components=3)
pca.fit(emb_centered.numpy())

B = torch.from_numpy(pca.components_).float()          # (3,64)
eigvals = torch.from_numpy(pca.explained_variance_).float()  # (3,)
explained_variance_ratio = eigvals / eigvals.sum()
total_variance_explained = explained_variance_ratio.sum()
print(total_variance_explained)

# -------------------- Step 3: Compute per-color vectors in PCA space --------------------
c_vectors = emb_centered @ B.T   # (num_colors,3)

# -------------------- Step 4: Save PCA results --------------------
torch.save(B, os.path.join(save_dir, "B.pt"))
torch.save(mu, os.path.join(save_dir, "mu.pt"))
torch.save(c_vectors, os.path.join(save_dir, "color_vectors.pt"))

# -------------------- Step 5: Compute PCA positions for 6 hues + black + white --------------------
# Define hex codes for canonical colors
canonical_hex = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#000000", "#FFFFFF"]
canonical_vectors = []

for hex_code in canonical_hex:
    file_path = os.path.join(root_dir, f"{hex_code}.pt")
    if os.path.exists(file_path):
        tensor = torch.load(file_path, map_location="cpu").squeeze(0)
        mean_emb = tensor.mean(0).float()
    else:
        # if not found, just use zeros
        mean_emb = torch.zeros(64)
    canonical_vectors.append(mean_emb)

canonical_vectors = torch.stack(canonical_vectors)
canonical_centered = canonical_vectors - mu
canonical_pca_positions = canonical_centered @ B.T  # shape: (8,3)

torch.save(canonical_pca_positions, os.path.join(save_dir, "anchors_positions.pt"))

print("Saved LCS params:")
print(" - B:", B.shape)
print(" - Mu:", mu.shape)
print(" - Anchors positions:", canonical_pca_positions.shape)
print(" - Color vectors:", c_vectors.shape)

with open(os.path.join(save_dir, "colors.txt"), "w") as file:
    for color in color_names:
        file.write(color + "\n")