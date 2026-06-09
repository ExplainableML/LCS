import torch
import os

# -------------------- Config --------------------
root_dir = "latents_walls"
timesteps = range(0, 51)
ref_t = 50
save_dir = "alphas_and_betas_6"
os.makedirs(save_dir, exist_ok=True)

# -------------------- Load PCA model --------------------
B = torch.load(os.path.join("lcs_params_6", "B.pt"))      # (3,64)
mu = torch.load(os.path.join("lcs_params_6", "mu.pt"))      # (64,)

B = B.float()
mu = mu.float()

# -------------------- Load color names --------------------
color_names = sorted(os.listdir(root_dir))

# -------------------- Load embeddings --------------------
def load_embeddings(t):
    embs = []
    for color_dir in color_names:
        file_path = os.path.join(root_dir, color_dir, f"{t}.pt")
        if os.path.exists(file_path):
            tensor = torch.load(file_path, map_location="cpu").squeeze(0).float()
            mean_embedding = tensor.mean(0)
            embs.append(mean_embedding)
    return torch.stack(embs)   # (C,64)

# -------------------- Compute per-timestep PCA magnitudes (average over images) --------------------
z_all = []
pca_means = []

for t in timesteps:
    embs = load_embeddings(t)          # (C,64)

    # Center embeddings by global PCA mean
    x_c = embs - mu                    # (C,64)
    
    # Project each image to PCA subspace
    z = x_c @ B.T                       # (C,3)
    
    # Compute mean PCA coordinates across images
    z_mean = z.mean(dim=0)              # (3,)
    pca_means.append(z_mean)

    # Compute magnitude per image in PCA space
    # Then subtract mean per timestep if desired
    z_centered = z - z_mean             # (C,3)
    z_all.append(z_centered)

# shape: (T,C,3)
z_all = torch.stack(z_all)
pca_means = torch.stack(pca_means)

# -------------------- Magnitudes --------------------
# Magnitudes per image
axis_mags = z_all.abs().mean(dim=1)     # (T,3) average magnitude per axis

# -------------------- Normalize to reference timestep --------------------
axis_mags_ref = axis_mags[ref_t]

axis_relative = axis_mags / axis_mags_ref.unsqueeze(0)

# -------------------- Save --------------------
torch.save(axis_relative, os.path.join(save_dir, "betas.pt"))   # (T,3)
torch.save(pca_means, os.path.join(save_dir, "alphas.pt"))    # (T,3)

print("Saved:")
print(" - Axis relative scales:", axis_relative.shape)
print(axis_relative)

print(" - PCA means per timestep:", pca_means.shape)
print(pca_means)


