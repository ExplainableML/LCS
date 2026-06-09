import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# -------------------- Style --------------------
sns.set_style("white")   # clean background
sns.set_context("talk")  # slightly larger fonts

# -------------------- Config --------------------
root_dir = "hsv_latents_6"
save_dir = "lcs_params_6"
plot_dir = "plot_lcs_6"
os.makedirs(plot_dir, exist_ok=True)

# -------------------- Load PCA results --------------------
c_vectors = torch.load(os.path.join(save_dir, "color_vectors.pt"))  # (num_colors,3)
mu = torch.load(os.path.join(save_dir, "mu.pt"))
B = torch.load(os.path.join(save_dir, "B.pt"))

# -------------------- Load color names --------------------
with open(os.path.join(save_dir, "colors.txt")) as f:
    color_names = [line.strip() for line in f]

hex_to_idx = {hexcode: i for i, hexcode in enumerate(color_names)}

# -------------------- Convert all colors to RGB --------------------
colors_rgb = np.array([
    [int(h[1:3],16)/255, int(h[3:5],16)/255, int(h[5:7],16)/255]
    for h in color_names
])

# -------------------- 3D Scatter Plot --------------------
fig = plt.figure(figsize=(3.3,3))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    c_vectors[:,0],
    c_vectors[:,1],
    c_vectors[:,2],
    c=colors_rgb,
    s=50,
    depthshade=False
)

# Remove panes
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_facecolor((1,1,1,0))
    axis.pane.set_edgecolor((1,1,1,0))

# Remove axis lines
ax.xaxis.line.set_lw(0.)
ax.yaxis.line.set_lw(0.)
ax.zaxis.line.set_lw(0.)

# Remove ticks and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("PC1", fontdict={'fontsize': 22, 'fontweight': 'normal', 'family': 'sans-serif'}) 
ax.set_ylabel("PC2", fontdict={'fontsize': 22, 'fontweight': 'normal', 'family': 'sans-serif'})
ax.set_zlabel("PC3", fontdict={'fontsize': 22, 'fontweight': 'normal', 'family': 'sans-serif'})

# # Remove grid
# ax.grid(False)

# Equal aspect
ax.set_box_aspect([1,1,1])

sns.despine(trim=False, offset=10)  # removes top/right spines, slight offset
plt.grid(False)                     # remove grid lines

# plt.tight_layout()
fig.subplots_adjust(left=0.0, right=0.8, bottom=0.1, top=1)

plt.savefig(os.path.join(plot_dir, "pca_3d_scatter.png"), dpi=300)
plt.close(fig)


# black background

# -------------------- 3D Scatter Plot --------------------
fig = plt.figure(figsize=(3.3,3), facecolor='black')   # figure background
ax = fig.add_subplot(111, projection='3d', facecolor='black')  # axes background

ax.scatter(
    c_vectors[:,0],
    c_vectors[:,1],
    c_vectors[:,2],
    c=colors_rgb,
    s=50,
    depthshade=False
)

# Make panes transparent (so only edges show if enabled)
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_facecolor((0,0,0,0))
    axis.pane.set_edgecolor('white')   # optional white pane edges

# Axis line colors
ax.xaxis.line.set_color("white")
ax.yaxis.line.set_color("white")
ax.zaxis.line.set_color("white")

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Axis labels (white so visible on black)
ax.set_xlabel("PC1", color="white", fontsize=15)
ax.set_ylabel("PC2", color="white", fontsize=15)
ax.set_zlabel("PC3", color="white", fontsize=15)

ax.set_box_aspect([1,1,1])

sns.despine(trim=False, offset=10)  # removes top/right spines, slight offset
plt.grid(False)                     # remove grid lines

# plt.tight_layout()
fig.subplots_adjust(left=0.0, right=0.8, bottom=0.1, top=1)

plt.savefig(os.path.join(plot_dir, "black_pca_3d_scatter.png"), dpi=300)
plt.close(fig)


# -------------------- 2D Scatter Plots --------------------
pairs = [(0,1),(0,2),(1,2)]
labels = ["PC1","PC2","PC3"]

for i,j in pairs:
    plt.figure(figsize=(3,3))
    plt.scatter(c_vectors[:,i], c_vectors[:,j], c=colors_rgb, s=50)
    plt.xlabel(labels[i])
    plt.ylabel(labels[j])

    # if i == 0:
    #     plt.xticks([-20, 0, 20])
    # else:
    #     plt.xticks([-10, 0, 10])

    # if j == 0:
    #     plt.yticks([-20, 0, 20])
    # else:
    #     plt.yticks([-10, 0, 10])

    plt.xticks([])
    plt.yticks([])

    # Minimal, paper-ready style
    sns.despine(trim=False, offset=10)  # removes top/right spines, slight offset
    plt.grid(False)                     # remove grid lines

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{labels[i]}_vs_{labels[j]}.png"), dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    print(f"Saved 2D PCA plot: {labels[i]} vs {labels[j]}")

print("All PCA plots saved to", plot_dir)
