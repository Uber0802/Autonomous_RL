import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cuml.manifold import UMAP

# ------------------------------------
# 1. Load or simulate your embeddings
# ------------------------------------
# Example shape: (steps, tasks, samples, dim)
# embeddings = np.load("your_embeddings.npy")
# Example placeholder for testing:
# embeddings = np.random.randn(80, 64, 256, 4096).astype(np.float32)
path = "/workspace/Autonomous_RL/SimplerEnv/wandb/offline-run-20251030_081749-6zbftt1w/glob/0/bottle_shovel-320-train_twice-joint-reset-seed_2_embed.npy"
embeddings = np.load(path)
steps, tasks, samples, dim = embeddings.shape
print(f"Embeddings shape: {embeddings.shape}")

# ------------------------------------
# 2. Reshape for UMAP input
# ------------------------------------
embeddings_flat = embeddings.reshape(-1, dim)  # (80*64*256, 4096)
print(f"Flattened shape: {embeddings_flat.shape}")

# Move data to GPU
embeddings_gpu = cp.asarray(embeddings_flat)

# ------------------------------------
# 3. Run GPU-accelerated UMAP
# ------------------------------------
print("Running GPU UMAP... this might take a few minutes.")
reducer = UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='cosine',
    random_state=42,
    verbose=True
)

embeddings_umap_gpu = reducer.fit_transform(embeddings_gpu)
embeddings_umap = cp.asnumpy(embeddings_umap_gpu)  # back to CPU for plotting

print("UMAP done:", embeddings_umap.shape)

# ------------------------------------
# 4. Build indices for tasks & steps
# ------------------------------------
step_idx = np.repeat(np.arange(steps), tasks * samples)
task_idx = np.tile(np.repeat(np.arange(tasks), samples), steps)

# ------------------------------------
# 5. Plot results
# ------------------------------------
plt.figure(figsize=(10, 8))
tasks = 4
cmap = plt.get_cmap("tab20", tasks)

for t in range(tasks):
    task_mask = task_idx == t
    color = cmap(t)

    for s in range(steps):
        step_mask = (step_idx == s) & task_mask
        alpha = (s + 1) / steps  # step progression from 1/80 → 1.0

        plt.scatter(
            embeddings_umap[step_mask, 0],
            embeddings_umap[step_mask, 1],
            color=color,
            alpha=alpha,
            s=5,
            label=f"Task {t}" if s == 0 else None
        )

plt.title("UMAP Projection of 320_32ep Embeddings")
#plt.xlabel("UMAP-1")
#plt.ylabel("UMAP-2")
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()

# Save the figure to file
plt.savefig("320_32ep.png", dpi=300)
plt.close()  # closes the figure without showing it