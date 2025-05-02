import numpy as np
import matplotlib.pyplot as plt
from helpers import stack_block_diag, stack_vecs, vec, project_to_SO2, plot_pose, rot_mat
import cvxpy as cp
import imageio
import random

N = 4
noise_level = np.pi / 4
# noise_level = 2 * np.pi
N_iters = 100
gamma = 1

np.random.seed(42)
random.seed(42)

rots = [np.random.random() * 2 * np.pi for _ in range(N)]
# Add identity rotation as a frame of reference
rots = [0] + rots

# Create adjacency matrix
Adj = np.zeros((N + 1, N + 1), dtype=int)


# # While it isn't connected, add edges
# while np.linalg.matrix_rank(Adj) < N + 1:
#     for i in range(N + 1):
#         for j in range(i + 1, N + 1):
#             # Randomly decide if there is an edge
#             if np.random.random() < 0.5:
#                 Adj[i, j] = 1
#                 Adj[j, i] = 1
# # Make adj a line
# for i in range(N + 1):
#     for j in range(i + 1, N + 1):
#         if i == j - 1:
#             Adj[i, j] = 1
#             Adj[j, i] = 1

# Adj[0,-1] = 1
# Adj[-1,0] = 1

Adj = np.ones((N + 1, N + 1), dtype=int) - np.eye(N + 1, dtype=int)

print(Adj)

rot_arrs = [rot_mat(rot) for rot in rots]

# Noisy relative rotations
relative_rotations = {}
for i in range(N + 1):
    for j in range(N + 1):
        if i != j:
            noise = (np.random.random() - 0.5) * noise_level
            R_ij = rot_mat(rots[j] + noise) @ rot_mat(rots[i]).T
            relative_rotations[(i, j)] = R_ij
        if i == j:
            relative_rotations[(i,j)] = np.eye(2)

# Make n distinct colors
colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
# Create a figure
fig, ax = plt.subplots(figsize=(8, 4))

# Create a plot of the percieved rotations for each agent
for i in range(N + 1):
    for j in range(N + 1):
        t = np.array([j,i])
        rot = relative_rotations[(i,j)] @ rot_mat(rots[i])
        plot_pose((t, rot), ax, color=colors[j])
        if i == j:
            # Draw a red square around the diagonal
            square = plt.Rectangle((i-0.25, j-0.25), 0.5, 0.5, color='r', fill=False)
            ax.add_patch(square)
plt.xlim(-1, N + 1)
plt.ylim(-1, N + 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.title('Recorded Rotations')
# Save the plot
plt.savefig('ground_truth_rotations.pdf')
plt.close(fig)

# Initialize
y = [vec(np.random.random((2,2))) for _ in range(N + 1)]
frames = []

for it in range(N_iters):
    # Update estimates
    for agent in range(N + 1):
        Num_neighbors = np.sum(Adj[agent])
        y[agent] = (1 - gamma) * y[agent]  + gamma * (1/(2 * Num_neighbors)) * sum(
            [np.kron(relative_rotations[(agent, j)] + relative_rotations[(j, agent)].T, np.eye(2)) @ y[j]
             for j in range(N + 1) if j != agent and Adj[agent, j] == 1]
        )
    # Reshape & project
    y_vecs = [y_vec.reshape(2, 2, order='F') for y_vec in y]
    Rs = [project_to_SO2(y_vec) for y_vec in y_vecs]

    change = rot_mat(rots[0]) @ Rs[0].T
    Rs_est = [change @ R for R in Rs]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    titles = ['Ground Truth', f'Iteration {it+1}']
    for ax, title in zip(axes, titles):
        ax.set_xlim(-1, N + 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        ax.set_title(title)
    # Ground truth (static)
    for idx, color in zip(range(N + 1), colors):
        plot_pose((np.array([idx, 0]), rot_arrs[idx]), axes[0], color=color)
        plot_pose((np.array([idx, 0]), Rs_est[idx]), axes[1], color=color)

    # Render and capture (robust across backends)
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    img = img[:, :, :3]           # keep RGB, drop alpha
    frames.append(img)
    plt.close(fig)

# Save GIF
gif_path = 'rotations.gif'
imageio.mimsave(gif_path, frames, fps=10)
print(f"Saved GIF to {gif_path}")
