import numpy as np
import matplotlib.pyplot as plt
from helpers import stack_block_diag, stack_vecs, vec, project_to_SO2, plot_pose, rot_mat, random_adjacency_matrix, J
import cvxpy as cp
import imageio
import random

"""
    This is the actual working (centralized) code for the SLAM problem
"""

N = 50
w_R = 1000
tr_var = .1
w_t = 1 / tr_var
# w_R = 2 * np.pi
N_iters = 50
gamma = 1

# np.random.seed(42)
# random.seed(42)

print("Generating random rotations and translations...")

rots = [np.random.random() * 2 * np.pi for _ in range(N)]
# Sample random translations
trs = [np.random.random((2,1)) * N for _ in range(N)]
# Add identity rotation as a frame of reference
rots = [0] + rots
trs = [np.zeros((2,1))] + trs

# Create adjacency matrix
Adj = random_adjacency_matrix(N, p=.1) 

print(Adj)

rot_arrs = [rot_mat(rot) for rot in rots]

# Noisy relative rotations
relative_rotations = {}
relative_translations = {}
for i in range(N + 1):
    for j in range(N + 1):
        if i != j:
            noise = np.random.vonmises(0,w_R)
            R_ij = rot_mat(rots[i]).T@rot_mat(rots[j])@rot_mat(noise)
            relative_rotations[(i, j)] = R_ij
            t_ij = rot_mat(rots[i]).T @ (trs[j] - trs[i]) + np.random.normal(0, tr_var, (2,1))
            relative_translations[(i, j)] = t_ij
        if i == j:
            relative_rotations[(i,j)] = np.eye(2)
            relative_translations[(i,j)] = np.zeros((2,1))


print("Poses generated.")
print("Generating ground truth plot...")

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

# Do the same with translations/rotations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for i in range(N + 1):
    plot_pose((trs[i], rot_arrs[i]), ax1, color=colors[i])
    for j in range(N+1):
        pred_rot = rot_arrs[i]@relative_rotations[(i,j)]
        t = trs[i] + rot_arrs[i]@relative_translations[(i,j)]
        plot_pose((t, pred_rot), ax2, color=colors[j])

ax1.set_title('Ground Truth')
ax2.set_title('Sensed Poses')
ax1.set_xlim(-1, N + 1)
ax1.set_ylim(-1, N + 1)
ax1.set_aspect('equal', 'box')
ax1.grid(True)
ax2.set_xlim(-1, N + 1)
ax2.set_ylim(-1, N + 1)
ax2.set_aspect('equal', 'box')
ax2.grid(True)
fig.suptitle('Sensed Poses', fontsize=16)
# Save the plot
plt.savefig('sensed_poses.pdf')

plt.close(fig)



print("Ground truth plot generated.")
print("Starting initial rotation optimization...")

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

    change = rot_arrs[0]@Rs[0].T
    # change = np.eye(2)
    Rs_est = [change@R for R in Rs]

    # Plot
    # fig, axes = plt.subplots(2, 1, figsize=(8, 4))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    titles = ['Ground Truth', f'Estimating Rotation {it+1}']
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
    fig.suptitle("Optimizing Rotations", fontsize=16)
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

print("Initial rotation optimization complete.")
print("Starting translation optimization...")

import numpy as np
import matplotlib.pyplot as plt
import imageio

# … assume rots, trs, Adj, relative_rotations, relative_translations,
#    Rs_est, N, colors, plot_pose, rot_mat, rot_arrs are all defined …

N_iters_T = 50
gamma_T  = 1

# y[i] is [theta, t_i], where t_i is a 2D translation
ys = [N * np.random.random((2,1)) for _ in range(N + 1)]

print("Extrema:")
print(np.max(ys))
print(np.min(ys))

# frames = []

for it in range(N_iters_T):
    print(f"Iteration {it+1}/{N_iters_T}")
    # --- step the solver ---
    for agent in range(N + 1):
        Num_neighbors = np.sum(Adj[agent])
        g_agent = sum(
            [
                Rs_est[k] @ relative_translations[(k,agent)] - Rs_est[agent]@relative_translations[(agent,k)]
                for k in range(N + 1) if k != agent and Adj[agent, k] == 1
            ]
        )
        Hy_sum = sum(
            [
                -2 * ys[k] for k in range(N+1) if k != agent and Adj[agent, k] == 1
            ]
        )

        ys[agent] = (1 - gamma_T )* ys[agent] + \
            gamma_T * 1 / (2 * Num_neighbors) * ( - Hy_sum + g_agent)
        
        
        
    change = trs[0] - ys[0] 
    translations = change + ys
    Rs_final = Rs_est

    # --- draw the current frame ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for ax, title in zip((ax1, ax2), ('Ground Truth', f'Estimating Translation It: {it+1}')):
        ax.set_xlim(-1, N + 1)
        ax.set_ylim(-1, N + 1)
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        ax.set_title(title)
    fig.suptitle("Optimizing Translations", fontsize=16)

    # ground truth (static)
    for i in range(N + 1):
        plot_pose((trs[i], rot_arrs[i]), ax1, color=colors[i])

    # current estimate
    for i in range(N + 1):
        plot_pose((translations[i], Rs_final[i]), ax2, color=colors[i])

    # render and capture
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    img = img[:, :, :3]           # drop alpha channel
    frames.append(img)
    plt.close(fig)

# Save GIF
gif_path = 'optimization.gif'
imageio.mimsave(gif_path, frames, fps=10)
print(f"Translation optimization complete. GIF saved to {gif_path}")
