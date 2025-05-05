import numpy as np
import matplotlib.pyplot as plt
from helpers import stack_block_diag, stack_vecs, vec, project_to_SO2, plot_pose, rot_mat, random_adjacency_matrix, J
import cvxpy as cp
import imageio
import random

"""
    This file is DEPRACATED. See simple_ts.py for the actual working (centralized) code for the SLAM problem.
"""

N = 30
w_R = 100
tr_var = .01
w_t = 1 / tr_var
# w_R = 2 * np.pi
N_iters = 10
gamma = 1

def update_y(i, ys, Adj, relative_rotations, relative_translations, pred_rots, gamma_cur, flagged):

    Num_neighbors = np.sum(Adj[i])
    H_ii = np.eye(3)

    # theta-theta term
    H_ii[0,0] = np.sum(
        [w_t**2 * np.linalg.norm(relative_translations[(i,j)])**2 + 2 * (w_t**2 + w_R**2) for j in range(N + 1) if j != i and Adj[i,j] == 1]) 
    # translation-theta term

    H_ii[1:,0] = - w_R**2 * sum(
        [pred_rots[i]@J@relative_translations[(i,j)] for j in range(N + 1) if j != i and Adj[i,j] == 1]
    )[:,-1]
    # theta-translation term
    H_ii[0,1:] = H_ii[1:,0].T
    # Translation-translation term
    H_ii[1:,1:] = Num_neighbors * (w_R**2 + w_t**2) * np.eye(2)

    H_ij_y_sum = np.zeros((3, 1))
    for j in range(N + 1):
        if j != i and Adj[i,j] == 1:
            H_ij = np.zeros((3, 3))
            
            H_ij[1:,1:] = np.eye(2) * (w_R**2 + w_t**2)
            H_ij[1:,0] = -w_t**2 * (pred_rots[j] @J@relative_translations[(j,i)])[:,-1]
            H_ij[0,1:] = -w_t**2 * relative_translations[(i,j)].T@J@pred_rots[i]
            H_ij[0,0] = w_R**2 * vec(J).T@(np.kron(relative_rotations[(i,j)] + relative_rotations[(j,i)].T, pred_rots[i].T@pred_rots[j]))@vec(J)
            H_ij_y_sum += H_ij@ys[j] if flagged[j] == 1 else 0

    g_i = np.zeros((3, 1))
    g_i[1:] = w_t**2 * sum(
        [pred_rots[i]@relative_translations[(i,j)] + pred_rots[j]@relative_translations[(j,i)] for j in range(N + 1) if j != i and Adj[i,j] == 1]
    )
    g_i[0] = sum([
        w_t**2 * np.linalg.norm(relative_translations[(i,j)])**2 + 2 * w_R**2 * vec(J).T@np.kron(relative_rotations[(i,j)], pred_rots[i].T)@vec(pred_rots[j]) + 2*w_R**2 for j in range(N + 1) if j != i and Adj[i,j] == 1
    ])

    # for i in range(N+1):
    #     ys[i][0] = 0

    y_new = gamma_cur * ys[i] + (1 - gamma_cur) * np.linalg.inv(H_ii)@(-1 * H_ij_y_sum + 2 * g_i)

    flagged[i] = 1

    return y_new

            



np.random.seed(42)
random.seed(42)

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
fig, ax = plt.subplots(figsize=(8, 4))
for i in range(N + 1):
    plot_pose((trs[i], rot_arrs[i]), ax, color=colors[i])

plt.xlim(-1, N + 1)
plt.ylim(-1, N + 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.title('Recorded Translations')
# Save the plot
plt.savefig('ground_truth_translations.pdf')

plt.close(fig)

print("Ground truth plot generated.")
print("Starting initial rotation optimization...")

# Initialize
y = [vec(np.random.random((2,2))) for _ in range(N + 1)]
frames = []

for it in range(N_iters):
    # Update estimates
    for agent in range(N + 1):
        if agent == 0:
            y[agent] = vec(rot_arrs[agent])
        else:
            Num_neighbors = np.sum(Adj[agent])
            y[agent] = (1 - gamma) * y[agent]  + gamma * (1/(2 * Num_neighbors)) * sum(
                [np.kron(relative_rotations[(agent, j)] + relative_rotations[(j, agent)].T, np.eye(2)) @ y[j]
                for j in range(N + 1) if j != agent and Adj[agent, j] == 1]
            )
    # Reshape & project
    y_vecs = [y_vec.reshape(2, 2, order='F') for y_vec in y]
    Rs = [project_to_SO2(y_vec) for y_vec in y_vecs]

    # change = rot_arrs[0].T@Rs[0]
    change = np.eye(2)
    Rs_est = [R@change.T for R in Rs]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))
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

print("Initial rotation optimization complete.")
print("Starting translation optimization...")

import numpy as np
import matplotlib.pyplot as plt
import imageio

# … assume rots, trs, Adj, relative_rotations, relative_translations,
#    Rs_est, N, colors, plot_pose, rot_mat, rot_arrs are all defined …

N_iters_T = 100
gamma_T  = 1

# y[i] is [theta, t_i], where t_i is a 2D translation
ys = [np.random.random((3,1)) for _ in range(N + 1)]
flagged = [0 for _ in range(N + 1)]

# Make the translations between 0 and N
for i in range(N + 1):
    ys[i][1:] = np.random.random((2,1)) * N
# grounding_vec = np.array([rots[0], trs[0][0][0], trs[0][1][0]]).reshape(3,1)

frames = []

for it in range(N_iters_T):
    print(f"Iteration {it+1}/{N_iters_T}")
    # --- step the solver ---
    for agent in range(N + 1):
        # if agent == 0:
        #     ys[agent] = grounding_vec
        ys[agent] = update_y(agent,
                        ys,
                        Adj,
                        relative_rotations,
                        relative_translations,
                        Rs_est,
                        gamma_T, flagged)

    # extract poses
    thetas       = [float(y[0][0]) for y in ys]
    print(thetas)
    translations = [y[1:]              for y in ys]
    Rs_final     = [Rs_est[i] @ rot_mat(thetas[i]).T
                    for i in range(N + 1)]

    # --- draw the current frame ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for ax, title in zip((ax1, ax2), ('Ground Truth', f'Estimate it {it+1}')):
        ax.set_xlim(-1, N + 1)
        ax.set_ylim(-1, N + 1)
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        ax.set_title(title)

    # ground truth (static)
    for i in range(N + 1):
        plot_pose((trs[i],     rot_arrs[i]), ax1, color=colors[i])

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
gif_path = 'training_progress.gif'
imageio.mimsave(gif_path, frames, fps=10)
print(f"Translation optimization complete. GIF saved to {gif_path}")
