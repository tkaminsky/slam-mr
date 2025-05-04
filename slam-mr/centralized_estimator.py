import numpy as np
import matplotlib.pyplot as plt
from helpers import stack_block_diag, stack_vecs, vec, project_to_SO2, plot_pose, rot_mat, random_adjacency_matrix
from plot import plot_gt_rotations, plot_gt_poses, get_rotation_frame, get_translation_frame
import imageio
import random
import yaml
from robot import SLAMRobot

"""
    This is the actual working (centralized) code for the SLAM problem
"""

# True for reproducibility
set_seed = False
plotting = True
if set_seed:
    np.random.seed(1123)
    random.seed(1123)

# Number of agnets
N = 5
# 'Concentration' parameter for von Mises distribution. Higher values mean less noise
w_R = 1000
# Variance for (gaussian) translation noise
tr_var = .05

print("Generating random rotations and translations...")

# Ground truth poses
gt_rots = [np.random.random() * 2 * np.pi for _ in range(N)]
gt_trs = [np.random.random((2,1)) * N for _ in range(N)]
# Add identity rotation as a frame of reference (we'll remove this later)
gt_rots = [0] + gt_rots
gt_rot_mats = [rot_mat(rot) for rot in gt_rots]
gt_trs = [np.zeros((2,1))] + gt_trs
# Create adjacency matrix
Adj = random_adjacency_matrix(N, p=.1) 

print("Adjacency matrix:")
print(Adj)

# Create noisy relative rotations and translations
relative_rotations = {}
relative_translations = {}
for i in range(N + 1):
    for j in range(N + 1):
        if i != j:
            noise = np.random.vonmises(0,w_R)
            R_ij = rot_mat(gt_rots[i]).T@rot_mat(gt_rots[j])@rot_mat(noise)
            relative_rotations[(i, j)] = R_ij
            t_ij = rot_mat(gt_rots[i]).T @ (gt_trs[j] - gt_trs[i]) + np.random.normal(0, tr_var, (2,1))
            relative_translations[(i, j)] = t_ij
        if i == j:
            relative_rotations[(i,j)] = np.eye(2)
            relative_translations[(i,j)] = np.zeros((2,1))

print("Poses generated.")

if plotting:
    print("Generating ground truth plot...")
    plot_gt_rotations(gt_rots, relative_rotations, N)
    plot_gt_poses(gt_trs, gt_rot_mats, relative_rotations, relative_translations, N)
    print("Ground truth plot generated.")
    frames = []

print("Starting initial rotation optimization...")

# Load config dict from yaml
config_path = 'robot_config.yaml'
config = yaml.safe_load(open(config_path, 'r'))

# Number of optimization steps for rotations and translations
N_iters_R = config['Optimization']['Rotation']['N_iters']
N_iters_T = config['Optimization']['Translation']['N_iters']

# Make robots
robots = [SLAMRobot(config, id=i) for i in range(N + 1)]

# Robots transmit 'sensor data'
for i in range(N + 1):
    neighbors = [j for j in range(N + 1) if Adj[i, j] == 1]
    relative_rots = {k: v for k, v in relative_rotations.items() if k[0] == i or k[1] == i}
    relative_translations_now = {k: v for k, v in relative_translations.items() if k[0] == i or k[1] == i}
    robots[i].init_rendezvous(neighbors, relative_rots, relative_translations_now)

    # Send initial estimates
    for j in neighbors:
        robots[i].update_neighbor_estimated_rotations([j], [robots[j].estimated_rotation])
        robots[i].update_neighbor_estimated_translations([j], [robots[j].estimated_translation])

# Rotation estimation
print(f"Iteration {0}/{N_iters_R}")
for it in range(N_iters_R):
    if it % 10 == 9:
        print(f"Iteration {it+1}/{N_iters_R}")
    # Update estimates
    for agent in range(N + 1):
        # Perform an update step
        robots[agent].update_estimated_rotation()
        # Send updated estimate to neighbors
        for neighbor in robots[agent].neighbors:
            robots[neighbor].update_neighbor_estimated_rotations([agent], [robots[agent].estimated_rotation])

    if plotting:
        y = [robots[i].estimated_rotation for i in range(N + 1)]
        y_vecs = [y_vec.reshape(2, 2, order='F') for y_vec in y]
        Rs_est = [project_to_SO2(y_vec) for y_vec in y_vecs]
        img = get_rotation_frame(Rs_est, gt_rot_mats, it, N)
        frames.append(img)

if plotting:
    gif_path = 'rotations.gif'
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Saved rotation optimization GIF to {gif_path}\n")

print("Initial rotation optimization complete.")
print("Starting translation optimization...")

# … assume rots, trs, Adj, relative_rotations, relative_translations,
#    Rs_est, N, colors, plot_pose, rot_mat, rot_arrs are all defined …

# frames = []

print(f"Iteration {0}/{N_iters_T}")
for it in range(N_iters_T):
    if it % 10 == 9:
        print(f"Iteration {it + 1}/{N_iters_T}")
    # --- step the solver ---
    for agent in range(N + 1):
        robots[agent].update_estimated_translation()
        # Send updated estimate to neighbors
        for neighbor in robots[agent].neighbors:
            robots[neighbor].update_neighbor_estimated_translations([agent], [robots[agent].estimated_translation])

    if plotting:
        ys = [robots[i].estimated_translation for i in range(N + 1)]
        Rs_final = Rs_est
        img = get_translation_frame(ys, Rs_est, gt_trs, gt_rot_mats, it, N)
        frames.append(img)

# Save GIF
if plotting:
    gif_path = 'optimization.gif'
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Translation optimization GIF saved to {gif_path}.")

print(f"Translation optimization complete.")
