import numpy as np
import matplotlib.pyplot as plt
from helpers import stack_block_diag, stack_vecs, vec, project_to_SO2, plot_pose, rot_mat, random_adjacency_matrix, rotate_pointcloud, group_points_by_cluster
from pc_plot import plot_gt_rotations, plot_gt_poses_with_pc, get_rotation_frame_with_pc, get_translation_frame_with_pc, save_rotation_frame, save_translation_frame, get_translation_rotation_frame_with_pc
import imageio
import random
import yaml
from robot import SLAMRobot

loaded = np.load('point_cloud_with_clusters_many.npz')
points = loaded['points']
labels = loaded['labels']
centers = loaded['cluster_centers']

points_arr = group_points_by_cluster(points, labels)

"""
    This is the actual working (centralized) code for the SLAM problem
"""

index = 1
centers[index] = centers[index] + np.array([.5,.5])

# True for reproducibility
set_seed = True
plotting = True
if set_seed:
    # np.random.seed(1123)
    # random.seed(1123)
    np.random.seed(2620)
    random.seed(2620)

rel_points = [points_arr[i] - centers[i] for i in range(len(centers))]

# Load config dict from yaml
config_path = 'robot_config.yaml'
config = yaml.safe_load(open(config_path, 'r'))

# Number of optimization steps for rotations and translations
N_iters_R = config['Optimization']['Rotation']['N_iters']
N_iters_T = config['Optimization']['Translation']['N_iters']

# Number of agnets
N = len(centers) - 1
# 'Concentration' parameter for von Mises distribution. Higher values mean less noise
w_R = 50
# Variance for (gaussian) translation noise
tr_var = .2

print("Generating random rotations and translations...")

# Ground truth poses
gt_rots = [np.random.random() * 2 * np.pi for _ in range(N + 1)]
# gt_trs are cluster centers
gt_trs = [centers[i].reshape(2,1) for i in range(N + 1)]
# Add identity rotation as a frame of reference (we'll remove this later)
gt_rot_mats = [rot_mat(rot) for rot in gt_rots]

rel_points = [rotate_pointcloud(rel_points[i], rot_mat(gt_rots[i])) for i in range(N + 1)]

# # Plot the relative points
# if plotting:
#     fig, ax = plt.subplots(figsize=(8, 4))
#     for i in range(N):
#         ax.scatter(rel_points[i][:, 0], rel_points[i][:, 1], label=f'Cluster {i}')
#     ax.set_title('Relative Points')
#     ax.legend()
#     plt.show()

# Rotate points by gt_rots, and center them around gt_trs
for i in range(N + 1):
    points[labels == i, 0] = np.cos(gt_rots[i]) * points[labels == i, 0] - np.sin(gt_rots[i]) * points[labels == i, 1]
    points[labels == i, 1] = np.sin(gt_rots[i]) * points[labels == i, 0] + np.cos(gt_rots[i]) * points[labels == i, 1]
    points[labels == i, :] += gt_trs[i].reshape(1,2)

# Create adjacency matrix
Adj = random_adjacency_matrix(N, p=.01)

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
    plot_gt_poses_with_pc(gt_trs, gt_rot_mats, relative_rotations, relative_translations, N, Adj, rel_points)
    print("Ground truth plot generated.")
    frames = []

print("Starting initial rotation optimization...")

# Make robots
robots = [SLAMRobot(config, id=i) for i in range(N + 1)]

# Robots transmit 'sensor data'
# every robot needs to know its neighbors sensed (relative) rotations and translations
# every robot needs to know its neighbor's estimated rotation and translation of itself
for i in range(N + 1):
    neighbors = [j for j in range(N + 1) if Adj[i, j] == 1]
    relative_rots = {k: v for k, v in relative_rotations.items() if k[0] == i or k[1] == i}
    relative_translations_now = {k: v for k, v in relative_translations.items() if k[0] == i or k[1] == i}
    robots[i].init_rendezvous(neighbors, relative_rots, relative_translations_now)

    # Send initial estimates
    for j in neighbors:
        robots[i].update_neighbor_estimated_rotations([j], [robots[j].estimated_rotation])
        robots[i].update_neighbor_estimated_translations([j], [robots[j].estimated_translation])
    
    print(robots[i].estimated_translation)

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
        trs_est = [robots[i].estimated_translation for i in range(N + 1)]

        img = get_translation_rotation_frame_with_pc(trs_est, Rs_est, gt_trs, gt_rot_mats, it, N,rel_points)

        # img = get_rotation_frame_with_pc(Rs_est, gt_rot_mats, it, N,rel_points)
        frames.append(img)

        if it == N_iters_R - 1:
            # Save the last img to a pdf
            save_rotation_frame(Rs_est, gt_rot_mats, it, N)

if plotting:
    gif_path = 'rotation_opt.gif'
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
        img = get_translation_frame_with_pc(ys, Rs_est, gt_trs, gt_rot_mats, it, N,rel_points)
        frames.append(img)

        if it == N_iters_T - 1:
            # Save the last img to a pdf
            save_translation_frame(ys, Rs_final, gt_trs, gt_rot_mats, it, N)


# Save GIF
if plotting:
    gif_path = 'translation_opt.gif'
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Translation optimization GIF saved to {gif_path}.")

print(f"Translation optimization complete.")
