import numpy as np
import matplotlib.pyplot as plt
from helpers import stack_block_diag, stack_vecs, vec, project_to_SO2, plot_pose, rot_mat, random_adjacency_matrix

def plot_gt_rotations(gt_rots, relative_rotations, N):
    # Make n distinct colors
    colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create a plot of the percieved rotations for each agent
    for i in range(N + 1):
        for j in range(N + 1):
            t = np.array([j,i])
            rot = relative_rotations[(i,j)] @ rot_mat(gt_rots[i])
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

def plot_gt_poses(gt_trs, gt_rot_mats, relative_rotations, relative_translations, N):
    colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(N + 1):
        plot_pose((gt_trs[i], gt_rot_mats[i]), ax1, color=colors[i])
        for j in range(N+1):
            pred_rot = gt_rot_mats[i]@relative_rotations[(i,j)]
            t = gt_trs[i] + gt_rot_mats[i]@relative_translations[(i,j)]
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

def get_rotation_frame(Rs_est, gt_rot_mats, it, N):
        colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        titles = ['Ground Truth', f'Estimated Rotation It: {it+1}']
        for ax, title in zip(axes, titles):
            ax.set_xlim(-1, N + 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', 'box')
            ax.grid(True)
            ax.set_title(title)
        # Ground truth (static)
        for idx, color in zip(range(N + 1), colors):
            plot_pose((np.array([idx, 0]), gt_rot_mats[idx]), axes[0], color=color)
            plot_pose((np.array([idx, 0]), Rs_est[idx]), axes[1], color=color)

        # Render and capture (robust across backends)
        fig.suptitle("Optimizing Rotations", fontsize=16)
        fig.canvas.draw()
        buf, (w, h) = fig.canvas.print_to_buffer()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        img = img[:, :, :3]           # keep RGB, drop alpha
        plt.close(fig)
        return img

def get_translation_frame(translations, Rs_final, gt_trs, rot_arrs, it, N):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
        for ax, title in zip((ax1, ax2), ('Ground Truth', f'Estimating Translation It: {it+1}')):
            ax.set_xlim(-1, N + 1)
            ax.set_ylim(-1, N + 1)
            ax.set_aspect('equal', 'box')
            ax.grid(True)
            ax.set_title(title)
        fig.suptitle("Optimizing Translations", fontsize=16)

        # ground truth (static)
        for i in range(N + 1):
            plot_pose((gt_trs[i], rot_arrs[i]), ax1, color=colors[i])

        # current estimate
        for i in range(N + 1):
            plot_pose((translations[i], Rs_final[i]), ax2, color=colors[i])

        # render and capture
        fig.canvas.draw()
        buf, (w, h) = fig.canvas.print_to_buffer()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        img = img[:, :, :3]           # drop alpha channel
        plt.close(fig)
        return img