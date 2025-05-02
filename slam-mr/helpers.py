import numpy as np

# Stacks a list of matrices into a block-diagonal matrix.
def stack_block_diag(mats):
    k = len(mats)
    I = np.eye(k, dtype=int)
    # sum E_{ii} ⊗ A_i
    B = sum(np.kron(I[i:i+1, i:i+1], mats[i]) for i in range(k))
    return B

# Stacks a list of vectors into a single column vector.
def stack_vecs(vecs):
    """
    Stack vectors into a single column vector.
    """
    return np.concatenate([v.reshape(-1, 1) for v in vecs], axis=0)

def project_to_SO2(A):
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def vec(M):
    return M.reshape(-1, order='F')

def plot_pose(pose, ax, color='k'):
    origin, R = pose
    x_end = origin + R[:, 0] * 0.2
    y_end = origin + R[:, 1] * 0.2
    ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], color=color)
    ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], color=color)