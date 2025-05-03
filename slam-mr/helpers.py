import numpy as np

J = np.array([
    [0, -1],
    [1,0]
])

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
    # If origin is 2x1 make it (2,)
    if origin.shape == (2, 1):
        origin = origin.reshape(-1)
    len = 0.5
    x_end = origin + R[:, 0] * len
    y_end = origin + R[:, 1] * len
    ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], color=color)
    ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], color=color)



def random_adjacency_matrix(N, p=0.5):
    Adj = np.zeros((N + 1, N + 1), dtype=int)
    # While it isn't connected, add edges
    while np.linalg.matrix_rank(Adj) < N + 1:
        for i in range(N + 1):
            for j in range(i + 1, N + 1):
                # Randomly decide if there is an edge
                if np.random.random() < p:
                    Adj[i, j] = 1
                    Adj[j, i] = 1

    return Adj