import numpy as np
from helpers import stack_block_diag, stack_vecs, vec, project_to_SO2, plot_pose, rot_mat, random_adjacency_matrix, J

"""
    This file is a work in progress. It does not work yet.
    The goal is to abstract away all state / update logic for each robot, but I'm too dumb to understand how for the robots to communicate with each other.
"""

class SLAMRobot:
    def __init__(self, config):
        self.config = config 

        self.idx = 0

        self.plot = False 

        self.N_iters_R = config['Optimization']['Rotation']['N_iters']
        self.N_iters_T = config['Optimization']['Translation']['N_iters']
        self.gamma_R = config['Optimization']['Rotation']['gamma']
        self.gamma_T = config['Optimization']['Translation']['gamma']

        self.estimated_rotation = np.random.random((2, 2))
        self.estimated_translation = np.random.random((2, 1))



    # relative_rots: [Dict] x 2 x 2 matrix of relative rotations 
    #                relative_rots[i,j] = R^i_j
    # neighbor_estimates: |N_i| x 4 x 1 matrix of running estimated rotations
    def update_rotation(self, relative_rots, neighbor_estimates):
        N_neighbors = len(neighbor_estimates)
        self.estimated_rotation = (1 - self.gamma_R) * self.estimated_rotation + \
            self.gamma_R * (1 / (2 * N_neighbors)) * sum(
                [np.kron(relative_rots[(self.idx, j)] + relative_rots[(j, self.idx)].T, np.eye(2)) @ neighbor_estimates[j]
                for j in range(N_neighbors)])