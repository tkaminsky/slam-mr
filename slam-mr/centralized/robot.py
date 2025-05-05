import numpy as np
from helpers import stack_block_diag, stack_vecs, vec, project_to_SO2, plot_pose, rot_mat, random_adjacency_matrix, J

"""
    This file is a work in progress. It does not work yet.
    The goal is to abstract away all state / update logic for each robot, but I'm too dumb to understand how for the robots to communicate with each other.
"""

class SLAMRobot:
    def __init__(self, config, id=None):

        if id is None:
            self.id = np.random.randint(0, 10000)
        else:
            self.id = id
        self.name = f"Robot {self.id}"
        self.config = config

        self.ploting = False

        self.pose = {
            'translation': np.zeros((2,1)),
            'rotation':np.eye(2)
        }

        self.N = config['N']

        self.N_iters_R = config['Optimization']['Rotation']['N_iters']
        self.N_iters_T = config['Optimization']['Translation']['N_iters']
        self.gamma_R = config['Optimization']['Rotation']['gamma']
        self.gamma_T = config['Optimization']['Translation']['gamma']

        self.estimated_rotation = vec(np.random.random((2, 2)))
        self.neighbor_estimated_rotations = {}
        self.estimated_translation = np.random.random((2, 1))
        self.estimated_translation[0] = self.estimated_translation[0] * 6 - 3
        self.estimated_translation[1] = self.estimated_translation[1] * 4 - 2

        self.neighbor_estimated_translations = {}
        self.rotation_estimates = {}

    def init_rendezvous(self, neighbors, relative_rots, relative_translations):
        self.N_neighbors = len(neighbors)
        self.neighbors = neighbors
        # Dictionary of relative rotations-- relative_rots[(i,j)] = R^i_j
        self.relative_rots = relative_rots
        # Dictionary of relative translations-- relative_translations[(i,j)] = t^i_j
        self.relative_translations = relative_translations

        # Clear neighbor estimates
        self.neighbor_estimated_rotations = {}
        self.neighbor_estimated_translations = {}
        self.rotation_estimates = {}


    def update_neighbor_estimated_rotations(self, indices, estimated_rotations):
        for idx, neighbor in enumerate(indices):
            self.neighbor_estimated_rotations[neighbor] = estimated_rotations[idx]

    def update_neighbor_estimated_translations(self, indices, estimated_translations):
        for idx, neighbor in enumerate(indices):
            self.neighbor_estimated_translations[neighbor] = estimated_translations[idx]

    def set_pose(self, pose):
        translation, rotation = pose
        self.pose['translation'] = translation
        self.pose['rotation'] = rotation

    def set_rotation(self, rotation):
        self.pose['rotation'] = rotation

    def set_translation(self, translation):
        self.pose['translation'] = translation

    def set_relative_rots(self, relative_rots):
        self.relative_rots = relative_rots
        self.N_neighbors = len(self.relative_rots)
        self.relative_rots = {k: v for k, v in relative_rots.items() if k[0] == self.idx or k[1] == self.idx}

    # relative_rots: [Dict] x 2 x 2 matrix of relative rotations
    #                relative_rots[i,j] = R^i_j
    # neighbor_estimates: |N_i| x 4 x 1 matrix of running estimated rotations
    def update_estimated_rotation(self):
        # Force reference frame
        if self.id == 0:
            self.estimated_rotation = vec(np.eye(2))
            return

        # Update estimate
        self.estimated_rotation = (1 - self.gamma_R) * self.estimated_rotation + \
            self.gamma_R * (1 / (2 * self.N_neighbors)) * sum(
                [np.kron(self.relative_rots[(self.id, j)] + self.relative_rots[(j, self.id)].T, np.eye(2)) @ self.neighbor_estimated_rotations[j]
                for j in self.neighbors])

    def update_estimated_translation(self):
        # Reference frame is the first robot
        if self.id == 0:
            self.estimated_translation = np.zeros((2,1))
            return

        # If rotation estimates are empty, get them all by projecting estimated rotations to SO2
        if len(self.rotation_estimates) == 0:
            self.rotation_estimates = {k: project_to_SO2(v.reshape((2,2)).T) for k, v in self.neighbor_estimated_rotations.items()}
            self.rotation_estimates[self.id] = project_to_SO2(self.estimated_rotation.reshape((2,2)).T)

        g_agent = sum(
            [
                self.rotation_estimates[k] @ self.relative_translations[(k,self.id)] - self.rotation_estimates[self.id]@self.relative_translations[(self.id,k)]
                for k in self.neighbors
            ]
        )

        Hy_sum = sum(
            [
                -2 * self.neighbor_estimated_translations[k] for k in self.neighbors
            ]
        )

        self.estimated_translation = (1 - self.gamma_T )* self.estimated_translation + \
            self.gamma_T * 1 / (2 * self.N_neighbors) * ( - Hy_sum + g_agent)
