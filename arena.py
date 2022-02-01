import numpy as np
import os
from os import path
import pickle
from camera import file_to_camera


arena_shape = (99, 70, 40)


def camera_to_world(r, t, point):
    point = point - t
    m = np.linalg.inv(np.matrix(r))
    return m * point


def world_to_camera(r, t, point):
    point = point + t
    m = np.matrix(r)
    return m * point


def load_arena(dir, timestamp):
    matrices_files = [file for file in os.listdir(dir) if file.find(timestamp) != -1 and file.find('.data') != -1]
    matrices = {}
    for file in matrices_files:
        print('loading matrix from file:', file)
        f = open(path.join(dir, file), 'rb')
        m = pickle.load(f)
        matrices[file_to_camera(file)] = m
        f.close()
    return Arena(matrices)


def in_arena(p, noise = 10):
    if len(p) == 0:
        return False
    x, y, z = p[0], p[1], p[2]
    x_max, y_max, z_max = arena_shape
    if min(x, y, z) < -1 * noise:
        return False
    if x > x_max + noise or y > y_max + noise or z > z_max + noise:
        return False
    return True


class Arena:
    def __init__(self, cameras_matrices):
        self.cameras_matrices = cameras_matrices

    def get_camera_matrices(self, camera):
        return self.cameras_matrices[camera]

    def translate_point_to_world(self, camera, p):
        if camera not in self.cameras_matrices:
            print('camera is not initialized in arena')
            return
        r, t = self.cameras_matrices[camera]
        m = np.linalg.inv(np.matrix(r))
        return m * (p - t)

