import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping
from toolbox import e2p


def get_points_correspondences(correspondences_file: str) -> Mapping[int, int]:
    correspondences = np.fromfile(correspondences_file, dtype=np.int32, sep=' \n')
    return {c[0]: c[1] for c in np.reshape(correspondences, newshape=(int(correspondences.shape[0] / 2), 2))}


def read_points(file_path: str) -> np.array:
    file_data = np.fromfile(file_path, dtype=np.float64, sep=' \n')
    return np.reshape(file_data, newshape=(int(file_data.shape[0] / 2), 2)).T


def show_points(points1: np.array, points2: np.array, indexes, correspondences: Mapping):
    for i in indexes:
        plt.plot([points1[0][i], points2[0][correspondences[i]]], [points1[1][i], points2[1][correspondences[i]]], color='black')