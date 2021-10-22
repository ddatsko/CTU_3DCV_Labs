import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping
from toolbox import e2p


def show_points(points: np.array, image_path: str, color: str or tuple = 'red'):
    plt.imshow(plt.imread(image_path))
    plt.scatter(points[0], points[1], color=color, marker='.')
    plt.show()


def get_points_correspondences(correspondences_file: str) -> Mapping[int, int]:
    correspondences = np.fromfile(correspondences_file, dtype=np.int32, sep=' \n')
    return {c[0]: c[1] for c in np.reshape(correspondences, newshape=(int(correspondences.shape[0] / 2), 2))}


def read_points(file_path: str) -> np.array:
    file_data = np.fromfile(file_path, dtype=np.float64, sep=' \n')
    return np.reshape(file_data, newshape=(int(file_data.shape[0] / 2), 2)).T


def show_homography(img_file: str, homography: np.array, points: np.array, color: str or tuple = 'red'):
    points = e2p(points)
    points_h = homography @ points
    points_h /= points_h[2]


    plt.imshow(plt.imread(img_file).dot([1, 1, 1]), cmap='gray')


    for i in range(points.shape[1]):
        plt.plot([points[0, i], points_h[0, i]], [points[1, i], points_h[1, i]], color=color)

def show_points(points1: np.array, points2: np.array, indexes, correspondences: Mapping):
    for i in indexes:
        plt.plot([points1[0][i], points2[0][correspondences[i]]], [points1[1][i], points2[1][correspondences[i]]], color='black')