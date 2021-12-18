import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping
from toolbox import e2p
from p5.ext import *


def p5gb(u1, u2):
    """
    Five-point calibrated relative pose problem (Grobner basis).
    Es = p5gb( u1, u2 ) computes the esential matrices E according to
    Nister-PAMI2004 and Stewenius-PRS2006.

    Input:
      u1, u2 ..  3x5 matrices, five corresponding points, HOMOGENEOUS coord.

    Output:
      Es .. list of possible essential matrices
    """

    q = np.vstack((u1[0] * u2[0], u1[1] * u2[0], u1[2] * u2[0],
                   u1[0] * u2[1], u1[1] * u2[1], u1[2] * u2[1],
                   u1[0] * u2[2], u1[1] * u2[2], u1[2] * u2[2])).T

    U, S, Vt = np.linalg.svd(q)

    XYZW = Vt[5:, :]

    A = p5_matrixA(XYZW)  # in/out is transposed (numpy data are row-wise)

    A = A[[5, 9, 7, 11, 14, 17, 12, 15, 18, 19]] @ \
        np.linalg.inv(A[[0, 2, 3, 1, 4, 8, 6, 10, 13, 16]])

    A = A.T

    M = np.zeros((10, 10))
    M[:6] = -A[[0, 1, 2, 4, 5, 7]]

    M[6, 0] = 1
    M[7, 1] = 1
    M[8, 3] = 1
    M[9, 6] = 1

    D, V = np.linalg.eig(M)

    ok = np.imag(D) == 0.0
    V = np.real(V)

    SOLS = V[6:9, ok] / V[9:10, ok]

    SOLS = SOLS.T

    Evec = SOLS @ XYZW[:3] + XYZW[3]

    Es = [None] * Evec.shape[0]
    for i in range(0, Evec.shape[0]):
        Es[i] = Evec[i].reshape((3, 3))
        Es[i] = Es[i] / np.sqrt(np.sum(Es[i] ** 2))

    return Es


def get_points_correspondences(correspondences_file: str) -> Mapping[int, int]:
    """
    Read points correspondences from specified file
    @param correspondences_file: name of the file to read
    @return: dict in form {image1_point: image2_point}
    """
    correspondences = np.fromfile(correspondences_file, dtype=np.int32, sep=' \n')
    return {c[0]: c[1] for c in np.reshape(correspondences, newshape=(int(correspondences.shape[0] / 2), 2))}


def read_points(file_path: str) -> np.array:
    """
    Read interesting points from the specified file
    @param file_path: path of the file to read points from
    @return: Points in np.array of shape (2, n), where n is the number of points
    """
    file_data = np.fromfile(file_path, dtype=np.float64, sep=' \n')
    return np.reshape(file_data, newshape=(int(file_data.shape[0] / 2), 2)).T


def show_inliers(points1: np.array, points2: np.array, indexes, correspondences: Mapping,
                 color: tuple or str = 'green'):
    inliers = set(indexes)
    for i in correspondences.keys():
        if i in inliers:
            plt.plot([points1[0][i], points2[0][correspondences[i]]], [points1[1][i], points2[1][correspondences[i]]],
                     color=color)
        else:
            plt.plot([points1[0][i], points2[0][correspondences[i]]], [points1[1][i], points2[1][correspondences[i]]],
                     color='black', linewidth=0.3)


def get_k_from_file(filename: str = 'data/scene_1/K.txt') -> np.array:
    k = np.fromfile(filename, dtype=np.float64, sep=' \n').reshape((3, 3))
    return k


def plot_line(line, color: str or tuple = 'blue'):
    line /= (line[0] ** 2 + line[1] ** 2) ** 0.5
    ax = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
    plt.plot(ax, list(
        map(lambda x: (-line[2] - line[0] * x) / line[1] if min(plt.ylim()) < (-line[2] - line[0] * x) / line[1] < max(
            plt.ylim()) else None, ax)), color=color)


def get_image_image_correspondences(image1_points: np.array,
                                    image2_points: np.array,
                                    Xu1: (np.array, np.array, np.array),
                                    Xu2: (np.array, np.array, np.array)) -> (np.array, np.array):
    """
    Get coordinates corresponding points of 2 images from image point to 3D point correspondences
    @param image1_points: pixel coordinates of first image interesting points
    @param image2_points: pixel coordinates of second image interesting points
    @param Xu1: (X, u, verified): 3d points indices, corresponding image1 interesting points indices, array of booleans
    where verified[i] means that correspondence is verified by reprojection error
    @param Xu2: (X, u, verified): 3d points indices, corresponding image2 interesting points indices, array of booleans
    where verified[i] means that correspondence is verified by reprojection error
    @return: 2 arrays: pixel coordinates of first image points and pixel coordinates of corresponding second image points
    """
    X1, u1, verified1 = Xu1
    X2, u2, verified2 = Xu2
    world_to_im1 = {X1[i]: u1[i] for i in range(X1.shape[0]) if verified1[i]}
    world_to_im2 = {X2[i]: u2[i] for i in range(X2.shape[0]) if verified2[i]}

    res = ([], [])
    for world_point in world_to_im1.keys():
        if world_point in world_to_im2:
            res[0].append(image1_points[:, world_to_im1[world_point]])
            res[1].append(image2_points[:, world_to_im2[world_point]])

    return np.array(res[0]).T, np.array(res[1]).T
