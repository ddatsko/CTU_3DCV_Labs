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
    correspondences = np.fromfile(correspondences_file, dtype=np.int32, sep=' \n')
    return {c[0]: c[1] for c in np.reshape(correspondences, newshape=(int(correspondences.shape[0] / 2), 2))}


def read_points(file_path: str) -> np.array:
    file_data = np.fromfile(file_path, dtype=np.float64, sep=' \n')
    return np.reshape(file_data, newshape=(int(file_data.shape[0] / 2), 2)).T


def show_inliers(points1: np.array, points2: np.array, indexes, correspondences: Mapping, color: str='green'):
    inliers = set(indexes)
    for i in correspondences.keys():
        if i in inliers:
            plt.plot([points1[0][i], points2[0][correspondences[i]]], [points1[1][i], points2[1][correspondences[i]]], color=color)
        # else:
        #     plt.plot([points1[0][i], points2[0][correspondences[i]]], [points1[1][i], points2[1][correspondences[i]]],
        #              color='grey')

def get_k_from_file(filename: str='data/scene_1/K.txt') -> np.array:
    k = np.fromfile(filename, dtype=np.float64, sep=' \n').reshape((3, 3))
    return k