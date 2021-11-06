import numpy as np
from typing import Mapping
import random
from image_utils import p5gb
from toolbox import e2p, p2e

A_B_COMBINATIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def cross_product_matrix(v: np.array):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def triangulate(u1: np.array, u2: np.array, t: np.array, R: np.array) -> (np.array, np.array):
    skew_x2 = cross_product_matrix(u2)
    scale = (skew_x2 @ t) / (skew_x2 @ R @ u1)

    return scale * u1, R @ (scale * u1) + t

def calculate_support(u1: np.array, u2: np.array, correspondences: Mapping,  R: np.array, t: np.array, K: np.array, threshold: float):
    E = (-cross_product_matrix(t) @ R)

    # TODO: try to rewrite this in pure numpy with no loops
    inliers = []
    support = 0
    for i in correspondences.keys():
        line = u2[:, correspondences[i]].T @ E
        line /= (line[0]**2 + line[1]**2)**0.5

        distance = line @ u1[:, i]
        # print(abs(distance))
        if abs(distance) < threshold:
            support += 1
            inliers.append(i)

    return support, inliers






def ransac_epipolar(points1: np.array, points2: np.array, correspondences: Mapping, K: np.array, thresh: float = 3,
                    p: float = 0.99):

    points1 = e2p(points1)
    points2 = e2p(points2)
    points1 = np.linalg.inv(K) @ points1
    points2 = np.linalg.inv(K) @ points2

    # points1 /= points1[2]
    # points2 /= points2[2]

    k = 20
    n = 0
    best_support = 0
    best_R = None
    best_t = None
    best_inliers = []

    while (n := n + 1) < k:
        if n % 20 == 0:
            print(f"Iteration {n} done...")
        random_points = random.sample(correspondences.keys(), 5)
        random_points1 = points1[:, random_points]
        random_points2 = points2[:, [correspondences[i] for i in random_points]]

        possible_Es = p5gb(random_points1, random_points2)

        for E in possible_Es:
            u, s, v_t = np.linalg.svd(E)
            for a, b in A_B_COMBINATIONS:
                u *= np.linalg.det(u)
                v_t *= np.linalg.det(v_t)

                R_21 = u @ np.array([[0, a, 0],
                                     [-a, 0, 0],
                                     [0, 0, 1]]) @ v_t
                t_21 = -b * u[:, 2]

                t_21 /= (-cross_product_matrix(t_21) @ R_21)[0][0] / E[0][0]

                for i in range(5):
                    p1 = random_points1[:, i]
                    p2 = random_points2[:, i]

                    c1_coords, c2_coords = triangulate(p1, p2, t_21, R_21)
                    if c1_coords[2] < 0 or c2_coords[2] < 0:
                        break
                else:
                    support, inliers = calculate_support(points1, points2, correspondences, R_21, t_21, K, thresh)
                    if support > best_support:
                        best_support = support
                        best_R = R_21
                        best_t = t_21
                        best_inliers = inliers
    return best_support, best_inliers, best_R, best_t
