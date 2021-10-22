import numpy as np
from typing import Callable, Mapping
import random
import scipy.linalg
from toolbox import e2p


def points_distance(p1: np.array, p2: np.array):
    p1 = p1 / p1[2]
    p2 = p2 / p2[2]
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def homography_distances(points_from: np.array, points_to: np.array, homography: np.array):
    points_under_hom = homography @ e2p(points_from)
    points_under_hom /= points_under_hom[2]
    diff = points_under_hom - e2p(points_to)

    return (diff[0] ** 2 + diff[1] ** 2) ** 0.5


def homography_support(image1_points: np.array, image2_points: np.array, homography: np.array, threshold: float):
    distances1 = homography_distances(image1_points, image2_points, homography)
    distances2 = homography_distances(image2_points, image1_points, np.linalg.inv(homography))
    distances = (distances1 + distances2) / 2

    return image1_points[:, distances < threshold], np.array(list(range(len(image1_points[0]))))[distances < threshold], \
            sum((1 - (distances[i] / threshold) ** 2 if distances[i] < threshold else 0 for i in range(len(distances))))


def homography_ransac(image1_points: np.array, image2_points: np.array, correspondences: Mapping, theta: float,
                      support_function: Callable, p: float = 0.99, n=-1):
    k = 0
    best_support = 0
    best_support_points_a = None
    best_support_points_b = None
    best_support_homography_a = None
    best_support_homography_b = None
    not_supported_points = None
    best_a = None
    if n == -1:
        n_max = float('inf')
    else:
        n_max = n
    possible_points = tuple(sorted(list(correspondences.keys())))

    while k < n_max:
        if k % 10 == 0:
            print(f'{k} iteration done')
        random_points = random.sample(possible_points, 4)
        a = []
        for point in random_points:
            x1 = image1_points[:, point].T
            x2 = image2_points[:, correspondences[point]].T

            a.append([x1[0], x1[1], 1, 0, 0, 0, -x2[0] * x1[0], -x2[0] * x1[1], -x2[0]])
            a.append([0, 0, 0, x1[0], x1[1], 1, -x2[1] * x1[0], -x2[1] * x1[1], -x2[1]])
        a = np.array(a)

        h_a = scipy.linalg.null_space(a)
        h_a = h_a.reshape((3, 3))
        support_points_a, support_points_indexes, support = homography_support(image1_points[:, possible_points],
                                                                      image2_points[:,
                                                                      [correspondences[i] for i in possible_points]],
                                                                      h_a, theta)


        restricted_possible_points = list(possible_points)
        for i in support_points_indexes:
            restricted_possible_points.remove(possible_points[i])

        second_homography_points_indexes = random.sample(restricted_possible_points, 3)

        h_a_inv = np.linalg.inv(h_a)
        u = h_a_inv @ e2p(image2_points[:, [correspondences[i] for i in second_homography_points_indexes]])
        Hu = e2p(image1_points[:, second_homography_points_indexes])

        u /= u[2]

        v = np.cross(np.cross(u[:, 0].T, Hu[:, 0].T), np.cross(u[:, 1].T, Hu[:, 1].T))
        #
        A = np.array([(Hu[0][0] * v[2] - Hu[2][0] * v[0]) * u[:, 0].T,
                      (Hu[0][1] * v[2] - Hu[2][1] * v[0]) * u[:, 1].T,
                      (Hu[0][2] * v[2] - Hu[2][2] * v[0]) * u[:, 2].T])

        b = np.array([[u[0][0] * Hu[2][0] - u[2][0] * Hu[0][0]],
                      [u[0][1] * Hu[2][1] - u[2][1] * Hu[0][1]],
                      [u[0][2] * Hu[2][2] - u[2][2] * Hu[0][2]]])

        # Sometimes matrix A seems to be singular (approx 1 time per 2000-3000 runs)
        try:
            a = np.linalg.inv(A) @ b
        except np.linalg.LinAlgError:
            continue

        H = np.identity(3) + (np.reshape(v, (3, 1)) @ a.T)

        h_b = np.linalg.inv(H @ h_a_inv)

        support_points_b, support_points_indexes, support_b = homography_support(image1_points[:, restricted_possible_points],
                                                 image2_points[:,
                                                 [correspondences[i] for i in restricted_possible_points]],
                                                 h_b, theta)

        support += support_b

        if support > best_support:
            best_a = a
            best_support_points_a = support_points_a
            best_support_points_b = support_points_b
            not_supported_points = restricted_possible_points.copy()

            for i in support_points_indexes:
                not_supported_points.remove(restricted_possible_points[i])

            best_support = support
            best_support_homography_a = h_a
            best_support_homography_b = h_b

            if n == -1:
                n_max = np.log(1 - p) / np.log(1 - (support / image1_points.shape[1]) ** 2)
        k += 1
    return best_support_homography_a, best_support_points_a, best_support_homography_b, best_support_points_b, best_a, not_supported_points
