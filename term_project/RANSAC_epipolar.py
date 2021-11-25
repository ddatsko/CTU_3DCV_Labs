import numpy as np
from typing import Mapping
import random
from utils import p5gb
from toolbox import *
import scipy.optimize

A_B_COMBINATIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def sampson_approx_jacobian(F, x1, x2) -> np.array:
    return np.vstack([(F.T @ x1)[:-1], (F @ x2)[:-1]])


def fit_E_matrix(u1: np.array, u2: np.array, R: np.array, t: np.array, K: np.array):
    k_inv = np.linalg.inv(K)
    min_error = float('inf')

    def epsilon(x: np.array, F: np.array, y: np.array):
        # return np.array([y[:, i].T @ F @ x[:, i] for i in range(x.shape[1])])
        return np.einsum('ij,ij->i', y.T, (F @ x).T)

    def error(x):
        # Calculate values from 6 parameters
        if np.linalg.norm(x[:3]) == 0:
            d_R = np.identity(3)
        else:
            d_R = R_from_rodrigues(np.array(x[:3]))

        d_t = np.array(x[3:])

        # Define new values
        new_R = R @ d_R
        new_t = t + d_t

        F = k_inv.T @ (cross_product_matrix(-new_t) @ new_R) @ k_inv

        # # Calculate error with new values
        jacobians = sampson_approx_jacobian(F, u1, u2).T
        JJ_T = 1 / np.einsum('ij,ij->i', jacobians, jacobians)

        eps = epsilon(u1, F, u2)
        errors = JJ_T * (eps ** 2)

        res = np.sum(errors)

        # Print the output about new min error function value
        nonlocal min_error
        if res < min_error:
            print(f"Optimized min error to {res}")
            min_error = res

        return res

    argmin_x = scipy.optimize.minimize(error, np.array([0, 0, 0, 0, 0, 0])).x
    return R @ R_from_rodrigues(np.array(argmin_x[:3])), t + np.array(argmin_x[3:])


def calculate_support(u1: np.array, u2: np.array, correspondences: Mapping, E: np.array, K: np.array, threshold: float):
    k_inv = np.linalg.inv(K)
    F = k_inv.T @ E @ k_inv

    keys = list(sorted(correspondences.keys()))

    points1 = u1[:, keys]
    points2 = u2[:, [correspondences[i] for i in keys]]

    lines1 = F.T @ points2
    lines1 /= (lines1[0] ** 2 + lines1[1] ** 2) ** 0.5
    distances1 = np.einsum('ij,ij->i', lines1.T, points1.T)

    lines2 = F @ points1
    lines2 /= (lines2[0] ** 2 + lines2[1] ** 2) ** 0.5
    distances2 = np.einsum('ij,ij->i', lines2.T, points2.T)

    distances = distances1 + distances2 / 2

    return sum(
        [1 - (distances[i] ** 2) / (threshold ** 2) for i in range(len(keys)) if abs(distances[i]) < threshold]), [
               keys[i] for i in range(len(keys)) if abs(distances[i]) < threshold]


def ransac_epipolar(points1: np.array, points2: np.array, correspondences: Mapping, K: np.array, thresh: float = 3,
                    p: float = 0.99):
    points1 = e2p(points1)
    points2 = e2p(points2)

    points1_original = points1
    points2_original = points2

    points1 = np.linalg.inv(K) @ points1
    points2 = np.linalg.inv(K) @ points2

    k = float('inf')
    n = 0
    best_support = 5
    best_R = None
    best_t = None
    best_inliers = []
    best_chosen_points = None

    while (n := n + 1) < k:
        if n % 20 == 0:
            print(f"Iteration {n} done...")
        random_points = random.sample(correspondences.keys(), 5)
        random_points1 = points1[:, random_points]
        random_points2 = points2[:, [correspondences[i] for i in random_points]]
        possible_Es = p5gb(random_points1, random_points2)

        for E in possible_Es:
            support, inliers = calculate_support(points1_original, points2_original, correspondences, E, K, thresh)
            if support > best_support:
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
                        best_support = support
                        best_R = R_21
                        best_t = t_21
                        best_inliers = inliers
        if np.log(1 - (best_support / points1.shape[1]) ** 5) != 0:
            k = min(np.log(1 - p) / np.log(1 - (len(best_inliers) / len(correspondences.keys())) ** 5), 10000)

    best_R, best_t = fit_E_matrix(points1_original[:, sorted(best_inliers)],
                                  points2_original[:, [correspondences[i] for i in sorted(best_inliers)]], best_R,
                                  best_t, K)

    return best_support, best_inliers, best_R, best_t
