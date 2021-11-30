import numpy as np
from toolbox import *
from utils import *
import p3p
from typing import Mapping
import random
import scipy.optimize


def optimize_error(X, u, R, t, K):
    min_error = float('inf')

    def error(x: np.array):
        new_R, new_t = Rt_from_array(x, R, t.reshape((1, 3)))
        P = Rt2P(K @ new_R, K @ new_t.reshape((3, 1)))

        res = reprojection_error(P, X, u)

        nonlocal min_error
        if min_error > res:
            min_error = res
            # print("Optimized error: ", min_error)

        return res

    optimized_x = scipy.optimize.minimize(error, np.array([0, 0, 0, 0, 0, 0])).x

    return Rt_from_array(optimized_x, R, t.reshape((1, 3)))


def _calculate_p3p_support(P, world_points, image_points, threshold):
    reprojected = (P @ world_points)
    reprojected /= reprojected[2]
    distances = np.sqrt(np.sum((reprojected.T - image_points.T) ** 2, axis=1))

    inliers = np.where(np.abs(distances) < threshold)[0]
        # print(best_support, k)
    return inliers, np.sum(1 - (distances[np.abs(distances) < threshold] ** 2) / (threshold ** 2))


def ransac_p3p(world_points: Mapping[int, np.array], X: np.array, image_interesting_points: np.array, u: np.array,
               K: np.array, threshold: float = 2, p: float = 0.9999):
    K_inv = np.linalg.inv(K)

    image_interesting_points = e2p(image_interesting_points)

    all_world_points = e2p(np.array([world_points[i] for i in X]).T)
    all_image_points = np.array([image_interesting_points[:, i] for i in u]).T

    # Save original points (without K undone to be able to measure error in pixels
    image_interesting_points = K_inv @ image_interesting_points
    image_interesting_points /= image_interesting_points[2]

    n = float('inf')
    k = 0

    possible_random_points = list(range(len(X)))
    best_support = 0
    best_R = None
    best_t = None
    best_inliers = []

    while k < n:
        chosen_points = random.sample(possible_random_points, 3)

        X_points = np.array([all_world_points[:, i] for i in chosen_points]).T
        u_points = np.array([image_interesting_points[:, u[i]] for i in chosen_points])

        possible_Xs = p3p.p3p_grunert(X_points, u_points.T)

        # print(X_points)
        # print(world_points[X[chosen_points[0]]])

        for possible_X in possible_Xs:
            # TODO: check the order of parameters here
            R, t = p3p.XX2Rt_simple(X_points, possible_X)

            # Reconstruct P with K to make error measurements be in pixels
            P = Rt2P(K @ R, K @ t)
            inliers, support = _calculate_p3p_support(P, all_world_points, all_image_points, threshold)
            if support > best_support:
                best_support = support
                best_R = R
                best_t = t
                best_inliers = np.array(inliers)
                print(f"Best support: {best_support}")

        if np.isclose(np.log(1 - (best_support / len(X)) ** 3), 0):
            n = n
        else:
            n = min(1000, np.log(1 - p) / np.log(1 - (best_support / len(X)) ** 3))
        k += 1


    # print(f"R, t before optimization: \n{best_R}\n{best_t}")
    R, t = optimize_error(all_world_points[:, best_inliers],
                          all_image_points[:, best_inliers],
                          best_R,
                          best_t,
                          K
                          )
    # print(f"R, t after optimization: \n{R}\n{t}")

    return best_inliers, best_R, best_t
