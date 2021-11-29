import numpy as np
from toolbox import *
from utils import *
import p3p
from typing import Mapping
import random


def _calculate_p3p_support(P, world_points, image_points, threshold):
    reprojected = (P @ world_points)
    reprojected /= reprojected[2]
    distances = np.sqrt(np.sum((reprojected.T - image_points.T) ** 2, axis=1))

    inliers = np.where(np.abs(distances) < threshold)





def ransac_p3p(world_points: Mapping[int, np.array], X: np.array, image_interesting_points: np.array, u: np.array, K: np.array, threshold: float=2, p: float=0.99):

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

    while k < n:
        chosen_points = random.sample(possible_random_points, 3)

        X_points = e2p(np.array([world_points[X[i]] for i in chosen_points]))
        u_points = np.array([image_interesting_points[:, u[i]] for i in chosen_points])

        possible_Xs = p3p.p3p_grunert(X_points, u_points)

        for possible_X in possible_Xs:
            # TODO: check the order of parameters here
            R, t = p3p.XX2Rt_simple(X_points, possible_X)
            print(R, t)
            print(X_points, possible_X)
            P = Rt2P(K @ R, K @ t)

            print(P @ X_points[:, 0])
            print(possible_X[:, 0])

            support = _calculate_p3p_support(P, all_world_points, all_image_points, threshold)



        break

