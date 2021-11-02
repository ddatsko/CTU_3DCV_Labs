import numpy as np
from typing import Mapping
import random
import p5.python.p5 as p5
from toolbox import e2p, p2e

def ransac_epipolar(points1: np.array, points2: np.array, correspondences: Mapping, thresh: float=3, p: float=0.99):
    points1 = e2p(points1)
    points2 = e2p(points2)

    k = float('inf')
    n = 0
    best_support = 0
    best_E = None

    while n < k:
        random_points = random.sample(correspondences.keys(), 5)
        random_points1 = points1[: random_points]
        random_points2 = points2[:, random_points]

        p5.p5gb(random_points1, random_points2)

