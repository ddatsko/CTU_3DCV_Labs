import cv2
from utils import *
import numpy as np
from toolbox import *
from typing import Iterable
import rectify
import matplotlib.pyplot as plt


def get_rectified_image_task(image1: np.array,
                                 image2: np.array,
                                 F: np.array,
                                 seed_correspondences: (np.array, np.array)) -> np.array:
    """
    Function for finding 3d point by given 2 images and their P matrices
    @param image1: First image
    @param image2: Second image
    @param F: fundamental matrix for cameras that took image 1 and image 2
    @param seed_correspondences: known verified image1 to image2 points correspondences. Two numpy
    arrays of shape (2, n) where n is the number of correspondences

    @return: one task that should be passed to the algorithm in MatLab
    """
    Ha, Hb, img_a_r, img_b_r = rectify.rectify(F, image1, image2)
    image1_points_rectified = p2e(Ha @ e2p(seed_correspondences[0]))
    image2_points_rectified = p2e(Hb @ e2p(seed_correspondences[1]))

    seeds = np.vstack([image1_points_rectified[0],
                       image2_points_rectified[0],
                       (image2_points_rectified[1] + image1_points_rectified[1]) / 2]).T

    return np.array([img_a_r, img_b_r, seeds], dtype=object)
