from toolbox import *
import rectify
import os
import scipy.io
from typing import List


def run_matlab_algorithm():
    os.system('matlab -nodisplay -nosplash -nodesktop -r "run(\'run_stereo_for_py.m\');exit;"')


def get_rectified_image_task(image1: np.array,
                             image2: np.array,
                             R1: np.array,
                             t1: np.array,
                             R2: np.array,
                             t2: np.array,
                             K: np.array,
                             seed_correspondences: (np.array, np.array)) -> List[np.array]:
    """
    Function for finding 3d point by given 2 images and their P matrices
    @param image1: First image
    @param image2: Second image
    @param R1: Rotation matrix for fhe first camera
    @param t1: Translation of the first camera
    @param R2: Rotation matrix for fhe second camera
    @param t2: Translation of the second camera
    @param K: calibration matrix of both cameras
    @param seed_correspondences: known verified image1 to image2 points correspondences. Two numpy
    arrays of shape (2, n) where n is the number of correspondences

    @return: one task that should be passed to the algorithm in MatLab
    """
    k_inv = np.linalg.inv(K)
    # Compose matrix F
    R_21 = R2 @ R1.T
    F = k_inv.T @ -cross_product_matrix((t2 - R_21 @ t1).flatten()) @ R_21 @ k_inv

    # Compose matrices P for both cameras
    P_1 = Rtk2P(R1, t1, K)
    P_2 = Rtk2P(R2, t2, K)

    # Rectify image and interesting points
    Ha, Hb, img_a_r, img_b_r = rectify.rectify(F, image1, image2)
    image1_points_rectified = p2e(Ha @ e2p(seed_correspondences[0]))
    image2_points_rectified = p2e(Hb @ e2p(seed_correspondences[1]))

    # Compose data for the algorithm in MatLab
    seeds = np.vstack([image1_points_rectified[0],
                       image2_points_rectified[0],
                       (image2_points_rectified[1] + image1_points_rectified[1]) / 2]).T
    task = np.vstack([np.array([img_a_r, img_b_r, seeds], dtype=object)])
    scipy.io.savemat('stereo_in.mat', {'task': task})

    run_matlab_algorithm()

    # Get list of 3d points from the Matlab algorithm output
    d = scipy.io.loadmat('stereo_out.mat')
    D = d['D']

    Ha_inv, Hb_inv = np.linalg.inv(Ha), np.linalg.inv(Hb)

    points = []
    for task_i in range(D.shape[0]):
        Di = D[task_i][0]

        i = Di.shape[0]
        j = Di.shape[1]

        points_1 = np.array([[idx // i for idx in range(i * j)], [idx % i for idx in range(i * j)], np.ones(i * j)])
        points_2 = np.array([[idx // i + Di[idx % i][idx // i] for idx in range(i * j)], [idx % i for idx in range(i * j)], np.ones(i * j)])

        not_nans = np.invert(np.isnan(points_2[0]))

        points_1 = points_1[:, not_nans]
        points_2 = points_2[:, not_nans]

        points_1_orig = Ha_inv @ points_1
        points_2_orig = Hb_inv @ points_2
        points_1_orig /= points_1_orig[2]
        points_2_orig /= points_2_orig[2]

        points.extend(triangulate_to_3d(points_1_orig, points_2_orig, P_1, P_2))

    return points



