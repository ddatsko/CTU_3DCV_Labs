import numpy as np
from typing import List


def e2p(array: np.ndarray) -> np.ndarray:
    return np.vstack([array, np.array([1] * array.shape[1])])


def p2e(array: np.ndarray) -> np.ndarray:
    array /= array[-1]
    return array[:-1]


def cross_product_matrix(v: np.array):
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def triangulate(u1: np.array, u2: np.array, t: np.array, R: np.array) -> (np.array, np.array):
    """
    Returns: X in first camera's coordinate system and in the second camera's one
    """
    P_2 = np.append(R, t.reshape((3, 1)), axis=1)
    P_1 = np.append(np.identity(3), np.array([[0], [0], [0]]), axis=1)

    X = triangulate_to_3d_hom(u1, u2, P_1, P_2)
    return P_1 @ X, P_2 @ X


def triangulate_to_3d_hom(u1: np.array, u2: np.array, P_1: np.array, P_2: np.array) -> np.array:
    """
    Returns: X in world coordinates
    """
    D = np.array([u1[0] * P_1[2] - P_1[0],
                  u1[1] * P_1[2] - P_1[1],
                  u2[0] * P_2[2] - P_2[0],
                  u2[1] * P_2[2] - P_2[1]], dtype=np.float32)

    u, o, v_t = np.linalg.svd(D)

    X = v_t.T[:, -1]
    X /= X[-1]
    return X


def sampson_corrected(u1: np.array, u2: np.array, F: np.array) -> (np.array, np.array):
    """
    Sampson correction for multiple points of the same pair of images and common matrix F
    TODO: leave only this function and remove the @sampson_corrected one
    @param u1: 2d np array of homogeneous points coordinates of the first image of shape (3, n)
    @param u2: 2d np array of homogeneous point coordinates of the second image of shape (3, n)
    @param F: Common fundamental matrix for correspondences
    @return: 2 np arrays: coordinates of corrected points on both images of shapes (2, n)
    """
    S = np.array([[1, 0, 0], [0, 1, 0]])
    v = np.array([u1[0], u1[1], u2[0], u2[1]])

    v -= (np.einsum('ij,ij->i', u2.T, (F @ u1).T) /
          (np.sum((S @ F @ u1) ** 2, axis=0) + np.sum((S @ F.T @ u2) ** 2, axis=0))) * np.array([(F[:, 0].reshape((1, 3)) @ u2)[0],
                                                                                                 (F[:, 1].reshape((1, 3)) @ u2)[0],
                                                                                                 (F[0].reshape((1, 3)) @ u1)[0],
                                                                                                 (F[1].reshape((1, 3)) @ u1)[0]])
    return v[:2], v[2:]


def triangulate_to_3d(u1: np.array, u2: np.array, P_1: np.array, P_2: np.array) -> np.array or List[np.array]:
    """
    Returns: X in world coordinates
    """
    # K_inv = np.linalg.inv(K)
    # # Using sampson correction here
    Q1, Q2 = P_1[:, :3], P_2[:, :3]
    q1, q2 = P_1[:, 3].reshape((3, 1)), P_2[:, 3].reshape((3, 1))
    F = (Q1 @ np.linalg.inv(Q2)).T @ cross_product_matrix(q1 - (Q1 @ np.linalg.inv(Q2)) @ q2)

    if u1.shape[1] == 1:
        u1, u2 = sampson_corrected(u1, u2, F)
        return triangulate_to_3d_hom(u1.flatten(), u2.flatten(), P_1, P_2)[:-1]

    else:

        u1, u2 = sampson_corrected(u1, u2, F)
        res = []
        for i in range(u1.shape[1]):
            res.append(triangulate_to_3d_hom(u1[:, i].flatten(), u2[:, i].flatten(), P_1, P_2)[:-1])
        return res


def R_from_rodrigues(a: np.array):
    # Using this formula: https://i.stack.imgur.com/a6wRa.png
    alpha = np.linalg.norm(a)
    a = a / alpha

    aa_t = a.reshape((3, 1)) @ a.reshape((1, 3))
    A = cross_product_matrix(a)

    R = np.identity(3) * np.cos(alpha) + aa_t * (1 - np.cos(alpha)) + A * np.sin(alpha)

    return R


def Rt2P(R: np.array, t: np.array) -> np.array:
    return np.append(R, t.reshape((3, 1)), axis=1)


def Rtk2P(R: np.array, t: np.array, k: np.array):
    return np.append(k @ R, k @ t.reshape((3, 1)), axis=1)


def Rt_from_array(x: np.array, R, t):
    if np.linalg.norm(x[:3]) == 0:
        d_R = np.identity(3)
    else:
        d_R = R_from_rodrigues(np.array(x[:3]))

    d_t = np.array(x[3:])

    # Define new values
    return R @ d_R, t + d_t


def reprojection_error(P: np.array, X: np.array, u: np.array):
    reprojected_inliers = P @ X
    reprojected_inliers /= reprojected_inliers[2]

    res = np.sum((reprojected_inliers - u) ** 2, axis=1)

    return np.sum(np.sqrt(res))


def sampson_jacobian(F, x1, x2) -> np.array:
    # return np.vstack([(F.T @ x1)[:-1], (F @ x2)[:-1]])
    return np.vstack([(F.T @ x2)[:-1], (F @ x1)[:-1]])
