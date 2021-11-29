import numpy as np


def e2p(array: np.ndarray) -> np.ndarray:
    return np.vstack([array, np.array([1] * array.shape[1])])


def p2e(array: np.ndarray) -> np.ndarray:
    return np.delete(array, np.shape[0] - 1, 0)


def cross_product_matrix(v: np.array):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def triangulate(u1: np.array, u2: np.array, t: np.array, R: np.array) -> (np.array, np.array):
    """
    Returns: X in first camera's coordinate system and in the second camera's one
    """
    P_2 = np.append(R, t.reshape((3, 1)), axis=1)
    P_1 = np.append(np.identity(3), np.array([[0], [0], [0]]), axis=1)

    X = triangulate_to_3d_hom(u1, u2, t, R)
    return P_1 @ X, P_2 @ X


def triangulate_to_3d_hom(u1: np.array, u2: np.array, t: np.array, R: np.array, K: np.array = np.identity(3)) -> np.array:
    """
    Returns: X in world coordinates
    """
    R = K @ R
    t = K @ t
    P_2 = np.append(R, t.reshape((3, 1)), axis=1)
    P_1 = np.append(K @ np.identity(3), np.array([[0], [0], [0]]), axis=1)

    D = np.array([u1[0] * P_1[2] - P_1[0],
                  u1[1] * P_1[2] - P_1[1],
                  u2[0] * P_2[2] - P_2[0],
                  u2[1] * P_2[2] - P_2[1]])

    u, o, v_t = np.linalg.svd(D)

    X = v_t.T[:, -1]
    X /= X[-1]
    return X


def triangulate_to_3d(u1: np.array, u2: np.array, t: np.array, R: np.array, K: np.array) -> np.array:
    """
    Returns: X in world coordinates
    """
    # print(triangulate_to_3d_hom(u1, u2, t, R))(u1, u2, t, R))
    return triangulate_to_3d_hom(u1, u2, t, R, K)[:-1]


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