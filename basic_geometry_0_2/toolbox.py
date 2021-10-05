import numpy as np


def e2p(array: np.ndarray) -> np.ndarray:
    return np.vstack(array, [1] * array.shape[1])


def p2e(array: np.ndarray) -> np.ndarray:
    array_copy = array
    array_copy[0] /= array_copy[2]
    array_copy[1] /= array_copy[2]

    return np.delete(array_copy, array_copy.shape[0] - 1, 0)
