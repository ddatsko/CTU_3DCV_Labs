import numpy as np


def e2p(array: np.ndarray) -> np.ndarray:
    return np.vstack(array, [1] * array.shape[1])


def p2e(array: np.ndarray) -> np.ndarray:
    return np.delete(array, np.shape[0] - 1, 0)
