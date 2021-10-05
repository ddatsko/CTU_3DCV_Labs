import numpy as np


class Point:
    def __init__(self, initial_array: np.ndarray or None = None):
        if initial_array is None:
            self.array = np.ndarray(shape=(3, 1), dtype=float)
        elif len(initial_array) == 2:
            self.array = np.ndarray(shape=(3, 1), dtype=float)
            self.array[0, 0] = initial_array[0]
            self.array[1, 0] = initial_array[1]
            self.array[2, 0] = 1
        else:
            self.array = initial_array

    def x(self):
        return self.array[0][0] / self.array[2][0]

    def y(self):
        return self.array[1][0] / self.array[2][0]

    def coordinates(self) -> list:
        return [self.x(), self.y()]

    def apply_homography(self, homography: np.ndarray):
        self.array = homography.dot(self.array)
        self.array /= self.array[2][0]
