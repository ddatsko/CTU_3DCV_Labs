import numpy as np
from point import Point


class Line:
    def __init__(self, initial_array: np.ndarray or None = None):
        if initial_array is None:
            self.array = np.ndarray(shape=(1, 3), dtype=float)
        else:
            self.array = initial_array

    def intersection(self, other_line: 'Line') -> Point:
        p_intersection = Point(np.cross(self.array.T, other_line.array.T).T)
        scale = p_intersection.array[2][0]
        if scale != 0:
            p_intersection.array /= scale
        return p_intersection

    def apply_homography(self, homography: np.ndarray) -> None:

        self.array = homography * self.array

    @staticmethod
    def from_two_points(point1: Point, point2: Point) -> 'Line':
        return Line(np.cross(point1.array.T, point2.array.T).T)
