import numpy as np
from toolbox import e2p, p2e
from typing import Callable


def _sac(points: np.array, threshold: float, fit_line: bool, p: float, support_function: Callable):
    points = e2p(points)
    model = None
    best_support = 0
    k = 0
    n_max = float('inf')
    while k < n_max:
        point_a = np.random.randint(0, points.shape[1])
        point_b = np.random.randint(0, points.shape[1])
        if point_a == point_b:
            continue

        random_points = points[:, [point_a, point_b]]

        line = np.cross(random_points[:, 0].T, random_points[:, 1].T)
        line = line.T / ((line[0] ** 2 + line[1] ** 2) ** 0.5)

        distances = np.abs(line @ points)

        if (support := support_function(distances, threshold)) > best_support:
            best_support = support
            model = random_points
            n_max = np.log(1 - p) / np.log(1 - (support / points.shape[1]) ** 2)

        k += 1

    final_line = np.cross(model[:, 0].T, model[:, 1].T)
    if not fit_line:
        return final_line

    final_line = final_line.T / ((final_line[0] ** 2 + final_line[1] ** 2) ** 0.5)

    distances = np.abs(final_line @ points)
    filtered_points = points[:, distances < threshold]

    fitted_line = np.polyfit(filtered_points[0], filtered_points[1], 1)
    return np.array([-fitted_line[0], 1, -fitted_line[1]])


def ransac(points: np.array, threshold: float, p: float = 0.99):
    return _sac(points, threshold, False, p, lambda x, t: sum(x < t))


def ransac_plus_line_fit(points: np.array, threshold: float, p: float = 0.99):
    return _sac(points, threshold, True, p, lambda x, t: sum(x < t))


def mlesac(points: np.array, threshold: float, p: float = 0.99):
    return _sac(points, threshold, False, p, lambda a, t: sum([1 - (x ** 2) / (t ** 2) for x in a if x < t]))


def mlesac_plus_line_fit(points: np.array, threshold: float, p: float = 0.99):
    return _sac(points, threshold, True, p, lambda a, t: sum([1 - (x ** 2) / (t ** 2) for x in a if x < t]))
