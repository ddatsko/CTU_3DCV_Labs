import numpy as np
import matplotlib.pyplot as plt
from point import Point
from line import Line

EPSILON = 0.0000001

HOMOGRAPHY = np.array([[1, 0.1, 0],
                       [0.1, 1, 0],
                       [0.004, 0.002, 1]])

LEFT_X = 1
RIGHT_X = 800
BOTTOM_Y = 1
TOP_Y = 600

CORNERS = (Point(np.array([LEFT_X, BOTTOM_Y])),
           Point(np.array([LEFT_X, TOP_Y])),
           Point(np.array([RIGHT_X, TOP_Y])),
           Point(np.array([RIGHT_X, BOTTOM_Y])))


def plot_border(axs, homography: np.ndarray or None = None):
    if homography is None:
        corners_h = CORNERS
    else:
        corners_h = [Point(corner.array) for corner in CORNERS]
        for i in range(len(corners_h)):
            corners_h[i].apply_homography(homography)
    for i in range(-1, len(CORNERS) - 1):
        axs[0].plot([CORNERS[i].x(), CORNERS[i + 1].x()], [CORNERS[i].y(), CORNERS[i + 1].y()], color='black')
        axs[1].plot([corners_h[i].x(), corners_h[i + 1].x()], [corners_h[i].y(), corners_h[i + 1].y()], color='black')


def plot_point(p: Point, axs, homography=None, color: str = 'green'):
    if not LEFT_X < p.x() < RIGHT_X or not BOTTOM_Y < p.y() < TOP_Y:
        return

    axs[0].scatter([p.array[0][0]], [p.array[1][0]], color=color)
    p_copy = Point(p.array)
    if homography is not None:
        p_copy.apply_homography(homography)
    axs[1].scatter([p_copy.array[0][0]], [p_copy.array[1][0]], color=color)
    plt.draw()


def plot_line(line: Line, axs, color: str = 'green', homography=None):
    borders = [Line.from_two_points(CORNERS[i], CORNERS[i - 1]) for i in range(4)]

    points_intersection = set()
    points_intersection_h = set()

    for border in borders:
        border_intersection = line.intersection(border)
        if 1 <= border_intersection.x() <= 800 + EPSILON and 1 <= border_intersection.y() <= 600 + EPSILON:
            points_intersection.add((border_intersection.x(), border_intersection.y()))
            if homography is not None:
                border_intersection.apply_homography(homography)
            points_intersection_h.add((border_intersection.x(), border_intersection.y()))

    points_intersection = list(points_intersection)
    points_intersection_h = list(points_intersection_h)

    if len(points_intersection) == 2:
        axs[0].plot(list(map(lambda x: x[0], points_intersection)), list(map(lambda x: x[1], points_intersection)),
                    color=color)

        axs[1].plot(list(map(lambda x: x[0], points_intersection_h)), list(map(lambda x: x[1], points_intersection_h)),
                    color=color)

    plt.draw()


def main():

    fig, axs = plt.subplots(1, 2)
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()

    plot_border(axs, homography=HOMOGRAPHY)

    pts = np.asarray(plt.ginput(2, timeout=-1))
    p1 = Point(pts[0])
    p2 = Point(pts[1])
    plot_point(p1, homography=HOMOGRAPHY, axs=axs)
    plot_point(p2, homography=HOMOGRAPHY, axs=axs)

    l1 = Line.from_two_points(p1, p2)
    plot_line(l1, axs, homography=HOMOGRAPHY)

    pts = np.asarray(plt.ginput(2, timeout=-1))
    p3 = Point(pts[0])
    p4 = Point(pts[1])
    plot_point(p3, color='blue', homography=HOMOGRAPHY, axs=axs)
    plot_point(p4, color='blue', homography=HOMOGRAPHY, axs=axs)

    l2 = Line.from_two_points(p3, p4)
    plot_line(l2, axs, 'blue', homography=HOMOGRAPHY)

    p_intersection = l1.intersection(l2)

    plot_point(p_intersection, homography=HOMOGRAPHY, axs=axs, color='red')

    plt.show()


if __name__ == "__main__":
    main()
