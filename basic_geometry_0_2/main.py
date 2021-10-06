import numpy as np
import matplotlib.pyplot as plt
from toolbox import e2p, p2e
from math import cos, sin

X1 = np.array([[-0.5, 0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2, 0, 0.5],
               [-0.5, -0.5, 0.5, 0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1, -0.5],
               [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]])

X2 = np.array([[-0.5, 0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2, 0, 0.5],
               [-0.5, -0.5, 0.5, 0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1, -0.5],
               [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]])

K = np.array([[1000, 0, 500],
              [0, 1000, 500],
              [0, 0, 1]])


def plot(u1, v1, u2, v2):
    plt.plot(u1, v1, 'r-', linewidth=2)
    plt.plot(u2, v2, 'b-', linewidth=2)
    plt.plot([u1, u2], [v1, v2], 'k-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.axis('equal')  # this kind of plots should be isotropic
    plt.show()


def make_camera_image(rot: np.array, translation: np.array):
    x1_c = (rot @ X1) - rot @ translation

    x2_c = (rot @ X2) - rot @ translation

    x1_proj = K @ x1_c
    x2_proj = K @ x2_c

    x1_coord = p2e(x1_proj)
    x2_coord = p2e(x2_proj)

    plot(x1_coord[0], x1_coord[1], x2_coord[0], x2_coord[1])


def make_image_camera_1():
    rot = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    translation = np.array([[0], [0], [0]])

    make_camera_image(rot, translation)


def make_image_camera_2():
    rot = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    translation = np.array([[0], [-1], [0]])

    make_camera_image(rot, translation)


def make_image_camera_3():
    rot = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    translation = np.array([[0], [0.5], [0]])

    make_camera_image(rot, translation)


def make_image_camera_4():
    rot = np.array([[1, 0, 0],
                    [0, cos(0.5), -sin(0.5)],
                    [0, sin(0.5), cos(0.5)]])

    translation = np.array([[0], [-3], [0.5]])

    make_camera_image(rot, translation)


def make_image_camera_5():
    rot = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])

    translation = np.array([[0], [-5], [4.2]])

    make_camera_image(rot, translation)


def make_image_camera_6():
    rot = np.array([[1, 0, 0],
                    [0, cos(0.8), -sin(0.8)],
                    [0, sin(0.8), cos(0.8)]]) @ \
          np.array([[cos(-0.5), 0, sin(-0.5)],
                    [0, 1, 0],
                    [-sin(-0.5), 0, cos(-0.5)]])

    translation = np.array([[-1.5], [-3], [1.5]])

    make_camera_image(rot, translation)


def main():
    make_image_camera_4()


if __name__ == "__main__":
    main()
