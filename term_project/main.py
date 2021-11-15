import numpy as np
import matplotlib.pyplot as plt
import sys
from image_utils import get_points_correspondences, read_points, get_k_from_file, show_inliers
from RANSAC import ransac_epipolar, cross_product_matrix, R_from_rodrigues
from toolbox import e2p

LINES_COLORS = ['red', 'green', 'blue', 'yellow', 'brown']


def plot_line(line, color: str or tuple = 'blue'):
    line /= (line[0] ** 2 + line[1] ** 2) ** 0.5
    ax = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
    plt.plot(ax, list(map(lambda x: (-line[2] - line[0] * x) / line[1] if min(plt.ylim()) < (-line[2] - line[0] * x) / line[1] < max(plt.ylim()) else None, ax)), color=color, )


def main():
    if len(sys.argv) < 2:
        image1_index = '07'
        image2_index = '11'
    else:
        image1_index = sys.argv[1].zfill(2)
        image2_index = sys.argv[2].zfill(2)

    image1_points = read_points(f'data/scene_1/corresp/u_{image1_index}.txt')
    image2_points = read_points(f'data/scene_1/corresp/u_{image2_index}.txt')
    k = get_k_from_file()

    correspondences = get_points_correspondences(f'data/scene_1/corresp/m_{image1_index}_{image2_index}.txt')

    # Find the cameras relative rotation and translation
    support, inliers, R, T, chosen = ransac_epipolar(image1_points, image2_points, correspondences, k, 2.5, 0.9999)
    # print(R, T)
    # print(support)
    # print(image1_points[:, [i for i in chosen]])
    # print(image2_points[:, [correspondences[i] for i in chosen]])

    plt.imshow(plt.imread(f'data/scene_1/images/{image2_index}.jpg'))
    show_inliers(image1_points, image2_points, inliers, correspondences)

    plt.show()

    image1_points = e2p(image1_points)
    image2_points = e2p(image2_points)

    # show lines that are a mapping of second point images on the first image
    k_inv = np.linalg.inv(k)
    F = k_inv.T @ cross_product_matrix(-T) @ R @ k_inv

    plt.imshow(plt.imread(f'data/scene_1/images/{image2_index}.jpg'))
    points_ind = 0
    for i in chosen:
        plt.plot([image2_points[0, correspondences[i]]], [image2_points[1, correspondences[i]]], color=LINES_COLORS[points_ind], marker='X')
        plot_line(F @ image1_points[:, i].reshape((3, 1)), color=LINES_COLORS[points_ind])
        points_ind += 1
    plt.show()

    plt.imshow(plt.imread(f'data/scene_1/images/{image1_index}.jpg'))
    points_ind = 0
    for i in chosen:
        plt.plot([image1_points[0, i]], [image1_points[1, i]], color=LINES_COLORS[points_ind], marker='X')
        plot_line(F.T @ image2_points[:, correspondences[i]].reshape((3, 1)), color=LINES_COLORS[points_ind])
        points_ind += 1
    plt.show()



if __name__ == '__main__':
    main()
