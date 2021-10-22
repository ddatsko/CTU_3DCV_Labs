import numpy as np
import matplotlib.pyplot as plt
import image_utils
from RANSAC import homography_ransac
import sys


def plot_line(line, color: str or tuple = 'blue'):
    ax = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
    plt.plot(ax, list(map(lambda x: (-line[2] - line[0] * x) / line[1], ax)), color=color)


def main():
    if len(sys.argv) < 3:
        book1, book2 = 1, 2
    else:
        book1, book2 = map(int, sys.argv[1:3])

    image1_points = image_utils.read_points(f"data/books_u{book1}.txt")
    # image_utils.show_points(image1_points, "data/book1.png")

    image2_points = image_utils.read_points(f"data/books_u{book2}.txt")
    # image_utils.show_points(image2_points, "data/book2.png")

    points_correspondences = image_utils.get_points_correspondences(f"data/books_m{book1}{book2}.txt")
    ha, support_points_a, hb, support_points_b, a, not_supported = homography_ransac(image1_points, image2_points,
                                                                                     points_correspondences, 5,
                                                                                     lambda x: x, n=1000)

    # print(len(not_supported))
    image_utils.show_points(image1_points, image2_points, not_supported, points_correspondences)
    image_utils.show_homography(f'data/book{book1}.png', hb, support_points_b, color=(0.168, 1, 0))
    image_utils.show_homography(f'data/book{book1}.png', ha, support_points_a, color=(1, 0, 0))
    plot_line(a, color=(1, 0, 1))

    plt.axis([0, 968, 648, 0])

    plt.show()


if __name__ == "__main__":
    main()
