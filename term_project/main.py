import numpy as np
import matplotlib.pyplot as plt
import sys
from image_utils import get_points_correspondences, read_points, get_k_from_file, show_inliers
from RANSAC import ransac_epipolar


def main():
    if len(sys.argv) < 2:
        image1_index = '06'
        image2_index = '10'
    else:
        image1_index = sys.argv[1].zfill(2)
        image2_index = sys.argv[2].zfill(2)

    image1_points = read_points(f'data/scene_1/corresp/u_{image1_index}.txt')
    image2_points = read_points(f'data/scene_1/corresp/u_{image2_index}.txt')
    k = get_k_from_file()

    correspondences = get_points_correspondences(f'data/scene_1/corresp/m_{image1_index}_{image2_index}.txt')

    support, inliers, R, T = ransac_epipolar(image1_points, image2_points, correspondences, k, 0.0005, 0.99)

    plt.imshow(plt.imread(f'data/scene_1/images/{image1_index}.jpg'))
    print(len(correspondences.keys()))
    print(len(inliers))
    show_inliers(image1_points, image2_points, inliers, correspondences)


    plt.show()





if __name__ == '__main__':
    main()
