import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import get_points_correspondences, read_points, get_k_from_file, show_inliers
from RANSAC_epipolar import ransac_epipolar, cross_product_matrix, R_from_rodrigues
from corresp import Corresp


def main():
    if len(sys.argv) < 2:
        image1_index = 6
        image2_index = 7
    else:
        image1_index = int(sys.argv[1])
        image2_index = int(sys.argv[2])
        if image2_index < image1_index:
            print("Index of the second index should be bigger than the first one...")
            exit(-1)

    image1_index -= 1
    image2_index -= 1

    # Run for only the specified number of images
    if len(sys.argv) >= 4:
        cameras_num = int(sys.argv[3])
    else:
        cameras_num = 12

    c = Corresp(cameras_num)

    cameras_R = [None] * cameras_num
    cameras_t = [None] * cameras_num

    # Append a None at the beginning of the array to make 1-indexation easier
    images_interesting_points = []
    for i in range(1, cameras_num + 1):
        images_interesting_points.append(read_points(f'data/scene_1/corresp/u_{str(i).zfill(2)}.txt'))

        for j in range(1, i):
            correspondences = get_points_correspondences(f'data/scene_1/corresp/m_{str(j).zfill(2)}_{str(i).zfill(2)}.txt')
            c.add_pair(j - 1, i - 1, np.array(list(correspondences.items())))

    k = get_k_from_file()

    # Find the cameras relative rotation and translation
    print(images_interesting_points[image1_index])
    initial_correspondences = get_points_correspondences(f'data/scene_1/corresp/m_{str(image1_index + 1).zfill(2)}_{str(image2_index + 1).zfill(2)}.txt')
    support, inliers, R, T = ransac_epipolar(images_interesting_points[image1_index],
                                             images_interesting_points[image2_index],
                                             initial_correspondences,
                                             k,
                                             2.5,
                                             0.9999)


if __name__ == '__main__':
    main()
