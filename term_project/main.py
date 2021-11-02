import numpy as np
import matplotlib.pyplot as plt
import sys
from image_utils import get_points_correspondences, read_points
from RANSAC import ransac_epipolar


def main():
    if len(sys.argv) < 2:
        image1_index = '05'
        image2_index = '06'
    else:
        image1_index = sys.argv[1].zfill(2)
        image2_index = sys.argv[2].zfill(2)

    image1_points = read_points(f'data/scene_1/corresp/u_{image1_index}.txt')
    image2_points = read_points(f'data/scene_1/corresp/u_{image2_index}.txt')

    correspondences = get_points_correspondences(f'data/scene_1/corresp/m_{image1_index}_{image2_index}.txt')

    ransac_epipolar(image1_points, image2_points, correspondences, 3, 0.99)



if __name__ == '__main__':
    main()
