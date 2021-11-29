import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import get_points_correspondences, read_points, get_k_from_file
from RANSAC_epipolar import ransac_epipolar, cross_product_matrix
from corresp import Corresp
from toolbox import triangulate_to_3d
from RANSAC_P3P import ransac_p3p
import ge


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
    initial_correspondences = c.get_m(image1_index, image2_index)
    correspondences_dict = {initial_correspondences[0][i]: initial_correspondences[1][i] for i in range(len(initial_correspondences[0]))}

    support, inliers, R, T, E = ransac_epipolar(images_interesting_points[image1_index],
                                             images_interesting_points[image2_index],
                                             correspondences_dict,
                                             k,
                                             1,
                                             0.9999)



    # A dict to store 3d point coordinates by their id
    points_by_id = {}

    # Find the indices of inliers in the correspondences array to feed them to the c object
    # Doing in O(n) time complexity
    correspondences_indices = []
    it = 0
    for i in inliers:
        while initial_correspondences[0][it] != i:
            it += 1
        correspondences_indices.append(it)

    initial_points_indices = np.array(list(range(len(correspondences_indices))))

    j = 0
    for i in correspondences_indices:
        points_by_id[j] = triangulate_to_3d(images_interesting_points[image1_index][:, initial_correspondences[0][i]],
                                            images_interesting_points[image2_index][:, initial_correspondences[1][i]],
                                            T,
                                            R, k)
        j += 1



    cloud = np.array(list(points_by_id.values()), dtype=np.float64)
    # colors = np.random.rand(*cloud.shape)
    colors = np.zeros(cloud.shape)

    C = -R.T @ T

    print(R)
    print(T)
    print(C)

    cloud = np.vstack([cloud, C,  np.array([0, 0, 0])]).T
    colors = np.vstack([colors, np.array([1, 1, 1]), np.array([1, 1, 1])]).T

    print(len(inliers))

    c.start(image1_index, image2_index, correspondences_indices, initial_points_indices)

    ig = c.get_green_cameras()

    camera_index_to_append = ig[0][np.argmax(ig[1])]
    print(f"Adding camera {camera_index_to_append}")

    X, u, _ = c.get_Xu(camera_index_to_append)

    ransac_p3p(points_by_id, X, images_interesting_points[camera_index_to_append], u, k, 2, 0.99)




if __name__ == '__main__':
    main()
