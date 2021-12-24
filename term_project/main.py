import matplotlib.pyplot as plt
import sys
from utils import get_points_correspondences, read_points, get_k_from_file, get_image_image_correspondences
from RANSAC_epipolar import ransac_epipolar, cross_product_matrix
from corresp import Corresp
from RANSAC_P3P import ransac_p3p
import ge
from stereo_matching import *

POINT_REPROJECTION_ERROR_FOR_ADDITION = 1.5  # in pixels

# Camera pairs used for dense point cloud reconstruction
# TODO: find out which pairs had better been filtered
CAMERA_PAIRS_FOR_RECONSTRUCTION = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
                                   (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11)]

# Colors of points corresponding to each camera pair
POINTS_COLORS = [np.array([(i * 170) % 256, (i * 170 ** 2) % 256, (i * 170 ** 3) % 256], dtype=np.uint8)
                 for i in range(len(CAMERA_PAIRS_FOR_RECONSTRUCTION))]


def main():
    if len(sys.argv) < 2:
        image1_index = 1
        image2_index = 2
    else:
        image1_index = int(sys.argv[1])
        image2_index = int(sys.argv[2])
        if image2_index < image1_index:
            print("Index of the second index should be bigger than the first one...")
            exit(-1)

    image1_index -= 1
    image2_index -= 1

    cameras_added = [image1_index, image2_index]

    # Run for only the specified number of images
    if len(sys.argv) >= 4:
        cameras_num = int(sys.argv[3])
    else:
        cameras_num = 12

    c = Corresp(cameras_num)

    # Arrays to store cameras configuration
    cameras_R: List[np.array or None] = [None] * cameras_num
    cameras_t: List[np.array or None] = [None] * cameras_num

    # Append a None at the beginning of the array to make 1-indexation easier
    images_interesting_points = []
    for i in range(1, cameras_num + 1):
        images_interesting_points.append(read_points(f'data/scene_1/corresp/u_{str(i).zfill(2)}.txt'))

        for j in range(1, i):
            correspondences = get_points_correspondences(
                f'data/scene_1/corresp/m_{str(j).zfill(2)}_{str(i).zfill(2)}.txt')
            c.add_pair(j - 1, i - 1, np.array(list(correspondences.items())))

    k = get_k_from_file()

    # Find the cameras relative rotation and translation
    initial_correspondences = c.get_m(image1_index, image2_index)
    correspondences_dict = {initial_correspondences[0][i]: initial_correspondences[1][i] for i in
                            range(len(initial_correspondences[0]))}

    support, inliers, R, T, E = ransac_epipolar(images_interesting_points[image1_index],
                                                images_interesting_points[image2_index],
                                                correspondences_dict,
                                                k,
                                                1.5,
                                                0.9999)

    T = T / np.linalg.norm(T)

    # Update the values in the array
    cameras_t[image1_index] = np.zeros((3, 1))
    cameras_R[image1_index] = np.identity(3)
    cameras_t[image2_index] = T.reshape((3, 1))
    cameras_R[image2_index] = R

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
    points_max_index = len(correspondences_indices) + 1

    # Triangulate all the inliers correspondences
    j = 0
    P_1 = Rtk2P(cameras_R[image1_index], cameras_t[image1_index], k)
    P_2 = Rtk2P(cameras_R[image2_index], cameras_t[image2_index], k)

    for i in correspondences_indices:
        u1 = e2p(images_interesting_points[image1_index][:, initial_correspondences[0][i]].reshape((2, 1)))
        u2 = e2p(images_interesting_points[image2_index][:, initial_correspondences[1][i]].reshape((2, 1)))
        points_by_id[j] = triangulate_to_3d(u1,
                                            u2,
                                            P_1,
                                            P_2)
        j += 1

    # Start the cluster using selected cameras and found points
    c.start(image1_index, image2_index, correspondences_indices, initial_points_indices)

    while True:
        # Find out, which camera to add next. Take the one with the biggest number of points ot image correspondences
        ig = c.get_green_cameras()
        print("Cameras left: ", ig)
        if len(ig[0]) == 0:
            break

        camera_index_to_append = ig[0][np.argmax(ig[1])]
        print(f"Adding camera {camera_index_to_append}")
        cameras_added.append(camera_index_to_append)

        # Calculate the Rotation and translation of the newly added camera bases on the correspondences
        X, u, _ = c.get_Xu(camera_index_to_append)
        inliers, R, T = ransac_p3p(points_by_id, X, images_interesting_points[camera_index_to_append], u, k, 1, 0.99999)

        cameras_t[camera_index_to_append] = T
        cameras_R[camera_index_to_append] = R
        c.join_camera(camera_index_to_append, inliers)

        # Add new points to the point cloud
        camera_neighbours = c.get_cneighbours(camera_index_to_append)

        for neighbour in camera_neighbours:
            mi, mic = c.get_m(camera_index_to_append, neighbour)
            new_points = []
            inliers_indices = []

            P_1 = Rtk2P(cameras_R[camera_index_to_append], cameras_t[camera_index_to_append], k)
            P_2 = Rtk2P(cameras_R[neighbour], cameras_t[neighbour], k)

            for i in range(len(mi)):
                u1 = e2p(images_interesting_points[camera_index_to_append][:, mi[i]].reshape((2, 1)))
                u2 = e2p(images_interesting_points[neighbour][:, mic[i]].reshape((2, 1)))
                X = e2p(triangulate_to_3d(u1, u2, P_1, P_2).reshape((3, 1)))
                error = max(reprojection_error(P_1, X, u1), reprojection_error(P_2, X, u2))

                # Add point if the error is relatively small and the points seems to be an inlier
                if error < POINT_REPROJECTION_ERROR_FOR_ADDITION and (P_1 @ X)[2] > 0 and (P_2 @ X)[2] > 0:
                    new_points.append(points_max_index)
                    inliers_indices.append(i)
                    points_by_id[points_max_index] = p2e(X).flatten()
                    points_max_index += 1

            print(f"Adding {len(new_points)} point to the cloud...")
            c.new_x(camera_index_to_append, neighbour, np.array(inliers_indices), np.array(new_points))

        # Verify the correspondences by checking the reprojection error
        cluster_cameras = c.get_selected_cameras()
        for camera in cluster_cameras:
            X, u, Xu_verified = c.get_Xu(camera)
            P = Rtk2P(cameras_R[camera], cameras_t[camera], k)

            good_points_indices = []
            for i in range(len(X)):
                if Xu_verified[i]:
                    continue
                X_i = e2p(points_by_id[X[i]].reshape((3, 1)))
                u_i = e2p(images_interesting_points[camera][:, u[i]].reshape((2, 1)))
                error = reprojection_error(P, X_i, u_i)
                if error < POINT_REPROJECTION_ERROR_FOR_ADDITION:
                    good_points_indices.append(i)
            c.verify_x(camera, good_points_indices)

        c.finalize_camera()

    # Make a sparse point cloud
    cloud = np.array(list(points_by_id.values()), dtype=np.float64).T
    colors = np.zeros(cloud.shape).T

    g = ge.GePly('sparse.ply')
    g.points(cloud, colors)
    g.close()

    # Construct the dense points cloud
    points = []
    for i in range(len(CAMERA_PAIRS_FOR_RECONSTRUCTION)):
        c1, c2 = CAMERA_PAIRS_FOR_RECONSTRUCTION[i]

        image1 = plt.imread(f'data/scene_1/images/{str(c1 + 1).zfill(2)}.jpg')
        image2 = plt.imread(f'data/scene_1/images/{str(c2 + 1).zfill(2)}.jpg')

        # Get seed correspondences from inlier correspondences to make algorithm find the right correspondences quicker
        seed_correspondences = get_image_image_correspondences(images_interesting_points[c1],
                                                               images_interesting_points[c2],
                                                               c.get_Xu(c1),
                                                               c.get_Xu(c2))
        # Run the algorithm and get set of points
        new_points = get_rectified_image_task(image1,
                                              image2,
                                              cameras_R[c1],
                                              cameras_t[c1],
                                              cameras_R[c2],
                                              cameras_t[c2],
                                              k,
                                              seed_correspondences)
        points_colors = [POINTS_COLORS[i]] * len(new_points)

        # Add points to the dense points cloud (with black color)
        points.extend(new_points)

        # Save temporary result to be able to analyse each camera pair independently
        g = ge.GePly(f'{c1}_{c2}.ply')
        g.points(np.array(new_points, dtype=np.float64).T, np.array(points_colors, dtype=np.uint8).T)
        g.close()

    # Export the dense point cloud with black points
    cloud = np.array(points, dtype=np.float64).T
    colors = np.zeros(cloud.shape())
    g = ge.GePly('dense.ply')
    g.points(cloud,
             colors)
    g.close()


if __name__ == '__main__':
    main()
