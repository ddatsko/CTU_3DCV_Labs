import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import get_points_correspondences, read_points, get_k_from_file
from RANSAC_epipolar import ransac_epipolar, cross_product_matrix
from corresp import Corresp
from toolbox import triangulate_to_3d, reprojection_error, e2p, p2e, Rtk2P
from RANSAC_P3P import ransac_p3p
import ge
from typing import List


POINT_REPROJECTION_ERROR_FOR_ADDITION = 2   # in pixels


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
                                             2,
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

    cameras_checked = 2

    while True:

        # if cameras_checked >= 5:
        #     break
        cameras_checked += 1
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

        # Add new point to the points cloud
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

        # TODO: mark unverified inliers as outliers to prevent printing them to the output file
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

    # Make a cloud point
    # cloud = np.array([p for p in points_by_id.values() if p[2] < 100], dtype=np.float64)
    cloud = np.array(list(points_by_id.values()), dtype=np.float64)
    colors = np.zeros(cloud.shape)

    # Add cameras to the points cloud and mark them with white color
    for i in range(len(cameras_R)):
        if cameras_R[i] is not None:
            C = (-cameras_R[i].T @ cameras_t[i]).flatten()
            cloud = np.vstack([cloud, C])
            colors = np.vstack([colors, np.array([1, 1, 1])])

    cloud = cloud.T
    colors = colors.T
    g = ge.GePly('out.ply')
    g.points(cloud,
             colors)
    g.close()

    # Plot cameras in matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")

    for i in cameras_added:
        # R @ X + t = 0, 0, 1
        # X = R.T ((0, 0, 1) - t)
        R, t = cameras_R[i], cameras_t[i]
        camera_origin = (-R.T @ t).flatten()
        camera_z = (R.T @ (np.array([[0], [0], [1]]) - t)).flatten()

        ax.plot([camera_origin[0], camera_z[0]], [camera_origin[1], camera_z[1]], [camera_origin[2], camera_z[2]], color='black')
        ax.text(camera_origin[0], camera_origin[1], camera_origin[2], f"{cameras_added.index(i)}", size=10, color="r")

    for i in range(cameras_num):
        for j in range(i + 1, cameras_num):
            if j - i == 1 and j // 4 == i // 4 or j - i == 4:
                camera1_c = (-cameras_R[i].T @ cameras_t[i]).flatten()
                camera2_c = (-cameras_R[j].T @ cameras_t[j]).flatten()
                ax.plot([camera1_c[0], camera2_c[0]],
                        [camera1_c[1], camera2_c[1]],
                        [camera1_c[2], camera2_c[2]],
                        color=('blue' if i != image1_index or j != image2_index else 'red'))

    plt.show()






if __name__ == '__main__':
    main()
