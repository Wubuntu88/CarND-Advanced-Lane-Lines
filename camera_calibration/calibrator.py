import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
This class performs camera calibration using all of the images located in the 'camera_calibration_src_images' folder.
Much of the code here was inspired by the 'calibrating Your Camera' Video in Udacity's Advanced lane line project.
'''

# load in the image files using glob
image_file_paths = glob.glob('camera_calibration_src_images/calibration*.jpg')

'''
The object and image points will both be 'arrays of arrays'.
Each array in the array will consist of the chessboard corners for a given image.
The image points array will hold the chessboard corners for the undistorted images.
The object points array will hold the points where we want the points to end up.
-Note: Each inner array in the objects larger array will be identical.

'''
image_points = []  # 2D points in the image plane (from the source image)
object_points = []  # 3D points in real world space (from the destination image)

chessboard_dimensions = (9, 6)

obj_pt_grid = np.zeros((9 * 6, 3), np.float32)
obj_pt_grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for path_to_file in image_file_paths:
    bgr_image = cv2.imread(path_to_file)

    # convert the image to grayscale in order to find the chessboard corners
    grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(image=grayscale_image, patternSize=chessboard_dimensions)

    if ret:  # if the chessboard corners were successfully found
        image_points.append(corners)
        object_points.append(obj_pt_grid)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, grayscale_image.shape[::-1], None, None)

np.save(file='saved_data_to_calibrate_images/mtx.npy', arr=mtx)
np.save(file='saved_data_to_calibrate_images/dist.npy', arr=dist)
