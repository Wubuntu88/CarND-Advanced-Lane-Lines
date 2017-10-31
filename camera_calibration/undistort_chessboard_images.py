import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

mtx = np.load('saved_data_to_calibrate_images/mtx.npy')
dist = np.load('saved_data_to_calibrate_images/dist.npy')

# load in the image files using glob
image_file_paths = glob.glob('camera_calibration_src_images/calibration*.jpg')

chessboard_dimensions = (9, 6)

for path_to_file in image_file_paths:
    bgr_image = cv2.imread(path_to_file)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    undistorted_image = cv2.undistort(rgb_image, mtx, dist, None, mtx)
    file_name = path_to_file.split('/')[-1]
    plt.imsave('camera_calibration_calibrated_images/' + file_name, undistorted_image)