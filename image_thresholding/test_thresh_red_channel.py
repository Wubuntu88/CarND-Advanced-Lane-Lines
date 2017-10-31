import numpy as np
import matplotlib.pyplot as plt
import cv2

import image_thresholding.image_thresholder as it

mtx = np.load('../camera_calibration/saved_data_to_calibrate_images/mtx.npy')
dist = np.load('../camera_calibration/saved_data_to_calibrate_images/dist.npy')

# load in the image files using glob
bgr_image = cv2.imread('../test_images/test3.jpg')
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

undistorted_image = cv2.undistort(rgb_image, mtx, dist, None, mtx)

binary_img = it.red_channel_sobol_thresh(rgb_img=rgb_image, thresh=(40, 250))

plt.title('red channel grayscale image', fontsize=20)
plt.imshow(binary_img, cmap='gray')
plt.show()
