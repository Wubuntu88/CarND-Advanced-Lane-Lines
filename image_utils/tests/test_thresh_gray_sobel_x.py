import numpy as np
import matplotlib.pyplot as plt
import cv2

import image_utils.image_thresholder as it

mtx = np.load('../../camera_calibration/saved_data_to_calibrate_images/mtx.npy')
dist = np.load('../../camera_calibration/saved_data_to_calibrate_images/dist.npy')

# load in the image files using glob
src_filename = 'straight_lines1.jpg'
bgr_image = cv2.imread('../../test_images/' + src_filename)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

undistorted_image = cv2.undistort(rgb_image, mtx, dist, None, mtx)

kernel = 9

binary_img = it.sobel_gray_thresh(rgb_img=undistorted_image, sobel_kernel=kernel, thresh_min=10, thresh_max=100)

plt.title('Sobel x binary image; kernel={}; filename: {}'.format(kernel, src_filename), fontsize=20)
plt.imshow(binary_img, cmap='gray')
plt.show()
