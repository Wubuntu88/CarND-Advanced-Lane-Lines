import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

mtx = np.load('saved_data_to_calibrate_images/mtx.npy')
dist = np.load('saved_data_to_calibrate_images/dist.npy')

# load in the image files using glob
bgr_image = cv2.imread('../test_images/test1.jpg')
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

undistorted_image = cv2.undistort(rgb_image, mtx, dist, None, mtx)

plt.imshow(undistorted_image)
plt.show()
