import numpy as np
import matplotlib.pyplot as plt
import cv2

import image_utils.image_thresholder as it
import image_utils.region_masker as rm

from pylab import rcParams
rcParams['figure.figsize'] = 9, 5

mtx = np.load('../../camera_calibration/saved_data_to_calibrate_images/mtx.npy')
dist = np.load('../../camera_calibration/saved_data_to_calibrate_images/dist.npy')

# load in the image files using glob
file_name = 'straight_lines2.jpg'
bgr_image = cv2.imread('../../test_images/' + file_name)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
undistorted_image = cv2.undistort(rgb_image, mtx, dist, None, mtx)

binary_img = it.combined_thresh(rgb_img=undistorted_image)

region_masked_image = rm.make_region_of_interest(gray_image=binary_img)

plt.title('region masked S, R, and L channels binary image', fontsize=20)
plt.imshow(region_masked_image, cmap='gray')
# plt.show()
plt.savefig('../../images/3_thresh+reg_mask_images/' + file_name.split('.')[0] + '.png')
