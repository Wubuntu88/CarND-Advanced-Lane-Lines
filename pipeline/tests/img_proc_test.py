import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from pylab import rcParams
rcParams['figure.figsize'] = 9, 5

import image_utils.image_thresholder as it
import image_utils.region_masker as rm

import pipeline.image_processor as ip

mtx = np.load('../../camera_calibration/saved_data_to_calibrate_images/mtx.npy')
dist = np.load('../../camera_calibration/saved_data_to_calibrate_images/dist.npy')

file_name = 'test6.jpg'
image = mpimg.imread('../../test_images/' + file_name)
# image = mpimg.imread('../../test_images/test5.jpg')

img_proc = ip.ImageProcessor(calibration_matrix=mtx,
                             distortion_coefficients=dist)

new_img = img_proc.process_image(rgb_image=image)

plt.title('Perspective transform with S, R, and L channels and region masking\nfilename: {}'.format(file_name), fontsize=16)
plt.imshow(new_img, cmap='gray')
# plt.show()
# plt.savefig('../../images/4_perspec_trans_rgb/' + file_name.split('.')[0] + '.png')
