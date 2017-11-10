import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import image_utils.poly_fitter as pf

from pylab import rcParams


rcParams['figure.figsize'] = 9, 5

import image_utils.image_thresholder as it
import image_utils.region_masker as rm

import pipeline.image_processor as ip

mtx = np.load('../../camera_calibration/saved_data_to_calibrate_images/mtx.npy')
dist = np.load('../../camera_calibration/saved_data_to_calibrate_images/dist.npy')

file_name = 'straight_lines1.jpg'
image = mpimg.imread('../../test_images/' + file_name)

img_proc = ip.ImageProcessor(calibration_matrix=mtx,
                             distortion_coefficients=dist)

new_img = img_proc.process_image(rgb_image=image)

plt.title('Full Pipeline Transformation\nfilename: {}'.format(file_name), fontsize=16)
plt.imshow(new_img)
plt.show()
# plt.savefig('../../images/4_perspec_trans_rgb/' + file_name.split('.')[0] + '.png')
