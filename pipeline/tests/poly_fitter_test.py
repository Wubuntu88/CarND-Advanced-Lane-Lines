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

file_name = 'test6.jpg'
image = mpimg.imread('../../test_images/' + file_name)

img_proc = ip.ImageProcessor(calibration_matrix=mtx,
                             distortion_coefficients=dist)

new_img = img_proc.process_image(rgb_image=image)

left_poly, right_poly = pf.simple_polyfit(grayscale_image=new_img)
print(left_poly)
print(right_poly)

# Generate x and y values for plotting
ploty = np.linspace(0, new_img.shape[0]-1, new_img.shape[0] )
left_fitx = left_poly[0]*ploty**2 + left_poly[1]*ploty + left_poly[2]
right_fitx = right_poly[0]*ploty**2 + right_poly[1]*ploty + right_poly[2]

# plt.title('Perspective transform with S, R, and L channels and region masking\nfilename: {}'.format(file_name), fontsize=16)
plt.imshow(new_img, cmap='gray')
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.show()
# plt.savefig('../../images/4_perspec_trans_rgb/' + file_name.split('.')[0] + '.png')
