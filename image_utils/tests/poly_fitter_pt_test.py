import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import image_utils.poly_fitter as pf
import image_utils.perspective_transformer as pt
import image_utils.line as line

from pylab import rcParams


rcParams['figure.figsize'] = 9, 5

import image_utils.image_thresholder as it
import image_utils.region_masker as rm


mtx = np.load('../../camera_calibration/saved_data_to_calibrate_images/mtx.npy')
dist = np.load('../../camera_calibration/saved_data_to_calibrate_images/dist.npy')

file_name = 'straight_lines1.jpg'
image = mpimg.imread('../../test_images/' + file_name)

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
source_points = pt.source_points(gray_image)
destination_points = pt.destination_points(gray_image)
undistorted_rgb_image = cv2.undistort(image, mtx, dist,
                                      None, mtx)
binary_img = it.combined_thresh(rgb_img=undistorted_rgb_image)
region_masked_image = rm.make_region_of_interest(gray_image=binary_img)

warped_image, M, Minv = pt.make_perspective_transform(gray_image=region_masked_image,
                                                              src_pts=source_points,
                                                              dst_pts=destination_points)
left_line = line.Line()
right_line = line.Line()
pf.find_polynomials(grayscale_frame=warped_image, left_line=left_line, right_line=right_line)

left_poly = left_line.current_poly
right_poly = right_line.current_poly

# Generate x and y values for plotting
ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
left_fitx = left_poly[0]*ploty**2 + left_poly[1]*ploty + left_poly[2]
right_fitx = right_poly[0]*ploty**2 + right_poly[1]*ploty + right_poly[2]

plt.title('Perspective transform with fitted polynomials\nfilename: {}'.format(file_name), fontsize=16)
plt.imshow(warped_image, cmap='gray')
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.show()
# plt.savefig('../../images/4_perspec_trans_rgb/' + file_name.split('.')[0] + '.png')
