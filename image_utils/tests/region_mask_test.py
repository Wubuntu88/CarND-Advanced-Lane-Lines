import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import image_utils.region_masker as rm

file_name = 'test6.jpg'
image = mpimg.imread('../../test_images/' + file_name)
print('This image is: ', type(image), 'with dimensions:', image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


masked_edges = rm.make_region_of_interest(gray_image=gray)

# Display the image
plt.title('Region Masking of Greyscale Image\nfilename: {}'.format(file_name), fontsize=20)
plt.imshow(masked_edges, cmap='gray')
plt.show()
