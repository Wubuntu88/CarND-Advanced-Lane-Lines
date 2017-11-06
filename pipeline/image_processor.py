import numpy as np
import matplotlib.pyplot as plt
import cv2

import image_utils.image_thresholder as it
import image_utils.region_masker as rm
import image_utils.perspective_transformer as pt


class ImageProcessor:

    def __init__(self, calibration_matrix=None, distortion_coefficients=None):
        self.calibration_matrix = calibration_matrix
        self.distortion_coefficients = distortion_coefficients
        self.source_points = None  # pt.source_points()
        self.destination_points = None  # pt.destination_points()

    def process_image(self, rgb_image):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        if self.source_points is None or self.destination_points is None:
            self.source_points = pt.source_points(gray_image)
            self.destination_points = pt.destination_points(gray_image)
        undistorted_rgb_image = cv2.undistort(rgb_image, self.calibration_matrix, self.distortion_coefficients,
                                              None, self.calibration_matrix)
        binary_img = it.combined_thresh(rgb_img=undistorted_rgb_image)
        region_masked_image = binary_img  # rm.make_region_of_interest(gray_image=binary_img)
        warped_image = pt.make_perpective_transform(gray_image=region_masked_image,
                                                    src_pts=self.source_points,
                                                    dst_pts=self.destination_points)
        return warped_image

    def process_image_make_rgb(self, rgb_image):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        if self.source_points is None or self.destination_points is None:
            self.source_points = pt.source_points(gray_image)
            self.destination_points = pt.destination_points(gray_image)
        undistorted_rgb_image = cv2.undistort(rgb_image, self.calibration_matrix, self.distortion_coefficients,
                                              None, self.calibration_matrix)
        warped_image = pt.make_rgb_perspective_transform(rgb_img=undistorted_rgb_image,
                                                         src_pts=self.source_points,
                                                         dst_pts=self.destination_points)
        return warped_image
