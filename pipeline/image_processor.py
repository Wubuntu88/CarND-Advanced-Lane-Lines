import numpy as np
import cv2

import image_utils.image_thresholder as it
import image_utils.region_masker as rm
import image_utils.perspective_transformer as pt
import image_utils.line as line
import image_utils.poly_fitter as pf


class ImageProcessor:

    def __init__(self, calibration_matrix=None, distortion_coefficients=None):
        self.calibration_matrix = calibration_matrix
        self.distortion_coefficients = distortion_coefficients
        self.source_points = None  # pt.source_points()
        self.destination_points = None  # pt.destination_points()
        self.left_line = line.Line()
        self.right_line = line.Line()

    def process_image(self, rgb_image):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        if self.source_points is None or self.destination_points is None:
            self.source_points = pt.source_points(gray_image)
            self.destination_points = pt.destination_points(gray_image)
        undistorted_rgb_image = cv2.undistort(rgb_image, self.calibration_matrix, self.distortion_coefficients,
                                              None, self.calibration_matrix)
        binary_img = it.combined_thresh(rgb_img=undistorted_rgb_image)
        region_masked_image = rm.make_region_of_interest(gray_image=binary_img)
        warped_image, M, Minv = pt.make_perspective_transform(gray_image=region_masked_image,
                                                              src_pts=self.source_points,
                                                              dst_pts=self.destination_points)
        pf.find_polynomials(grayscale_frame=warped_image, left_line=self.left_line, right_line=self.right_line)
        color_warped = ImageProcessor.create_color_masked_region_image(warped_grayscale_image=warped_image,
                                                                       left_line=self.left_line,
                                                                       right_line=self.right_line)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        unwarped_color = cv2.warpPerspective(color_warped, Minv, (color_warped.shape[1], color_warped.shape[0]))
        # Combine the result with the original image
        result_image = cv2.addWeighted(undistorted_rgb_image, 1, unwarped_color, 0.3, 0)

        self.draw_radii_of_curvature(image=result_image)

        self.draw_offset_from_center(image=result_image)

        return result_image

    # def process_image_make_rgb(self, rgb_image):
    #     gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    #     if self.source_points is None or self.destination_points is None:
    #         self.source_points = pt.source_points(gray_image)
    #         self.destination_points = pt.destination_points(gray_image)
    #     undistorted_rgb_image = cv2.undistort(rgb_image, self.calibration_matrix, self.distortion_coefficients,
    #                                           None, self.calibration_matrix)
    #     warped_image, M, Minv = pt.make_rgb_perspective_transform(rgb_img=undistorted_rgb_image,
    #                                                               src_pts=self.source_points,
    #                                                               dst_pts=self.destination_points)
    #     # unfinished (not needed)
    #     return warped_image

    @staticmethod
    def create_color_masked_region_image(warped_grayscale_image, left_line: line.Line, right_line: line.Line):
        """
        Much of the code here was taken from the udacity course's 'Tips and Tricks for the Project'
        :param warped_grayscale_image:
        :param left_line:
        :param right_line:
        :return:
        """
        left_polynomial = left_line.best_poly
        right_polynomial = right_line.best_poly

        ploty = np.linspace(0, warped_grayscale_image.shape[0] - 1, warped_grayscale_image.shape[0])
        left_fitx = left_polynomial[0] * ploty ** 2 + left_polynomial[1] * ploty + left_polynomial[2]
        right_fitx = right_polynomial[0] * ploty ** 2 + right_polynomial[1] * ploty + right_polynomial[2]

        ploty = np.int32(ploty)
        left_fitx = np.int32(left_fitx)
        right_fitx = np.int32(right_fitx)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_grayscale_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # drawing lines
        for i in range(1, len(ploty)):
            cv2.line(color_warp, (left_fitx[i - 1], ploty[i - 1]), (left_fitx[i], ploty[i]), color=(255, 255, 0),
                     thickness=10)
            cv2.line(color_warp, (right_fitx[i - 1], ploty[i - 1]), (right_fitx[i], ploty[i]), color=(255, 255, 0),
                     thickness=10)
        return color_warp

    def draw_radii_of_curvature(self, image):
        bottom_y_pixel = image.shape[0] - 1

        left_roc = self.left_line.radius_of_curvature(y_eval=bottom_y_pixel)
        right_roc = self.right_line.radius_of_curvature(y_eval=bottom_y_pixel)

        p1 = pf.project_poly_on_range(polynomial=self.left_line.current_poly, ploty=np.array([bottom_y_pixel]))[0]
        p2 = pf.project_poly_on_range(polynomial=self.right_line.current_poly, ploty=np.array([bottom_y_pixel]))[0]

        roc_text = 'Radius of Curvature -> Left: {:10.3f}, Right: {:10.3f} (meters)'.format(left_roc, right_roc)

        cv2.putText(img=image,
                    text=roc_text,
                    org=(30, 30),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    lineType=1)

    def draw_offset_from_center(self, image):
        bottom_y_pixel = image.shape[0] - 1
        p1 = pf.project_poly_on_range(polynomial=self.left_line.current_poly, ploty=np.array([bottom_y_pixel]))[0]
        p2 = pf.project_poly_on_range(polynomial=self.right_line.current_poly, ploty=np.array([bottom_y_pixel]))[0]

        camera_position = image.shape[1] / 2
        midpoint_in_pixels = (p1 + p2) / 2.0
        offset_from_center_pixels = abs(midpoint_in_pixels - camera_position)
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        offset_from_center_meters = offset_from_center_pixels * xm_per_pix

        offset_text = 'Offset from center: {:10.3f} (meters)'.format(offset_from_center_meters)

        cv2.putText(img=image,
                    text=offset_text,
                    org=(30, 60),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    lineType=1)
