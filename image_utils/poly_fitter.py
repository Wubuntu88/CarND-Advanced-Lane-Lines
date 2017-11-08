import numpy as np
import cv2
import matplotlib.pyplot as plt
import image_utils.difference_constants as dc
import image_utils.line as line
from typing import Tuple, List


def find_polynomials(grayscale_frame, left_line: line.Line, right_line: line.Line):

    def is_poly_legit(the_line: line.Line,
                      fit_x,
                      difference_mean: float, difference_stdev) -> bool:
        difference = sum(abs(the_line.previous_poly - fit_x))
        return True if difference < difference_mean + difference_stdev * 2 else False

    def update_line(gray_image, polynomial, lane_line: line.Line, diff_mean: float, diff_stdev: float):
        if polynomial is not None:
            ploty = np.linspace(0, gray_image.shape[0] - 1, gray_image.shape[0])
            fit_x = project_poly_on_range(polynomial=polynomial, ploty=ploty)
            if lane_line.previous_poly is None:  # if it is the first line, just add it
                lane_line.add_or_replace_poly(polynomial=polynomial)
                lane_line.current_poly = polynomial
                lane_line.best_poly = polynomial
                lane_line.recent_plot_y = ploty
                lane_line.recent_x_points = fit_x
            else:  # if it is not the first line
                is_legit = is_poly_legit(the_line=lane_line,
                                         fit_x=fit_x,
                                         difference_mean=diff_mean,
                                         difference_stdev=diff_stdev)
                if is_legit:  # if it is a legitimate line
                    lane_line.add_or_replace_poly(polynomial=polynomial)
                    lane_line.previous_poly = left_line.current_poly
                    lane_line.current_poly = polynomial
                    lane_line.best_poly = lane_line.current_poly
                    lane_line.recent_plot_y = ploty
                    lane_line.recent_x_points = fit_x
                else:  # if it is not a legitimate line, average the past n lines and discard the current
                    lane_line.best_poly = left_line.avg_poly()

    out_img, left_polynomial, right_polynomial, left_xy_pts_tup, right_xy_pts_tup = complex_polyfit(grayscale_frame)

    update_line(gray_image=grayscale_frame, polynomial=left_polynomial, lane_line=left_line,
                diff_mean=dc.LEFT_LANE_DIFF_MEAN, diff_stdev=dc.LEFT_LANE_DIFF_STD)
    update_line(gray_image=grayscale_frame, polynomial=right_polynomial, lane_line=right_line,
                diff_mean=dc.RIGHT_LANE_DIFF_MEAN, diff_stdev=dc.RIGHT_LANE_DIFF_STD)
    return left_xy_pts_tup, right_xy_pts_tup


def complex_polyfit(binary_warped):
    """
    This function is mostly taken from the code from Udacity in their 'Finding The Lines' section in the course.
    :return: return two polynomials
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return out_img, left_fit, right_fit, (leftx, lefty), (rightx, righty)


def project_poly_on_range(polynomial: List[float], ploty):
    assert len(polynomial) == 3
    return polynomial[0] * ploty ** 2 + polynomial[1] * ploty + polynomial[2]


def simple_polyfit(grayscale_image):
    midpoint = grayscale_image.shape[1] // 2
    end = grayscale_image.shape[1]
    left_half = grayscale_image[:, 0:midpoint]
    right_half = grayscale_image[:, midpoint:end]

    left_nonzero = np.nonzero(left_half)
    left_nonzero_y = left_nonzero[0]
    left_nonzero_x = left_nonzero[1]

    right_nonzero = np.nonzero(right_half)
    right_nonzero_y = right_nonzero[0]
    right_nonzero_x = right_nonzero[1] + midpoint

    if len(left_nonzero_y) == 0 or len(right_nonzero_x) == 0:
        return None, None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_nonzero_y, left_nonzero_x, 2)
    right_fit = np.polyfit(right_nonzero_y, right_nonzero_x, 2)

    return left_fit, right_fit
