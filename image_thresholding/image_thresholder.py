import numpy as np
import cv2


def sobel_gray_thresh(rgb_img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobel_x = np.absolute(sobel_x)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sx_binary


def red_channel_sobol_thresh(rgb_img, thresh=(0, 255)):
    red_channel = rgb_img[:, :, 0]

    sobel_x = cv2.Sobel(red_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobel_x = np.absolute(sobel_x)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_img = np.zeros_like(red_channel)
    binary_img[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_img


def red_channel_thresh(rgb_img, thresh=(0, 255)):
    red_channel = rgb_img[:, :, 0]

    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_img = np.zeros_like(red_channel)
    binary_img[(red_channel >= thresh_min) & (red_channel <= thresh_max)] = 1
    return binary_img
