import numpy as np
import cv2


def gray_binary_thresh(rgb_image, thresh_min, thresh_max):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh_min) & (gray <= thresh_max)] = 1
    return binary


def sobel_gray_thresh(rgb_img, thresh_min, thresh_max, orient='x', sobel_kernel=3):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobel_x = np.absolute(sobel_x)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sx_binary


def red_channel_sobol_thresh(rgb_img, thresh_min, thresh_max):
    red_channel = rgb_img[:, :, 0]

    sobel_x = cv2.Sobel(red_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobel_x = np.absolute(sobel_x)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

    binary_img = np.zeros_like(red_channel)
    binary_img[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_img


def red_channel_thresh(rgb_img, thresh_min, thresh_max):
    R = rgb_img[:, :, 0]

    binary_img = np.zeros_like(R)
    binary_img[(R > thresh_min) & (R <= thresh_max)] = 1
    return binary_img


def s_channel_thresh(rgb_img, thresh_min, thresh_max):
    HLS = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    S = HLS[:, :, 2]

    binary_img = np.zeros_like(S)
    binary_img[(S > thresh_min) & (S <= thresh_max)] = 1
    return binary_img


def l_channel_thresh(rgb_img, thresh_min, thresh_max):
    HLS = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    L = HLS[:, :, 1]

    binary_img = np.zeros_like(L)
    binary_img[(L > thresh_min) & (L <= thresh_max)] = 1
    return binary_img


def combined_thresh(rgb_img):
    red_binary = red_channel_thresh(rgb_img, thresh_min=200, thresh_max=255)
    s_binary = s_channel_thresh(rgb_img, thresh_min=110, thresh_max=255)
    l_binary = l_channel_thresh(rgb_img, thresh_min=110, thresh_max=255)
    gray_binary = gray_binary_thresh(rgb_img, thresh_min=200, thresh_max=255)

    return s_binary | red_binary | l_binary | gray_binary