import numpy as np


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


