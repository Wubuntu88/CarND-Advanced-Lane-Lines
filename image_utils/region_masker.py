import cv2
import numpy as np
import image_utils.perspective_transformer as pt


def make_region_of_interest(gray_image):
    vertices = pt.region_mask_points(gray_image=gray_image)
    mask = np.zeros_like(gray_image)
    ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(gray_image, mask)
    return masked_edges