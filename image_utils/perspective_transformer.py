import numpy as np
import cv2


def make_perpective_transform(gray_image, src_pts, dst_pts):
    src_pts = np.float32(src_pts[0])
    dst_pts = np.float32(dst_pts[0])
    M = cv2.getPerspectiveTransform(src=src_pts, dst=dst_pts)
    img_size = (gray_image.shape[1], gray_image.shape[0])
    warped_image = cv2.warpPerspective(gray_image, M, img_size)
    return warped_image

def make_rgb_perspective_transform(rgb_img, src_pts, dst_pts):
    src_pts = np.float32(src_pts[0])
    dst_pts = np.float32(dst_pts[0])
    img_size = rgb_img.shape[1::-1]
    M = cv2.getPerspectiveTransform(src=src_pts, dst=dst_pts)
    warped_image = cv2.warpPerspective(rgb_img, M, img_size)
    return warped_image


def source_points(gray_image):
    x_size = gray_image.shape[1]
    y_size = gray_image.shape[0]
    src = np.array(
        [
            [
                [(x_size / 2) - 55, y_size / 2 + 100],
                [((x_size / 6) - 10), y_size],
                [(x_size * 5 / 6) + 60, y_size],
                [(x_size / 2 + 55), y_size / 2 + 100]
            ]
        ], dtype=np.int32)
    return src

# def source_points(gray_image):
#     x_size = gray_image.shape[1]
#     y_size = gray_image.shape[0]
#     src = np.array(
#         [
#             [
#                 [(x_size / 2) - 55, y_size / 2 + 100],
#                 [((x_size / 6) - 10), y_size],
#                 [(x_size * 5 / 6) + 60, y_size],
#                 [(x_size / 2 + 55), y_size / 2 + 100]
#             ]
#         ], dtype=np.int32)
#     return src

def destination_points(gray_image):
    x_size = gray_image.shape[1]
    y_size = gray_image.shape[0]
    dst = np.array(
        [
            [
                [(x_size / 4), 0],
                [(x_size / 4), y_size],
                [(x_size * 3 / 4), y_size],
                [(x_size * 3 / 4), 0]
            ]
        ], dtype=np.int32)
    return dst


def region_mask_points(gray_image):
    x_size = gray_image.shape[1]
    y_size = gray_image.shape[0]
    bottom_left = (x_size * 0.05, y_size)
    top_left = (x_size * .45, y_size * .6)
    top_right = (x_size * .55, y_size * .6)
    bottom_right = (x_size * 0.95, y_size)
    src_points = np.array(
        [
            [
                bottom_left,
                top_left,
                top_right,
                bottom_right

            ]

        ],
        dtype=np.int32
    )
    return src_points

# def destination_points(gray_image):
#     x_size = gray_image.shape[1]
#     y_size = gray_image.shape[0]
#     bottom_left = (x_size * 0.1, y_size)
#     top_left = (x_size * .1, 0.0)
#     top_right = (x_size * .9, y_size)
#     bottom_right = (x_size * 0.9, 0.0)
#     dest_points = np.array(
#         [
#             [
#                 bottom_left,
#                 top_left,
#                 bottom_right,
#                 top_right
#             ]
#         ],
#         dtype=np.int32
#     )
#     return dest_points

# def destination_points(gray_image):
#     x_size = gray_image.shape[1]
#     y_size = gray_image.shape[0]
#     bottom_left = (x_size * 0.25, y_size)
#     top_left = (x_size * .25, 0.0)
#     top_right = (x_size * .75, y_size)
#     bottom_right = (x_size * 0.75, 0.0)
#     dest_points = np.array(
#         [
#             [
#                 bottom_left,
#                 top_left,
#                 bottom_right,
#                 top_right
#             ]
#         ],
#         dtype=np.int32
#     )
#     return dest_points