import sys

import cv2
import numpy as np

from processing.rectification_funcs import *
from processing.RANSAC import get_codirectional_lines


def find_lines(image, threshold, dots_per_line):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    median = cv2.medianBlur(dilated, 5)
    diff2 = 255 - cv2.subtract(median, gray)
    normed = cv2.normalize(diff2,None, 10, 255, cv2.NORM_MINMAX )
    bw = cv2.threshold(normed, threshold, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(bw, 200, 120, apertureSize=3, L2gradient=True)
    
    lines = cv2.HoughLines(edges, 1, np.pi/360, dots_per_line,)
    lines_coefs = np.empty((len(lines), 3))
    coef = max(image.shape)
    for i, line in enumerate(lines):
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + coef*(-b))
            y1 = int(y0 + coef*(a))
            x2 = int(x0 - coef*(-b))
            y2 = int(y0 - coef*(a))
            lines_coefs[i] = np.cross((x1, y1, 1), (x2, y2, 1))
    return lines_coefs



if __name__ == "__main__":
    
    path_to_image = sys.argv[-4]
    threshold = int(sys.argv[-3])
    dots_per_line = int(sys.argv[-2])
    expected_angle_between_lines = int(sys.argv[-1])
    
    image = cv2.imread(path_to_image)
    lines = find_lines(image, threshold, dots_per_line)
    
    # apply RANSAC to get codirectional lines
    print(f"found lines: {len(lines)}")  
    line_group1, mask, v_point1 = get_codirectional_lines(lines, iters=1000, epsilon=0.005)
    print(f"lines in group 1: {len(line_group1)}")
    line_group2, _, v_point2 = get_codirectional_lines(lines[~mask], iters=1000, epsilon=0.005)
    print(f"lines in group 2: {len(line_group2)}")
    
    # find vanishing points corespond to +Ox and +Oy axises
    hor_point, vert_point = find_axis_points(v_point1, v_point2)
    # point order must be changed
    homography = find_homography(vert_point, hor_point, expected_angle_between_lines, shift=True, im_shape=image.shape[:2])

    img_shape=(image.shape[1], image.shape[0])
    img_corners = np.array([[0, 0, 1], [0, img_shape[1], 1], [*img_shape, 1], [img_shape[0], 0, 1]])
    img_corners_transformed = transform_points(img_corners, homography)

    # find bounding box and its shape
    min_x = np.min(img_corners_transformed[:, 0])
    max_x = np.max(img_corners_transformed[:, 0])
    min_y = np.min(img_corners_transformed[:, 1])
    max_y = np.max(img_corners_transformed[:, 1])
    bounding_box = np.array([
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y]
    ])
    new_image_shape = (int(max_x - min_x), int(max_y - min_y))

    # move bounding box to the left upper corner
    srcpts = np.float32([bounding_box])
    destpts = np.float32([[0, 0], [0, new_image_shape[1]], new_image_shape, [new_image_shape[0], 0]])
    resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
    # find full transformation homography
    full_homography = resmatrix.dot(homography)

    # transform and save image
    img_new_1 = cv2.warpPerspective(image, full_homography, new_image_shape)
    cv2.imwrite('./images/results/rectified_img.png', img_new_1)
    print('The first result saved to images/results/rectified_img.png')

    # additional step. crop image by most extreme lines
    corners = find_corner_points(lines1=line_group1, lines2=line_group2, vp1=v_point1, vp2=v_point2)
    sheet_corners = transform_points(corners, full_homography)

    # the following step is only for oblique lines in order to catch a square area
    if expected_angle_between_lines != 90:
        x_left = min(sheet_corners[0][0], sheet_corners[1][0])
        y_bottom = max(sheet_corners[1][1], sheet_corners[2][1])
        x_right = max(sheet_corners[2][0], sheet_corners[3][0])
        y_top = min(sheet_corners[3][1], sheet_corners[0][1])
        sheet_corners = np.array(
        [
            [x_left, y_top],
            [x_left, y_bottom],
            [x_right, y_bottom],
            [x_right, y_top]
        ])
        new_image_shape = (int(x_right - x_left), int(y_bottom - y_top))

    # get homography, transform and save image
    srcpts = np.float32([sheet_corners])
    destpts = np.float32([[0, 0], [0, new_image_shape[1]], new_image_shape, [new_image_shape[0], 0]])
    resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
    img_new_2 = cv2.warpPerspective(img_new_1, resmatrix, new_image_shape)
    cv2.imwrite('./images/results/rectified_cropped_img.png', img_new_2)
    print('The second result saved to images/results/rectified_cropped_img.png')
    