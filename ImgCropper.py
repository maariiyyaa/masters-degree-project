import cv2
from numpy import(
    seterr,
    pi,
    empty,
    cross,
    cos,
    sin,
    concatenate,
    float32
)
from numpy.linalg import norm
import matplotlib.pyplot as plt

from RANSAC import get_inliers



class ImgCropper:
    
    def __init__(self, warning=False):
        if not warning:
            seterr(invalid='ignore')
        self.edges = None
        self.sheet_corners = []
        self.cropped_shape = None
        

    def _find_edges(self, image,):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dilated = cv2.morphologyEx(
            gray, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        )
        diff2 = 255 - cv2.subtract(cv2.medianBlur(dilated, 5), gray)
        normed = cv2.normalize(diff2, None, 10, 255, cv2.NORM_MINMAX)
        bw = cv2.threshold(normed, 210, 255, cv2.THRESH_BINARY)[1]
        self.edges = cv2.Canny(bw, 100, 120, apertureSize=3, L2gradient=True)
      
       
    @staticmethod
    def _find_lines(edges):
        max_axis = max(edges.shape)
        min_axis = min(edges.shape)
        lines = cv2.HoughLines(edges, 1, pi/360, int(min_axis/5),) 
        lines_coefs = empty((len(lines), 3))
        for i, line in enumerate(lines):
            for rho, theta in line:
                a = cos(theta)
                b = sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + max_axis*(-b))
                y1 = int(y0 + max_axis*(a))
                x2 = int(x0 - max_axis*(-b))
                y2 = int(y0 - max_axis*(a))
                lines_coefs[i] = cross((x1, y1, 1), (x2, y2, 1))
        return lines_coefs


    def _find_corner_points(self, lines):
        l1 = lines[(lines[:, 1]/lines[:, 2]).argmin()] #y-intercept - top
        l2 = lines[(lines[:, 0]/lines[:, 2]).argmin()] #x-intercept - left
        l3 = lines[(lines[:, 0]/lines[:, 2]).argmax()] #x-intercept - bottom
        l4 = lines[(lines[:, 1]/lines[:, 2]).argmax()] #y-intercept - right
        corner_points = cross([l1, l2, l3, l4], [l2, l3, l4, l1])
        self.sheet_corners = (corner_points / corner_points[:, 2][:, None])[:, :2]

    
    @staticmethod
    def _get_ratio(sheet_corners):
        #get frame ratio
        length_list = empty(len(sheet_corners))
        for i, point in enumerate(sheet_corners[:-1]):
            length_list[i] = (norm(point - sheet_corners[i+1]))
        length_list[i+1] = (norm(sheet_corners[-1] - sheet_corners[0]))
        x_length = max(length_list[1], length_list[3])
        y_length = max(length_list[0], length_list[2])
        return int(x_length), int(y_length)


    def crop_image(self, image, resize=True):
        self._find_edges(image,)
        lines = ImgCropper._find_lines(self.edges)
        #find sheet borders
        line_group1, mask = get_inliers(lines, iters=800, epsilon=0.05)
        line_group2, _ = get_inliers(lines[~mask], iters=800, epsilon=0.05)
        selected_lines = concatenate((line_group1, line_group2), axis=0)
        self._find_corner_points(selected_lines)
        #get shape of cropped image
        if resize:
            self.cropped_shape = ImgCropper._get_ratio(self.sheet_corners)
        else:
            self.cropped_shape = image.shape[1], image.shape[0]
        srcpts = float32([self.sheet_corners])
        destpts = float32([[0, 0], [0, self.cropped_shape[1]], self.cropped_shape, [self.cropped_shape[0], 0]])
        resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
        return cv2.warpPerspective(image, resmatrix, self.cropped_shape) 
    