import cv2
from numpy import(
    seterr,
    pi,
    empty,
    cross,
    cos,
    sin,
    float32
)
from numpy.linalg import norm

from processing.RANSAC import get_codirectional_lines



class ImgCropper:
    
    def __init__(self, warning=False):
        if not warning:
            seterr(invalid='ignore')
        self.sheet_corners = []
        self.cropped_shape = None
        
    @staticmethod
    def get_binary_img(image,):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dilated = cv2.morphologyEx(
            gray, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        )
        diff2 = 255 - cv2.subtract(cv2.medianBlur(dilated, 5), gray)
        normed = cv2.normalize(diff2, None, 10, 255, cv2.NORM_MINMAX)
        return cv2.threshold(normed, 210, 255, cv2.THRESH_BINARY)[1]
         
      
       
    @staticmethod
    def _find_lines(edges, hl_threshold):
        max_axis = max(edges.shape)
        min_axis = min(edges.shape)
        lines = cv2.HoughLines(edges, 1, pi/360, hl_threshold,) 
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


    def _find_corner_points(self, lines1, vp1, lines2):
        if vp1[2] != 0:
            vp1 = vp1/vp1[2]
        if abs(vp1[0]/vp1[1]) < 1:
            vertical = lines1
            horizontal = lines2
        else:
            vertical = lines2
            horizontal = lines1
        l1 = horizontal[(-horizontal[:, 2]/horizontal[:, 1]).argmin()] #y-intercept - top
        l2 = vertical[(-vertical[:, 2]/vertical[:, 0]).argmin()] #x-intercept - left
        l3 = horizontal[(-horizontal[:, 2]/horizontal[:, 1]).argmax()] #x-intercept - bottom
        l4 = vertical[(-vertical[:, 2]/vertical[:, 0]).argmax()] #y-intercept - right
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


    def crop_image(self, image, hl_threshold=500, resize=True):
        edges = cv2.Canny(
            ImgCropper.get_binary_img(image), 
            100, 120, apertureSize=3, L2gradient=True
        )
        lines = ImgCropper._find_lines(edges, hl_threshold)
        #find sheet borders
        line_group1, mask, vp1 = get_codirectional_lines(lines, iters=2000, epsilon=0.05)
        line_group2, _, _ = get_codirectional_lines(lines[~mask], iters=2000, epsilon=0.05)
        self._find_corner_points(line_group1, vp1, line_group2)
        #get shape of cropped image
        if resize:
            self.cropped_shape = ImgCropper._get_ratio(self.sheet_corners)
        else:
            self.cropped_shape = image.shape[1], image.shape[0]
        srcpts = float32([self.sheet_corners])
        destpts = float32([[0, 0], [0, self.cropped_shape[1]], self.cropped_shape, [self.cropped_shape[0], 0]])
        resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
        return cv2.warpPerspective(image, resmatrix, self.cropped_shape) 
    