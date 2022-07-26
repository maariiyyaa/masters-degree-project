import cv2
import numpy as np
import matplotlib.pyplot as plt

from RANSAC import RANSAC



class ImgCropper:
    
    def __init__(self, warning=True):
        if not warning:
            np.seterr(invalid='ignore')
        self.image = None
        self.edges = None
        self.sheet_corners = []
        self.cropped_shape = None
        
        
    @staticmethod   
    def _imge_enhancement(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dilated = cv2.morphologyEx(
            gray, cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))
        diff2 = 255 - cv2.subtract(cv2.medianBlur(dilated, 5), gray)
        normed = cv2.normalize(diff2,None, 10, 255, cv2.NORM_MINMAX )
        bw = cv2.threshold(normed, 210, 255, cv2.THRESH_BINARY)[1]
        return bw

        
    def _find_edges(self, do_enhancement):
        if do_enhancement:
            self.image = ImgCropper._imge_enhancement(self.image)
        self.edges = cv2.Canny(self.image, 100, 120, apertureSize=3, L2gradient=True)
      
       
    @staticmethod
    def _find_lines(edges):
        max_axis = max(edges.shape)
        min_axis = min(edges.shape)
        lines = cv2.HoughLines(edges, 1, np.pi/360, 500,) 
        lines_coefs = np.empty((len(lines), 3))
        for i, line in enumerate(lines):
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + max_axis*(-b))
                y1 = int(y0 + max_axis*(a))
                x2 = int(x0 - max_axis*(-b))
                y2 = int(y0 - max_axis*(a))
                lines_coefs[i] = np.cross((x1, y1, 1), (x2, y2, 1))
        return lines_coefs



    def _find_corner_points(self,lines):
        l1 = lines[(lines[:, 1]/lines[:, 2]).argmin()] #y-intercept - top
        l2 = lines[(lines[:, 0]/lines[:, 2]).argmin()] #x-intercept - left
        l3 = lines[(lines[:, 0]/lines[:, 2]).argmax()] #x-intercept - bottom
        l4 = lines[(lines[:, 1]/lines[:, 2]).argmax()] #y-intercept - right
        self.sheet_corners = np.array(
            list(map(lambda x, y: (np.cross(x, y)/np.cross(x, y)[2])[:2], 
                     [l1, l2, l3, l4],
                     [l2, l3, l4, l1])))

    @staticmethod
    def _get_border_length(point1, point2):
        # sqrt((x1-x0)^2 + (y1-y0)^2), distance between corner points
        return np.linalg.norm(point2-point1)

    
    #get frame ratio
    def _get_ratio(self):  
        length_list = []
        for i, point in enumerate(self.sheet_corners):
            try:
                length_list.append(ImgCropper._get_border_length(point, self.sheet_corners[i+1]))
            except IndexError:
                length_list.append(ImgCropper._get_border_length(point, self.sheet_corners[0]))
        x_length = max(length_list[1], length_list[3])
        y_length = max(length_list[0], length_list[2])
        x_length_norm = x_length / max(x_length, y_length)
        y_length_norm = y_length / max(x_length, y_length)
        return x_length_norm, y_length_norm


    def crop_image(self, image, do_enhancement=True, verbose=False):
        self.image = image
        self._find_edges(do_enhancement)
        lines = ImgCropper._find_lines(self.edges)
        #find sheet borders
        ransac = RANSAC(iters=800, eps=0.05)
        line_group1 = ransac.get_direction_lines(lines)
        line_group2 = ransac.get_direction_lines(lines[~ransac.inliers_mask])
        selected_lines = np.concatenate((line_group1, line_group2), axis=0)
        
#         if verbose:
#             img_copy = self.image.copy()
#             for line in [top, left, bottom, right]:
#                 cv2.line(img_copy,line[0][:2], line[1][:2],(255,255,255), 5)
#             plt.figure(figsize=(5,10))
#             plt.clf()
#             plt.axis("off")
#             plt.title("Lines")
#             plt.imshow(img_copy)
#             del img_copy

        self._find_corner_points(selected_lines)
        #get shape of cropped image
        self.cropped_shape = tuple(map(lambda x: int(max(image.shape)*x), self._get_ratio()))
        srcpts = np.float32([self.sheet_corners]) 
        destpts = np.float32([[0, 0],  [0, self.cropped_shape[1]], self.cropped_shape, [self.cropped_shape[0], 0]]) 
        resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
        return cv2.warpPerspective(image, resmatrix, self.cropped_shape) 