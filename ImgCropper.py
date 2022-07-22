import cv2
import numpy as np
import matplotlib.pyplot as plt



class ImgCropper:
    
    def __init__(self):
        self.image = None
        self.edges = None
        self.sheet_corners = []
        self.cropped_shape = None

        
    def _find_edges(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        low_sigma = cv2.GaussianBlur(gray, (1,1), 0) #how to define a proper kernel programatically 
        high_sigma = cv2.GaussianBlur(gray, (5,5), 0) #how to define a proper kernel programatically
        # Calculate the DoG by subtracting
        dog = low_sigma - high_sigma
        #find image edges
        self.edges = cv2.Canny(gray, 60, 80, apertureSize=3, L2gradient=True) #how to define proper thresholds programatically

        
    @staticmethod
    def _find_lines(edges):
        max_axis = max(edges.shape)
        min_axis = min(edges.shape)
        horizontal_lines = []
        vertical_lines = []  

        lines = cv2.HoughLines(edges, 1, np.pi/360, int(min_axis/5),) #how to define proper threshold
        for i, _ in enumerate(lines):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + max_axis*(-b))
                y1 = int(y0 + max_axis*(a))
                x2 = int(x0 - max_axis*(-b))
                y2 = int(y0 - max_axis*(a))
                # should be replaced with RANSAC
                if abs(x1-x2) > 1000:
                    horizontal_lines.append(((x1, y1, 1), (x2, y2, 1)))
                else:
                    vertical_lines.append(((x1, y1, 1), (x2, y2, 1)))
        return horizontal_lines, vertical_lines



    def _find_corner_points(self, *boundaries):
        line_coef_list = list(map(lambda x: np.cross(x[0], x[1]), boundaries))
        for i, coefs in enumerate(line_coef_list):
            try:
                self.sheet_corners.append((np.cross(coefs, line_coef_list[i+1])/
                                           np.cross(coefs, line_coef_list[i+1])[2])[:2])
            except IndexError:
                self.sheet_corners.append((np.cross(coefs, line_coef_list[0])/
                                           np.cross(coefs, line_coef_list[0])[2])[:2])


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


    def crop_image(self, image, verbose=False):
        self.image = image
        self._find_edges()
        horizontal_lines, vertical_lines = ImgCropper._find_lines(self.edges)
        #find sheet borders
        left = vertical_lines[np.array(vertical_lines)[:,0, 0].argmin()]
        right = vertical_lines[np.array(vertical_lines)[:,0, 0].argmax()]
        top = horizontal_lines[np.array(horizontal_lines)[:,0, 1].argmin()]
        bottom = horizontal_lines[np.array(horizontal_lines)[:,0, 1].argmax()]
        if verbose:
            img_copy = self.image.copy()
            for line in [top, left, bottom, right]:
                cv2.line(img_copy,line[0][:2], line[1][:2],(255,255,255), 5)
            plt.figure(figsize=(5,10))
            plt.clf()
            plt.axis("off")
            plt.title("Lines")
            plt.imshow(img_copy)
            del img_copy

        self._find_corner_points(top, left, bottom, right)
        #get shape of cropped image
        self.cropped_shape = tuple(map(lambda x: int(max(image.shape)*x), self._get_ratio()))
        
        srcpts = np.float32([self.sheet_corners]) 
        destpts = np.float32([[0, 0],  [0, self.cropped_shape[1]], self.cropped_shape, [self.cropped_shape[0], 0]]) 
        resmatrix = cv2.getPerspectiveTransform(srcpts, destpts)
        return cv2.warpPerspective(image, resmatrix, self.cropped_shape) 