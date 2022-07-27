from numpy import arange
from numpy import cross
from numpy import array
from numpy import seterr
from numpy import arccos
from numpy.random import choice
from numpy.linalg import norm


class RANSAC:

    def __init__(self, iters=500, eps=0.01):
        """

        :param iters: amount of iterations
        :param eps: distance threshold
        """
        seterr(invalid='ignore')
        self.iters = iters
        self.eps = eps
        self.cross_point = None

    def _filter_lines(self, lines, l1, cross_point):
        """
        Inequality under the sum in vanishing point formula
        :param lines: array of lines
        :param l1: a line for which a cross_point was found
        :param cross_point: point of intersection of l1 and another line
        :return: array of True or False
        """
        point = cross(l1, x)
        return [abs(arccos(cross_point.T.dot(point/norm(point)))) < self.eps for x in lines]

    def _get_inliers(self, lines):
        """
        Apply formula to get vanishing point
        :param lines: array of lines
        :return: None
        """
        self.inliers_mask = array([False], dtype=bool)
        for _ in range(self.iters):
            l1, l2 = lines[choice(arange(len(lines)), size=2, replace=False)]
            intercept = cross(l1, l2)
            cross_point_temp = intercept / norm(intercept)
            mask_temp = array(self._filter_lines(lines, l1, cross_point_temp), dtype=bool)
            if len(mask_temp[mask_temp]) > len(self.inliers_mask[self.inliers_mask]):
                self.inliers_mask = mask_temp
                self.cross_point = cross_point_temp

    def get_direction_lines(self, lines):
        """
        Get one direction lines
        :param lines: np.ndarray(shape=(n, 3)),
              lines in projective space defined as np.cross((x1, y1, 1), (x2, y2, 1))
        :return: subset of lines which have one direction
        """
        self._get_inliers(lines)
        return lines[self.inliers_mask]
