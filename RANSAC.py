from numpy import (
    arange,
    cross,
    array,
    seterr,
    arccos,
    inner,
    absolute as abs
)
from numpy.random import choice
from numpy.linalg import norm
        
    
def get_inliers(lines, iters=500, epsilon=0.01):
    """
    Apply formula to get vanishing point
    :param lines: np.ndarray(shape=(n, 3)),
                  lines in projective space defined as np.cross((x1, y1, 1), (x2, y2, 1))
    :param iters: int, default=500
                  amount of alhorithm iterations 
    :param iters: float, default=0.01
                  distance treshold
    :return: subset of lines which have one direction, inlier mask array
    """
    inliers_mask = array([False], dtype=bool)
    for _ in range(iters):
        l1, l2 = lines[choice(arange(len(lines)), size=2, replace=False)]
        intersect = cross(l1, l2)
        cross_point_temp = intersect / norm(intersect)
        cross_point_temp_norm = cross_point_temp / norm(cross_point_temp)
        points = cross(l1, lines)
        points_norm = points / norm(points, axis=1)[:,None]
        #Inequality under the sum in vanishing point formula
        mask_temp = abs(arccos(inner(cross_point_temp_norm, points_norm))) < epsilon 
        if len(mask_temp[mask_temp]) > len(inliers_mask[inliers_mask]):
            inliers_mask = mask_temp
            cross_point = cross_point_temp
    return lines[inliers_mask], inliers_mask
