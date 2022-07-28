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
                  number of iterations of the algorithm 
    :param iters: float, default=0.01
                  distance threshold
    :return: subset of lines which have one direction, inlier mask array
    """
    inliers_mask = array([False], dtype=bool)
    for _ in range(iters):
        l1, l2 = lines[choice(arange(len(lines)), size=2, replace=False)]
        vanishing_temp = cross(l1, l2)
        vanishing_temp_norm = vanishing_temp/ norm(vanishing_temp)
        cross_points = cross(l1, lines)
        cross_points_norm = cross_points / norm(cross_points, axis=1)[:,None]
        #Inequality under the sum in vanishing point formula
        mask_temp = abs(arccos(inner(vanishing_temp_norm, cross_points_norm))) < epsilon 
        if len(mask_temp[mask_temp]) > len(inliers_mask[inliers_mask]):
            inliers_mask = mask_temp
    return lines[inliers_mask], inliers_mask
