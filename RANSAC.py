from numpy import (
    arange,
    cross,
    array,
    seterr,
    arccos,
    inner,
    sort,
    absolute as abs
)
from numpy.random import choice
from numpy.linalg import norm
        
    
def get_codirectional_lines(lines, iters=500, epsilon=0.01):
    """
    Applying a RANSAC method to find the best vanish point and
    the intersecting in it lines.
    
    :param lines: np.ndarray(shape=(n, 3)),
                  lines in projective space defined as np.cross((x1, y1, 1), (x2, y2, 1))
    :param iters: int, default=500
                  number of iterations of the algorithm 
    :param epsilon: float, default=0.01
                  distance threshold
    :return: subset of lines which have one direction, inlier mask array, vanish point
    """
    
    inliers_mask = array([False], dtype=bool)
    for _ in range(iters):
        l1, l2 = lines[choice(arange(len(lines)), size=2, replace=False)]
        vanish_temp = cross(l1, l2)
        vanish_temp_norm = vanish_temp / norm(vanish_temp)
        cross_points = cross(l1, lines)
        cross_points_norm = cross_points / norm(cross_points, axis=1)[:,None]
        #Inequality under the sum in vanishing point formula
        mask_temp = abs(arccos(inner(vanish_temp_norm, cross_points_norm))) < epsilon 
        if len(mask_temp[mask_temp]) > len(inliers_mask[inliers_mask]):
            inliers_mask = mask_temp
            vanish_point = vanish_temp
    return lines[inliers_mask], inliers_mask, vanish_point




def get_best_dist(rhos, axis_lenght, iters=1000, epsilon=5):
    """
    Get approximate distance between co-directional lines
    
    :param rhos: np.ndarray(shape=(n, )),
        list of rhos of co-directional lines
    :param axis_lenght: int 
        the length of axis across which co-directional lines are passed
    :param iters: int, default=1000
                  number of iterations of the algorithm 
    :param epsilon: float, default=5
                  allowed distance deviation
    """
    best_count = 0
    for _ in range(iters):
        counter = 0
        ro1, ro2 = sort(rhos[choice(arange(len(rhos)), size=2, replace=False)])
        dist = ro2 - ro1
        #steps to left
        for i in range(int(ro1/dist)+1):
            left = rhos[(rhos > ro1 - i*dist - epsilon) & (rhos < ro1 - i*dist + epsilon)]
            if len(left):
                counter+=1
            else:
                break
        #steps to right
        for j in range(int((axis_lenght-ro2)/dist)+1):
            right = rhos[(rhos > ro2 + j*dist - epsilon) & (rhos < ro2 + j*dist + epsilon)]
            if len(right):
                counter+=1
            else:
                break
        if best_count < counter:
            best_count = counter
            best_dist = dist
            start_line = ro1%dist
    return int(best_dist), int(start_line)
