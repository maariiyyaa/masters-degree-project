from numpy import (
    arange,
    cross,
    array,
    seterr,
    arccos,
    inner,
    sort,
    absolute as npabs
)
from numpy.random import choice
from numpy.linalg import norm
        
    
def get_codirectional_lines(lines, lines_translated=[], iters=500, epsilon=0.01):
    """
    Applying RANSAC method to find the best vanishing point and
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
    lines_idxs = arange(len(lines))
    if len(lines_translated)==0:
        lines_translated = lines
    for i in range(iters):
        #get 2 random lines
        choices = choice(lines_idxs, size=2, replace=False)
        l1, l2 = lines_translated[choices]
        #find normilized cross points in the 2-sphere in 3-space 
        vanish_temp = cross(l1, l2)
        cross_points = cross(l1, lines_translated)
        vanish_temp_norm = vanish_temp / norm(vanish_temp)
        cross_points_norm = cross_points / norm(cross_points, axis=1)[:,None]
        #get distance between vanish_temp_norm and each of cross_points_norm and create a filter by `epsilon`
        mask_temp = arccos(npabs(inner(vanish_temp_norm, cross_points_norm))) < epsilon 
        #add l1 to the mask
        mask_temp[choices[0]] = True
        if len(mask_temp[mask_temp]) > len(inliers_mask[inliers_mask]):
            inliers_mask = mask_temp
            best_choice = choices
            
    return lines[inliers_mask], inliers_mask, cross(*lines[best_choice])


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
