from math import radians

from numpy import(
    array,
    arctan2,
    argmin,
    argmax,
    abs as npabs,
    sin,
    cos,
    eye,
    cross,
    matmul,
    zeros
)
from numpy.linalg import norm, det, inv


def find_axis_points(v_inf1, v_inf2):  
    
    v_inf1 = v_inf1 / norm(v_inf1)
    v_inf2 = v_inf2 / norm(v_inf2)
    
    # set all possible directions of vp
    dirs = array([[v_inf1[0], -v_inf1[0], v_inf2[0], -v_inf2[0]],
                  [v_inf1[1], -v_inf1[1], v_inf2[1], -v_inf2[1]],
                  [v_inf1[2], -v_inf1[2], v_inf2[2], -v_inf2[2]]]) 
    
    # calculate angles between Ox and each point
    thetas = arctan2(dirs[0], dirs[1])
    
    # get idx of a point that is the most closest to +Ox axis
    horizont_vp_idx = argmin(npabs(thetas)) 
    
    # get idx the most closest to +Oy axis point
    if horizont_vp_idx//2 == 0:
        vertical_vp_idx = 2 + argmax(thetas[2:])
    else:
        vertical_vp_idx = argmax(thetas[:2])
    
#     print(f'points idxs:   h - {horisont_vp_idx}, v - {vertical_vp_idx}')    
    hor_v_point = dirs[:, horizont_vp_idx]
    vert_v_point = dirs[:, vertical_vp_idx] 
    return hor_v_point, vert_v_point


def find_horizontal_vp(v_inf1, v_inf2):  
    
    v_inf1 = v_inf1 / norm(v_inf1)
    v_inf2 = v_inf2 / norm(v_inf2)
    
    # set all possible directions of vp
    dirs = array([[v_inf1[0], -v_inf1[0], v_inf2[0], -v_inf2[0]],
                  [v_inf1[1], -v_inf1[1], v_inf2[1], -v_inf2[1]],
                  [v_inf1[2], -v_inf1[2], v_inf2[2], -v_inf2[2]]]) 
    
    # calculate angles between Ox and each point
    thetas = arctan2(dirs[0], dirs[1])
    # get idx of a point that is the most closest to +Ox axis
    horizont_vp_idx = argmin(npabs(thetas)) 
    # get idx of a point, the most closest to +Oy axis
    return horizont_vp_idx//2    


def find_homography(hor_point, vert_point, expected_angle_between_lines, shift=None, im_shape=None):
    
    # rotate the vertical v.p. to get pi/2 angle between horizontal v.p. and vertical v.p.
    rotate_angle = radians(90-expected_angle_between_lines)
    cs = cos(rotate_angle);
    sn = sin(rotate_angle);  
    vert_point_new = [
        vert_point[0] * cs + vert_point[1] * sn,
        vert_point[1] * cs - vert_point[0] * sn,
        vert_point[2]
    ]
    # find homography
    H = eye(3)
    H[0] = cross(vert_point_new, cross(hor_point, vert_point_new)) 
    H[1] = cross(cross(hor_point, vert_point_new), hor_point)
    H[2] = cross(hor_point, vert_point_new)
    
    if det(H) < 0:
        H[:, 0] = -H[:, 0]
    
    if shift:
        T = array([
            [1., 0., -float(im_shape[0])/2],
            [0., 1., -float(im_shape[1])/2],
            [0., 0.,   1.]
        ])
        T_1 = inv(T)
        H = matmul(T_1, H, T)
    return H


def transform_points(points, homography):
    
    # transform a list of points in projection plane using homography
    points_new = zeros(points.shape)
    for i, point in enumerate(points):
        points_new[i] = matmul(homography, point)
    return (points_new/points_new[:, 2][:, None])[:, :2]


def find_corner_points(lines1, lines2, vp1, vp2):
    
    if find_horizontal_vp(vp1, vp2) == 0:
        # order must be changed
        horizontal = lines2
        vertical = lines1
    else:
        # order must be changed
        horizontal = lines1
        vertical = lines2
    l1 = horizontal[(-horizontal[:, 2]/horizontal[:, 1]).argmin()] # y-intercept - top
    l2 = vertical[(-vertical[:, 2]/vertical[:, 0]).argmin()] # x-intercept - left
    l3 = horizontal[(-horizontal[:, 2]/horizontal[:, 1]).argmax()] # x-intercept - bottom
    l4 = vertical[(-vertical[:, 2]/vertical[:, 0]).argmax()] # y-intercept - right
    corner_points = cross([l1, l2, l3, l4], [l2, l3, l4, l1])
    return corner_points / corner_points[:, 2][:, None]

