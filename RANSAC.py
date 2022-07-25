import numpy as np


class RANSAC:
    
    def __init__(self, iters=500, eps=0.1):
        np.seterr(invalid='ignore')
        self.iters = iters
        self.eps = eps
        self.intercept = None
        self.inliers_mask = None
        
        
    def _vanishing_point_filter(self, x, l1):
        return abs(np.arccos(
            self.intercept.T.dot(np.cross(l1, x)/np.linalg.norm(np.cross(l1, x)))
            )) < self.eps
    
    
    def _line_filter(self, x):
        return abs(np.arccos(self.intercept.T.dot(x/np.linalg.norm(x)))) < self.eps
    
    
    def _get_best_vp(self, lines):
        _cross_points = np.zeros((self.iters, 3))
        _num_lines = np.zeros((self.iters))
        for i in range(self.iters):
            _l1, _l2 = lines[np.random.choice(np.arange(len(lines)), size=2, replace=False), :]
            self.intercept = np.cross(_l1,_l2)/np.linalg.norm(np.cross(_l1,_l2))
            _cross_points[i] = self.intercept
            _num_lines[i] = len(list(filter(lambda x: self._vanishing_point_filter(x, _l1), lines)))

        self.intercept = _cross_points[_num_lines.argmax()]
        
    
    def get_direction_lines(self, lines):
        self._get_best_vp(lines)
        self.inliers_mask = np.array(list(map(lambda x: self._line_filter(x), lines)))
        return lines[self.inliers_mask]