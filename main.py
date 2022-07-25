import cv2
import numpy as np

from RANSAC import RANSAC

imagergb = cv2.imread('images/nb.jpg')
img = cv2.cvtColor(imagergb,cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## do morph-dilate-op
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
dilated = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
## do medianBlur
median = cv2.medianBlur(dilated, 5)
diff2 = 255 - cv2.subtract(median, gray)
## do normalize 
normed = cv2.normalize(diff2,None, 10, 255, cv2.NORM_MINMAX)

edges = cv2.Canny(bw, 100, 120, apertureSize=3, L2gradient=True)


lines = cv2.HoughLines(edges, 1, np.pi/360, 500,)
lines_points = np.empty((len(_lines), 4), dtype=int)
lines_coefs = np.empty((len(_lines), 3))

for i, line in enumerate(lines):
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 4000*(-b))
        y1 = int(y0 + 4000*(a))
        x2 = int(x0 - 4000*(-b))
        y2 = int(y0 - 4000*(a))
        lines_points[i] = (x1, y1, x2, y2)
        lines_coefs[i] = np.cross((x1, y1, 1), (x2, y2, 1))
        
        
print(f"found lines: {len(lines_coefs)}")  
r = RANSAC()
line_group1 = r.get_direction_lines(lines_coefs)
print(f"lines in group 1: {len(line_group1)}")

