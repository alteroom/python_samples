import math
import cv2
import numpy as np


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.hypot(x2-x1, y2-y1)

def findGroup(groups, square) -> list:
    s_center, _, _ = square[0]
    for group in groups:
        rect, _ =  group[0]
        center, _, _ = rect
        if distance(center, s_center) < 5:
            return group
    return None

# affine transform
# параллельный перенос translate / translation
# поворот rotate / rotation
# масштабирование scale / resize


filename = "images/qr2.png"
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, None, fx=2, fy=2)

image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
original = image_copy.copy()

image[image >= 127] = 255
image[image < 127] = 0

image = cv2.bitwise_not(image)

h, w = image.shape
cv2.imshow("image", image)

contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

squares = []
for i, cnt in enumerate(contours):
    rect = cv2.minAreaRect(cnt)
    center, size, angle = rect
    ratio = size[1] / size[0]
    if 0.85 <= ratio <= 1.15:
        squares.append((rect, cnt))
        cv2.drawContours(image_copy, contours, i, (0, 255, 0), 1)


groups = []
for i, square in enumerate(squares):
    group = findGroup(groups, square)
    if group is None:
        group = []
        groups.append(group)
    group.append(square)


dots = []
for group in groups:
    if len(group) == 3:
        cnts = [square[1] for square in group]
        cv2.drawContours(image_copy, cnts, -1, (0, 0, 255), 1)
        dots.append(group)

centers = []
if len(dots) > 0:
    for group in dots:
        square = group[0]
        #for square in group:
        rect = square[0]
        center, size, angle = rect
        centers.append(list(center))
    #        angles.append(int(angle))
    center, size, angle = dots[0][0][0]
    print(angle)
cv2.imshow("contours", image_copy)
#print(centers)
h, w = original.shape[:2]
scale = 1.0
rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
centers2 = cv2.transform(np.array([centers]), rot_mat)[0]
#print(centers2)
real_center = None
max_dist = 0
for i in range(3):
    a = i
    b = (i+1) % 3
    
    x1, y1 = centers2[a][0], centers2[a][1]
    x2, y2 = centers2[b][0], centers2[b][1]
    d = math.hypot(x2-x1, y2-y1)
    if d > max_dist:
        #print(d)
        real_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        max_dist = d

print(real_center)
top = 0
left = 0
for c in centers2:
    if c[0] < real_center[0]:
        left += 1
    if c[1] < real_center[1]:
        top += 1

print(f"LEFT: {left} TOP: {top}")

# left 2 top 2 0
# ##
# #.
new_angle = angle
# left 1 top 2 +90
# ##
# .#
if left == 1 and top == 2:
    new_angle = angle + 90
# left 2 top 1 -90
# #.
# ##
if left == 2 and top == 1:
    new_angle = angle - 90
# left 1 top 1 +180
# .#
# ##
if left == 1 and top == 1:
    new_angle = angle + 180

rot_mat = cv2.getRotationMatrix2D(real_center, new_angle, scale)

rotated_image = cv2.warpAffine(original, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
cv2.imshow("rotated", rotated_image)




cv2.waitKey()


