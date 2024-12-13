from skimage import morphology,draw,color,io,feature
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import maximum_filter

def gujia (iamge):
    skeleton = np.zeros(iamge.shape, np.uint8)
    while (True):
        if np.sum(iamge) == 0:
            break
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        iamge = cv2.erode(iamge, kernel, None, None, 1)
        open_dst = cv2.morphologyEx(iamge, cv2.MORPH_OPEN, kernel)
        result = iamge - open_dst
        skeleton = skeleton + result
        cv2.waitKey(1)
    return skeleton

img = cv2.imread('data/123456_interpolation/001020.jpg', cv2.IMREAD_GRAYSCALE) # 读取二值化图像
edges = cv2.Canny(img, 100, 200)
cv2.imshow('original', img)
cv2.waitKey(0)
cv2.imshow('edge', edges)
cv2.waitKey(0)
# 首先通过腐蚀和膨胀，分别得到一定是主血管和一定不是主血管的区域，两者做差的到不确定区域
# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))   # 卷积核
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

sure_fg = cv2.erode(opening, kernel, iterations=2)  # 腐蚀
sure_bg = cv2.dilate(opening, kernel, iterations=2)  # 膨胀
unknown = cv2.subtract(sure_bg, sure_fg)  # unknown area


cv2.imshow('sure_fg', sure_fg)
cv2.waitKey(0)
cv2.imshow('sure_bg', sure_bg)
cv2.waitKey(0)
cv2.imshow('unknown', unknown)
cv2.waitKey(0)

# 距离变换
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
dist_normal = cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('dist_normal', dist_normal)
cv2.waitKey(0)

kernel_size = 10
local_maxima = maximum_filter(dist_normal, size=kernel_size) == dist_normal
non_zero_mask = dist_normal > 0
local_maxima[~non_zero_mask] = False    # 只保留非零区域内的局部极值点
local_max_coords = np.column_stack(np.where(local_maxima))
# 显示局部极值点
cv2.imshow('Local Maxima', local_maxima.astype(np.uint8) * 255)  # 转为 0 和 255
cv2.circle(edges, local_max_coords, 3, (0, 255, 0), -1)  # 绿色圆圈标记



cv2.waitKey(0)
cv2.destroyAllWindows()