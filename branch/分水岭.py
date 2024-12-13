import numpy as np
from scipy import ndimage
from skimage import io
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import imutils

img = cv2.imread('data/123456_interpolation/001020.jpg', cv2.IMREAD_GRAYSCALE) # 读取图像
# 去除噪声
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))   # 卷积核
img_clean = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
# 先把它扩大一圈，分支中心会更明显一些，可能效果会更好
kernel_2 = np.ones((5, 5), np.uint8)
bigger = cv2.dilate(img_clean, kernel_2, iterations=4)  # sure background area
opening = np.uint8(bigger) # 转为uint8_t

# 距离变换,Euclidean Distance Transform (EDT)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5) # 距离变换
# dist_normal = cv2.normalize(-dist_transform, -dist_transform, 0, 1.0, cv2.NORM_MINMAX)    # 归一化

# 第一种局部最大值方法
local_max = maximum_filter(dist_transform, size=3) == dist_transform
non_zero_mask = dist_transform > 0
peaks = (local_max & non_zero_mask).astype(np.uint8)    # 去除掉背景，峰值点是1，其余点是0
markers_max = ndimage.label(peaks,structure=np.ones((3,3)))[0]
labels_max = watershed(-dist_transform,markers_max,mask = img_clean) # 由于输入了负的 dist，算法将局部最大值作为起点，沿梯度递增分割图像。标记的区域是基于距离图的峰值扩展而成的。

# coords = peak_local_max(dist_transform, footprint=np.ones((3, 3)), labels=opening)
# peaks = np.zeros(dist_transform.shape, dtype=bool)
# peaks[tuple(coords.T)] = True
# markers_max, _ = ndimage.label(peaks)
# labels_max = watershed(-dist_transform, markers_max, mask=opening)



# 把每个区域的轮廓画出来
for label in np.unique(labels_max):
    # 0表示背景，忽略不管
    if label == 0:
        continue
    mask = np.zeros(opening.shape, dtype="uint8")
    mask[labels_max == label] = 255 # 将这个区域用白色标记出来
    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # 绘制这个区域的轮廓
    cnts = imutils.grab_contours(cnts)  # 确保轮廓数据格式的一致性
    c = max(cnts, key=cv2.contourArea)

    # 创建一个空白图像用于单独显示当前轮廓
    single_contour_img = np.zeros_like(img)  # 创建与原始图像大小相同的空白图像
    cv2.drawContours(single_contour_img, [c], -1, (255), 1)  # 绘制当前轮廓

    # 显示当前轮廓
    cv2.imshow(f"Contour #{label}", single_contour_img)
    cv2.waitKey(0)
    # 显示当前轮廓
    cv2.imshow(f"Contour #{label}", single_contour_img)
    cv2.waitKey(0)
    # # draw a circle enclosing the object
    # ((x, y), r) = cv2.minEnclosingCircle(c)
    # cv2.circle(opening, (int(x), int(y)), int(r), (0, 255, 0), 2)
    # cv2.putText(opening, "#{}".format(label), (int(x) - 10, int(y)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# cv2.imshow('original', img)
# cv2.waitKey(0)
# heatmap = cv2.applyColorMap(np.uint8(dist_normal*255), cv2.COLORMAP_JET)
# cv2.imshow('Distance Map', heatmap)
# cv2.waitKey(0)
# cv2.imshow('Peaks', peaks * 255)  # 将局部峰值点标记为白色以便显示
# cv2.waitKey(0)

fig, axes = plt.subplots(ncols=5, figsize=(18, 6), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img_clean, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(opening, cmap=plt.cm.gray)
ax[1].set_title('Dilate')
ax[2].imshow(-dist_transform, cmap='jet')
ax[2].set_title('Distances')
ax[3].imshow(peaks * 255, cmap=plt.cm.gray)
ax[3].set_title('Peaks')
ax[4].imshow(labels_max, cmap=plt.cm.nipy_spectral)
ax[4].set_title('Separated objects')
for a in ax:
    a.set_axis_off()
fig.tight_layout()
plt.show()

# 首先通过腐蚀和膨胀，分别得到一定是主血管和一定不是主血管的区域，两者做差的到不确定区域
# # 背景
# sure_bg = cv2.dilate(opening, kernel, iterations=1)  # sure background area
# sure_bg = np.uint8(sure_bg) # 转为uint8_t
#
# # 前景
# ret, sure_fg = cv2.threshold(dist_normal, 0.5*dist_normal.max(), 255, 0)    # 二值化，阈值是最大值的0.5
# sure_fg = np.uint8(sure_fg) # 转为uint8_t
# unknown = cv2.subtract(sure_bg,sure_fg) # 两者相减

# cv2.imshow('sure_bg', sure_bg)
# cv2.waitKey(0)
# cv2.imshow('sure_fg', sure_fg)
# cv2.waitKey(0)
# cv2.imshow('unknown', unknown)
# cv2.waitKey(0)

# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# markers = markers+1 # 该算法给出的背景是0，但是分水岭算法需要背景是1，所以将其整体+1
# markers[unknown>200] = 0    # 然后，我们令不确定区域是0
# # 到目前，背景=1，不确定区域=0，前景>1
# markers_copy = markers.copy()   # 将其可视化出来
# markers_copy[markers==1] = 0    # 黑色表示背景，1
# markers_copy[markers==0] = 150  # 灰色表示不确定区域，0
# markers_copy[markers>1] = 255   # 白色表示前景，2
# markers_copy = np.uint8(markers_copy)
# cv2.imshow('result', markers_copy)
# cv2.waitKey(0)
#
# # 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
# # 创建一个三通道图像用于显示分割边界
# img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
# # 使用分水岭算法执行基于标记的图像分割
# mask = cv2.watershed(img_color, markers)
#
# # 将边界标记为红色
# img_color[mask == -1] = [255, 0, 0]
#
# # 显示结果
# cv2.imshow('Watershed Result', img_color)
# cv2.waitKey(0)