import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('data/edge_2+edge_filter_gauss/001024.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 将图像转换为浮动32位图像
gray_image = np.float32(gray_image)

# 使用 Harris 角点检测
dst = cv2.cornerHarris(gray_image, 2, 3, 0.016)

# 扩展角点的区域
dst = cv2.dilate(dst, None)

# 显示角点
image[dst > 0.01 * dst.max()] = [0, 0, 255]

# 显示结果
plt.imshow(image)
plt.title("Harris Corner Detection")
plt.show()
