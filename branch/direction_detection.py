import cv2
import numpy as np
# 读取图像并转换为灰度图
image = cv2.imread('data/123456_interpolation/001024.jpg', cv2.IMREAD_GRAYSCALE)
edge = np.zeros_like(image) # 创建空白图像以绘制轮廓
# 使用Canny边缘检测
edges = cv2.Canny(image, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 查找轮廓
image_with_center = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)  # 以原图作为输出的背景,方便可视化
cv2.drawContours(image_with_center, contours, -1, (255,255,255), thickness=1)    # 绘制轮廓，线宽=1
'''
中心提取
'''
# 计算距离变换：计算所有白点到黑点的距离（我用的是插值后的原图，白色直接代表内部）
distance_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
normalized_distance = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX)  # 将距离变换的结果归一化到 [0, 255] 范围
normalized_distance = np.uint8(normalized_distance) # 转换为 uint8 类型以便于显示
flat_indices = np.argsort(normalized_distance, axis=None)[::-1][:20]  # 获取前20个最大值的扁平化索引
max_coords = np.unravel_index(flat_indices, normalized_distance.shape)# 将扁平化的索引转换为二维坐标
'''
# 将前10个最大值的点标记出来
for i in range(1):
    x, y = max_coords[1][i], max_coords[0][i]  # 交换顺序因为 OpenCV 的坐标是 (x, y)
    # cv2.circle(normalized_distance, (x, y), 5, (0, 255, ), -1)  # 绿色圆圈标记
    cv2.circle(image_with_center, (x, y), 5, (0, 255, 0), -1)  # 绿色圆圈标记
# 可视化归一化后的距离变换
cv2.imshow("Normalized Distance Transform", normalized_distance)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
x_coords = max_coords[1]  # 获取x坐标
y_coords = max_coords[0]  # 获取y坐标
# 计算平均坐标
avg_x = int(np.mean(x_coords))
avg_y = int(np.mean(y_coords))
# 中心可视化
cv2.circle(image_with_center, (avg_x, avg_y), 3, (0, 255, 0), -1)  # 绿色圆圈标记
'''
中心-最远边缘点
'''
# 初始化最远点的距离和坐标
max_distance = 0
farthest_point = (avg_x, avg_y)

# 遍历每个轮廓
for contour in contours:
    for point in contour:
        x, y = point[0]  # 获取轮廓点的坐标

        # 计算当前点到图像中心的欧氏距离
        distance = np.sqrt((x - avg_x) ** 2 + (y - avg_y) ** 2)

        # 更新最远点
        if distance > max_distance:
            max_distance = distance
            farthest_point = (x, y)

# 如果找到了最远点，标记出来
if farthest_point is not None:
    cv2.circle(image_with_center, farthest_point, 3, (0, 255, 0), -1)  # 绿色圆圈标记
'''
重心提取
'''
# 计算图像的重心
M = cv2.moments(image)  # 使用Hu矩
cx = int(M["m10"] / M["m00"])  # 计算X轴方向的质心
cy = int(M["m01"] / M["m00"])  # 计算Y轴方向的质心
# 重心可视化
cv2.circle(image_with_center, (cx, cy), 3, (0, 0, 255), -1)  # 在质心位置画一个红点

'''
重心-长短轴
mu00 = M["m00"]
mu11 = M["m11"] - cx*M["m01"]
mu20 = M["m20"] - cx*M["m10"]
mu02 = M["m02"] - cy*M["m01"]
theta = 1/2*np.arctan2(2*mu11/mu00, (mu20 - mu02)/mu00)
print('Angle {0:.2f}'.format(theta*180/np.pi))
# visual
rho = 150
dx_major = rho * np.cos(theta)
dy_major = rho * np.sin(theta)
dx_minor = 0.3 * rho * np.cos(theta - np.pi / 2)
dy_minor = 0.3 * rho * np.sin(theta - np.pi / 2)
# short
short_axis=[(int(cx-dx_minor),int(cy-dy_minor)),(int(cx),int(cy)),(int(cx+dx_minor),int(cy+dy_minor))]
for i in range(len(short_axis)-1):
    cv2.line(image_with_center,short_axis[i],short_axis[i+1],color=(255,255,0),thickness=2)

# long
long_axis = [(int(cx - dx_major), int(cy - dy_major)), (int(cx), int(cy)), (int(cx + dx_major), int(cy + dy_major))]
for i in range(len(long_axis) - 1):
    cv2.line(image_with_center, long_axis[i], long_axis[i + 1], color=(0, 255, 255), thickness=2)
'''
# 对每个轮廓进行曲率计算
for contour in contours:
    if len(contour) >= 5:  # 确保至少有5个点来计算曲率
        for i in range(2, len(contour) - 2, 2):  # 避免访问越界，确保i周围有足够的点
            # 获取当前点和相邻的两个点
            p_prev2 = contour[i - 2][0]  # P(i-2)
            p_prev1 = contour[i - 1][0]  # P(i-1)
            p_curr = contour[i][0]  # P(i)
            p_next1 = contour[i + 1][0]  # P(i+1)
            p_next2 = contour[i + 2][0]  # P(i+2)

            # 曲率的计算公式
            x1, y1 = p_prev2
            x2, y2 = p_prev1
            x3, y3 = p_curr
            x4, y4 = p_next1
            x5, y5 = p_next2

            # 曲率公式（使用相邻5个点估计）
            numerator = (y4 - y3) * (x3 - x2) - (x4 - x3) * (y3 - y2)
            denominator = ((x4 - x3) ** 2 + (y4 - y3) ** 2) ** (3 / 2)

            # 曲率值（如果分母为零，则跳过）
            if denominator != 0:
                curvature = numerator / denominator
            else:
                curvature = 0  # 这里避免了分母为零的情况

            # 通过曲率正负对点进行标记
            if curvature > 0:  # 假设0.1为阈值，可以调整
                cv2.circle(image_with_center, tuple(p_curr), 2, (0, 0, 255), -1)  # 红色标记曲率较大的点
            if curvature < 0:  # 假设0.1为阈值，可以调整
                cv2.circle(image_with_center, tuple(p_curr), 2, (0, 255, 0), -1)  # 红色标记曲率较大的点


cv2.imshow('Image with 2 Centers', image_with_center)
cv2.waitKey(0)
cv2.destroyAllWindows()
