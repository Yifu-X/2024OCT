import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置控制台编码为utf-8（解决中文显示问题）
if sys.version_info[0] < 3:
    sys.reload(sys)
    sys.setdefaultencoding('utf-8')

# 设置Windows控制台的输出编码为utf-8
os.system('chcp 65001')

# 读取图像并转换为灰度图
image = cv2.imread('data/123456_interpolation/001019.jpg', cv2.IMREAD_GRAYSCALE)
# 使用Canny边缘检测
background = np.zeros_like(image) # 创建空白图像以绘制轮廓
edges = cv2.Canny(image, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 查找外部轮廓
image_with_center = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)    #   单通道转为多通道
cv2.drawContours(image_with_center, contours, -1, (255,255,255), thickness=1)    # 绘制轮廓，线宽=1

image_with_center_2 = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

'''
中心提取
'''
# 计算距离变换：计算所有白点到黑点的距离（我用的是插值后的原图，白色直接代表内部）
distance_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
normalized_distance = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX)  # 将距离变换的结果归一化到 [0, 255] 范围
normalized_distance = np.uint8(normalized_distance) # 转换为 uint8 类型以便于显示
flat_indices = np.argsort(normalized_distance, axis=None)[::-1][:20]  # 获取前20个最大值的扁平化索引
max_coords = np.unravel_index(flat_indices, normalized_distance.shape)# 将扁平化的索引转换为二维坐标
x_coords = max_coords[1]  # 获取x坐标
y_coords = max_coords[0]  # 获取y坐标
# 计算前20个最大值的平均坐标
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
num = 0
max_num = None

# 遍历每个轮廓
for contour in contours:
    for point in contour:   # 逆时针遍历
        x, y = point[0]  # 获取轮廓点的坐标

        # 计算当前点到图像中心的欧氏距离
        distance = np.sqrt((x - avg_x) ** 2 + (y - avg_y) ** 2)

        # 更新最远点
        if distance > max_distance:
            max_distance = distance
            farthest_point = (x, y)
            max_num = num
        num += 1

# 如果找到了最远点，标记出来
if farthest_point is not None:
    cv2.circle(image_with_center, farthest_point, 3, (0, 255, 255), -1)  # 黄色圆圈标记
    print(f"：最远点半径：{max_distance}")
    print(f"最远点序号：{max_num}")
'''
重心提取
'''
# 计算图像的重心
M = cv2.moments(image)  # 使用Hu矩
cx = int(M["m10"] / M["m00"])  # 计算X轴方向的质心
cy = int(M["m01"] / M["m00"])  # 计算Y轴方向的质心
# 重心可视化
cv2.circle(image_with_center, (cx, cy), 3, (255, 0, 0), -1)  # 在质心位置画一个红点

'''

'''
# 存储当前轮廓点的极坐标
r_values = []
theta_values = []
# 一阶导数
r_prime_values = []  # 存储r_prime值
# 计算曲率
curvature_values = []
# 遍历每个轮廓
for contour in contours:
    # 计算每个点的极坐标 (r, theta)
    points_with_polar = []
    for point in contour:
        x, y = point[0]
        # 以中心为极点建立极坐标系
        dx = np.float32(x - avg_x)
        dy = np.float32(y - avg_y)
        # 计算到中心的距离r 和 角度θ
        r, theta = cv2.cartToPolar(np.array([dx]), np.array([dy]))  # 转换到极坐标系
        # 将极坐标值添加到列表中
        points_with_polar.append((r[0], theta[0], point))

    # 按照极角（theta）对点进行排序，确保从极角最小的点开始
    sorted_points = sorted(points_with_polar, key=lambda p: p[1])  # p[1] 是 θ
    # 将排序后的点添加到 r_values 和 theta_values
    for r, theta, point in sorted_points:
        r_values.append(r)
        theta_values.append(theta)

    r_prime_last = 0
    # 计算曲率，使用数值差分估算r'(θ) 和 r''(θ)
    for i in range(2, len(r_values) - 2, 2):  # 避免越界，确保有足够的点来计算曲率
        # 计算r'(θ)的数值差分（前向差分，后向差分）
        r_prime = (r_values[i + 1] - r_values[i - 1]) / (theta_values[i + 1] - theta_values[i - 1])
        r_double_prime = (r_values[i + 1] - 2 * r_values[i] + r_values[i - 1]) / (
                    (theta_values[i + 1] - theta_values[i]) ** 2)

        r_prime_values.append(r_prime)  # 将r_prime保存到列表中

        # 计算曲率
        # curvature = r_double_prime / (1 + r_prime ** 2) ** (3 / 2)
        curvature = (r_values[i]**2+2*r_prime**2-r_values[i]*r_double_prime)/(r_values[i]**2 + r_prime ** 2)** (3 / 2)
        curvature_values.append(curvature)

         # 可视化曲率（例如曲率大于某个阈值的点标记出来）
        #if r_prime*r_prime_last <0 :  # 变号
            #cv2.circle(image_with_center, tuple(contour[i][0]), 2, (0, 0, 255), -1)  # 用红色标记曲率较大的点
        r_prime_last = r_prime

cv2.circle(image_with_center, tuple(contour[128][0]), 3, (0, 0, 255), -1)  # 用红色标记曲率较大的点
cv2.circle(image_with_center, tuple(contour[204][0]), 3, (0, 0, 255), -1)  # 用红色标记曲率较大的点

# 遍历轮廓的所有点，除了 128 到 204 之间的点
# 假设contour是一个轮廓，start_idx 和 end_idx 是你想绘制的区间的起始和结束点索引
start_idx = 0  # 起始点索引
end_idx = 200    # 结束点索引
# 提取子轮廓的点
sub_contour1 = contour[:128]
sub_contour2 = contour[204:]

# 绘制这段轮廓
cv2.polylines(image_with_center_2, [sub_contour1], isClosed=False, color=(255, 255, 255), thickness=1)
cv2.polylines(image_with_center_2, [sub_contour2], isClosed=False, color=(255, 255, 255), thickness=1)

# 绘制图像
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
# 绘制 r_values
axs[0].plot(r_values, label='r (radius)', color='b')
axs[0].set_title('Radius (r) over contour points')
axs[0].set_xlabel('Point index')
axs[0].set_ylabel('Radius (r)')
axs[0].legend()

# 绘制 r_prime_values
axs[1].plot(r_prime_values, label="r' (radius derivative)", color='m')
axs[1].set_title("r' (Derivative of radius) over contour points")
axs[1].set_xlabel('Point index')
axs[1].set_ylabel("r' (radius derivative)")
axs[1].legend()

theta_values_degrees = np.degrees(theta_values)  # 将弧度转换为角度制
# 绘制 theta_values
axs[2].plot(theta_values_degrees, label='θ (angle)', color='g')# 输出角度值
axs[2].set_title('Angle (θ) over contour points')
axs[2].set_xlabel('Point index')
axs[2].set_ylabel('Angle (θ)')
axs[2].legend()

# 自动调整布局
plt.tight_layout()

# 显示图形
#plt.show()

# cv2.imshow('Image with 2 Centers', image_with_center)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('主血管轮廓提取', image_with_center_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

