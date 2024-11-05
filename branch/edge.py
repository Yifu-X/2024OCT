'''

边缘提取，Canny方法，输出是一个宽度可调的二值图像

'''
import cv2
import os
import numpy as np

def extract_contours_with_width(input_folder, output_folder, line_thickness=5):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 可根据需要添加文件格式
            # 读取图像
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 使用Canny边缘检测
            edges = cv2.Canny(img, 100, 200)

            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 创建空白图像以绘制轮廓
            contour_image = np.zeros_like(img)

            # 绘制轮廓
            cv2.drawContours(contour_image, contours, -1, (255), thickness=line_thickness)

            # 保存提取的轮廓图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, contour_image)

# 使用示例
line_thinckness = 3 # 可调整线宽
input_folder = r'data\123456_interpolation'  # 输入文件夹路径
output_folder = rf'data\edge_{line_thinckness}'  # 输出文件夹路径
extract_contours_with_width(input_folder, output_folder, line_thinckness)
