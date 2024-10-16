'''

v2采用先进先出、快速循环的算法，减少内存的使用，显著提速（12.3%）
v1基于距离变换的断层间图像插值方法
该程序可用来演示插值算法的效果，
配置：输入文件夹路径、输出文件夹路径、插值数 三个参数
输入：原始断层图像
输出：插值后断层图像
'''

import time
from PIL import Image
import numpy as np
import os

from scipy.ndimage import distance_transform_edt
import shutil

# 确保输出文件夹存在；删除文件夹中的原有内容
def folder_preparation(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print('输出文件夹已建立')
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # 删除文件
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 递归删除文件夹及其所有内容
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    print('输出文件夹已清空')
    print('输出文件夹已准备就绪')

# 计算图像中非零点到最近背景点（即0）的距离
# 距离变换
def compute_distance_map(matrix):

    distance_inside = distance_transform_edt(matrix)

    # 反转矩阵，得到轮廓外部的距离（值为0的像素）
    distance_outside = distance_transform_edt(1 - matrix)

    # 将外部距离变为负数
    distance_outside = -distance_outside

    # 轮廓外部使用负值，内部使用正值，轮廓点为0
    distance_map = np.where(matrix == 1, distance_inside, 0)
    distance_map = np.where(matrix == 0, distance_outside, distance_map)

    return distance_map


# 将距离图转换为矩阵
def convert_distance_map_to_matrix(distance_map):
    # 将负数变为0，其他的变为1
    binary_map = np.where(distance_map < 0, 0, 1)
    matrix = binary_map * 255
    return matrix


# 将矩阵保存为png
def save_matrix_as_png(matrix, folder_path, number):
    img = Image.fromarray(matrix.astype(np.uint8))  # 将矩阵转换为PIL图像对象
    file_path = os.path.join(folder_path, str(number).zfill(6) + '.png')  # 构造完整的文件路径,自动补0
    img.save(file_path)  # 保存图像为PNG文件


def main(input_folder, output_folder, num_interpolations):

    folder_preparation(output_folder)
    start_tim = time.time()
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    # distance_map1=np.array( dtype=int)
    # distance_map2 = np.array( dtype=int)
    for i, image_file in enumerate(image_files):
        start_inter = time.time()
        if i == (len(image_files)-1):   # 插值结束
            break
        if i == 0:  # 第一次进入，先加载第一张
            image_path_1 = os.path.join(input_folder, image_file)  # 组成文件完整的路径
            img1 = Image.open(image_path_1).convert('L')  # 读取图像（灰度模式）
            img1.save(os.path.join(output_folder, str(0).zfill(6)+'.png'))
            img1_matrix = np.array(img1)  # 将图片转换为numpy矩阵
            binary_matrix1 = np.where(img1_matrix > 128, 1, 0)  # 二值化
            distance_map1 = compute_distance_map(binary_matrix1)  # 生成距离图
        image_path_2 = os.path.join(input_folder, image_files[i + 1])  # 组成文件完整的路
        img2 = Image.open(image_path_2).convert('L')  # 读取图像（灰度模式）
        img2.save(os.path.join(output_folder, str((i + 1) * (num_interpolations + 1)).zfill(6) + '.png'))
        img_matrix2 = np.array(img2)  # 将图片转换为numpy矩阵
        binary_matrix2 = np.where(img_matrix2 > 128, 1, 0)  # 二值化
        distance_map2 = compute_distance_map(binary_matrix2)  # 生成距离图
        for j in range(1, num_interpolations + 1):
            t = j / (num_interpolations + 1)    # 计算插值系数 t: 从 0 到 1 的线性比例
            interpolated_map = (1 - t) * distance_map1 + t * distance_map2 # 插值公式: (1-t)*map1 + t*map2
            interpolated_matrix = convert_distance_map_to_matrix(interpolated_map)
            save_matrix_as_png(interpolated_matrix, output_folder, i * (num_interpolations + 1) + (j))

        distance_map1 = distance_map2   # 本次的第二张图是下次的第一张图
        end_inter = time.time()
        print(f'第{i+1}组插值完成，耗时:{end_inter - start_inter:.4f}秒')
    end_tim = time.time()
    print(f'转换总耗时：{end_tim - start_tim:.4f} 秒')

if __name__ == '__main__':
    # 设置输入和输出文件夹路径
    input_folder = 'detection_out_interpolation_in'  # 输入图片所在的文件夹
    output_folder = 'interpolation_out_3d_in'  # 输出图片的文件夹
    num_interpolations = 3  # 插值数
    main(input_folder, output_folder, num_interpolations)
