'''
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

# 删除文件夹中的所有内容
def clear_output_folder(folder_path):

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # 删除文件
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 递归删除文件夹及其所有内容
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    print('输出文件夹已清空!')

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

# 插值
def interpolate_between_maps(map1, map2, num_interpolations):
    interpolated_maps = []

    for i in range(1, num_interpolations + 1):

        # 计算插值系数 t: 从 0 到 1 的线性比例
        t = i / (num_interpolations + 1)

        # 插值公式: (1-t)*map1 + t*map2
        interpolated_map = (1 - t) * map1 + t * map2

        interpolated_maps.append(interpolated_map)

    return interpolated_maps

# 将距离图转换为二值图
def convert_distance_map_to_binary(distance_map):
    # 将负数变为0，其他的变为1
    binary_map = np.where(distance_map < 0, 0, 1)
    return binary_map

# 将二值图转换为png并保存
def save_matrix_as_png(matrix, folder_path, number):
    # 确保文件夹存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    matrix = matrix * 255   # 将矩阵值从0和1映射到0和255（黑色和白色）
    img = Image.fromarray(matrix.astype(np.uint8))  # 将矩阵转换为PIL图像对象
    file_path = os.path.join(folder_path, str(number).zfill(6)+'.png')  # 构造完整的文件路径,自动补0
    img.save(file_path)     # 保存图像为PNG文件

# 读取全部png文件
def read_png_images(folder_path):

    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    return image_files  # 文件名序列

# 读取PNG图片，并转换为距离图
# 张量，适合使用pytorch库
def transform_distance_maps(folder_path):

    image_files = read_png_images(folder_path)    # 所有文件名的列表
    distance_maps = []
    count = 0
    start_dis_all = time.time()
    # 循环处理（考虑这里是否必要循环处理，是否可以整体处理，因为图片与图片连接部位并不会影响）
    for image_file in image_files:
        start_dis = time.time()
        # 读取文件名,建立索引
        image_path = os.path.join(folder_path, image_file)     # 组成文件完整的路径
        img = Image.open(image_path).convert('L')  # 读取图像（灰度模式）
        # 二值化
        img_matrix = np.array(img)  # 将图片转换为numpy矩阵
        binary_matrix = np.where(img_matrix > 128, 1, 0)  # 将图片转换为二值矩阵，白色(255)为1，黑色(0)为0，128为阈值，调整可区分黑白
        # 生成距离图
        distance_map = compute_distance_map(binary_matrix)  # 生成距离图
        # 保存为距离图张量
        distance_maps.append(distance_map)
        end_dis = time.time()
        print(f'Image {count}: {image_file} 转换耗时：{end_dis - start_dis:.4f} 秒')
        count = count + 1
    end_dis_all = time.time()
    print(f'转换总耗时：{end_dis_all - start_dis_all:.4f} 秒')
    return distance_maps,image_files

def process_images_in_folder(input_folder, output_folder,num_interpolations=3):
    start_tim = time.time()
    distance_maps, image_files = transform_distance_maps(input_folder)
    print(f'插值数量：{num_interpolations}')
    for idx in range(len(distance_maps) - 1):

        # 获取相邻的两张distance_map
        map1 = distance_maps[idx]
        map2 = distance_maps[idx + 1]

        # 保存原始的第一张map
        if idx==0 :
            binary1 = convert_distance_map_to_binary(map1)
            save_matrix_as_png(binary1, output_folder, 0)

        # 计算插值图像
        start_int = time.time()
        interpolated_maps = interpolate_between_maps(map1, map2, num_interpolations)
        for j, interpolated_map in enumerate(interpolated_maps):
            itpl = convert_distance_map_to_binary(interpolated_map)
            save_matrix_as_png(itpl, output_folder, idx*(num_interpolations+1)+(j+1))
        end_int = time.time()
        print(f'第 {idx+1}组插值耗时：{end_int - start_int:.4f} 秒')

        # 保存第二张map
        binary2 = convert_distance_map_to_binary(map2)
        save_matrix_as_png(binary2, output_folder, (idx+1)*(num_interpolations+1))
    end_tim = time.time()
    print(f'转换总耗时：{end_tim - start_tim:.4f} 秒')

if __name__ == '__main__':
    # 设置输入和输出文件夹路径
    input_folder = 'detection_out_interpolation_in'  # 输入图片所在的文件夹
    output_folder = 'interpolation_out_3d_in'  # 输出图片的文件夹
    num_interpolations = 3  # 插值数
    clear_output_folder(output_folder)     # 清空输出文件夹内的文件
    process_images_in_folder(input_folder, output_folder, num_interpolations)
