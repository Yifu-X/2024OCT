'''

轮廓线平滑,平滑算法可选

'''

import cv2
import os
from tqdm import tqdm # 终端进度条

def filter(input_folder, output_folder, method:str, core):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 批量处理所有图像文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(('.jpg')):  # 可根据需要添加文件格式

            # 读取二值轮廓线
            img_path = os.path.join(input_folder, filename)
            binary_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if(method == "gauss"):
                # 应用高斯滤波
                smoothed_image = cv2.GaussianBlur(binary_image, core, 0)
            if (method == "mean"):
                # 应用均值滤波
                smoothed_image = cv2.blur(binary_image, core)

            # 保存滤波后的轮廓线
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, smoothed_image)

# 使用示例
method = "mean" # 可选滤波方法:mean,gauss
core = (2,2) # 卷积核大小
input_folder = r'data\edge_1'  # 输入文件夹路径
output_folder = rf'data\edge_filter_{method}'  # 输出文件夹路径
filter(input_folder, output_folder, method, core)
