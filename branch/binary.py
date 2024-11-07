'''
批量二值化处理图像
'''
import os
import cv2

def binarize_image(image_path, threshold=128):
    """使用 OpenCV 将图片二值化"""
    # 读取图片并转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 应用阈值进行二值化处理
    _, binarized_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binarized_image

def process_images_in_folder(folder_path, output_folder, threshold=128):
    """读取文件夹中所有jpg图片，并将其二值化"""
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 逐个处理每个图片文件
    for jpg_file in jpg_files:
        image_path = os.path.join(folder_path, jpg_file)
        binarized_image = binarize_image(image_path, threshold)

        # 保存二值化后的图片
        output_path = os.path.join(output_folder, f"{jpg_file}")
        cv2.imwrite(output_path, binarized_image)
        print(f"Saved binarized image: {output_path}")

# 使用示例
input_folder = r'D:\Project\python\2024OCT\branch\data\123456_interpolation_branch'  # 输入文件夹路径
output_folder = r'D:\Project\python\2024OCT\branch\data\123456_interpolation_branch_2'  # 输出文件夹路径
process_images_in_folder(input_folder, output_folder)