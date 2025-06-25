import cv2
import numpy as np
import os

def extract_clothing_shape(square_image_path, clothing_image_path, output_path):
    square_image = cv2.imread(square_image_path)
    if square_image is None:
        print("正方形图片读取失败，请检查路径是否正确")
        return
    square_image = cv2.resize(square_image, (256, 256))
    # 读取背景为白色的服装图片
    clothing_image = cv2.imread(clothing_image_path)
    if clothing_image is None:
        print("服装图片读取失败，请检查路径是否正确")
        return
    clothing_image = cv2.resize(clothing_image, (256, 256))
    # 转换为HSV颜色空间
    hsv_clothing_image = cv2.cvtColor(clothing_image, cv2.COLOR_BGR2HSV)
    # 设置HSV颜色范围以去除白色背景
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    # 创建掩码，排除白色背景
    mask = cv2.inRange(hsv_clothing_image, lower_white, upper_white)
    # 反转掩码，使得服装区域为1
    mask_inv = cv2.bitwise_not(mask)
    # 形态学操作，去除小噪声
    kernel = np.ones((3, 3), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    # 确保正方形图片是RGB格式
    if square_image.ndim == 2:  # 如果是灰度图像，转换为BGR
        square_image = cv2.cvtColor(square_image, cv2.COLOR_GRAY2BGR)
    # 创建一个白色背景
    white_background = np.full_like(square_image, 255)
    # 使用掩码将服装区域从正方形图片中提取出来，并应用于白色背景
    clothing_shape = np.where(mask_inv[:, :, np.newaxis] == 255, square_image, white_background)
    # 保存结果
    cv2.imwrite(output_path, clothing_shape)