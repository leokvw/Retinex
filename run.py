import sys
import os
import numpy as np
import cv2
import json
# 导入Retinex实现文件
import retinex

# 图像直方图均衡
def coloredHistoEqual(img):
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_cvt[:, :, 0] = cv2.equalizeHist(img_cvt[:, :, 0])
    img_histogram = cv2.cvtColor(img_cvt, cv2.COLOR_YCrCb2BGR)
    return img_histogram


# 伽马变换
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

data_path = 'data'
# 遍历data文件夹下的所有文件（默认均为图像文件）
img_list = os.listdir(data_path)
if len(img_list) == 0:
    # 如果文件夹为空，则报错
    print('Data directory is empty.')
    exit()
# 读入自定义参数
with open('config.json', 'r') as f:
    config = json.load(f)
# 遍历读入的图像
for img_name in img_list:
    if img_name == '.gitkeep':
        continue
    # 读入图像
    img = cv2.imread(os.path.join(data_path, img_name))

    img_histogram = coloredHistoEqual(img)
    img_gamma  = adjust_gamma(img)
    # 多尺度Retinex带色彩恢复
    img_msrcr = retinex.MSRCR(
        img,
        config['sigma_list'],
        config['G'],
        config['b'],
        config['alpha'],
        config['beta'],
        config['low_clip'],
        config['high_clip']
    )

    img_amsrcr = retinex.automatedMSRCR(
        img,
        config['sigma_list']
    )
    # MSR with chromaticity preservation
    img_msrcp = retinex.MSRCP(
        img,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']        
    )    

    # 实验结果
    shape = img.shape
    cv2.imshow('Image', img)
    cv2.imshow('retinex', img_msrcr)
    cv2.imshow('Histogram equalization', img_histogram)
    cv2.imshow('Gamma Correction', img_gamma)
    cv2.imshow('Automated retinex', img_amsrcr)
    cv2.imshow('MSRCP', img_msrcp)
    cv2.waitKey()
