#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 15:28
# @Author  : Yin Guibao
# @File    : datasets.py
import glob
import random
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset2(Dataset):
    def __init__(self, root, transforms_1=None,transforms_2=None, unaligned=False, mode='train'):
        self.transform2 = transforms.Compose(transforms_2)
        self.unaligned = unaligned

        self.files_content = sorted(glob.glob(os.path.join(root, '%s/fengge' % mode) + '/*.*'))
        self.files_style = sorted(glob.glob(os.path.join(root, '%s/neirong' % mode) + '/*.*'))

    def __getitem__(self, index):
        # 根据 unaligned 选择内容图像和风格图像
        if self.unaligned:
            content_idx = random.randint(0, len(self.files_content) - 1)
            style_idx = random.randint(0, len(self.files_style) - 1)
        else:
            content_idx = index % len(self.files_content)
            style_idx = index % len(self.files_style)

        # 打开图像，转换为RGB，并应用transforms
        item_content = self.transform2(Image.open(self.files_content[content_idx]).convert('RGB'))
        item_style = self.transform2(Image.open(self.files_style[style_idx]).convert('RGB'))

        # 返回一个字典，包含图像Tensor和文件名
        return {
            'neirong': item_content,
            'fengge': item_style,
            'content_name': os.path.basename(self.files_content[content_idx]),
            'style_name': os.path.basename(self.files_style[style_idx])
        }


    def __len__(self):
        #此处的代码其实是有误的，自己在做训练的时候，先保证轮廓，内容，风格图片数量都相同，或者说先保证内容与风格的数量一致
        return max(len(self.files_style), len(self.files_content))



# transforms_2 = [transforms.Resize(int(512 * 1.12), transforms.InterpolationMode.BICUBIC),  # 调整输入图片的大小
#                     #transforms.RandomCrop(opt.size),  # 随机裁剪256*256
#                     transforms.Resize(512),
#                     # transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 归一化，这两行不能颠倒顺序呢，归一化需要用到tensor型
#
# # 创建ImageDataset实例
# dataset = ImageDataset(root='datasets/edges2shoes', transforms_2=transforms_2, unaligned=True)
#
# # 创建DataLoader实例
# dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=False)

# 使用DataLoader
# for batch in dataloader:
#     content_images = batch['neirong']  # 获取内容图像Tensor
#     style_images = batch['fengge']  # 获取风格图像Tensor
#     content_names = batch['content_name']  # 获取内容图像的文件名
#     style_names = batch['style_name']  # 获取风格图像的文件名
#     # 使用列表推导式移除 '.jpg' 并合并为单一字符串
#     content_names = ''.join([file_name[:-4] for file_name in content_names])
#     style_names = ''.join([file_name[:-4] for file_name in style_names])
#     print(content_names,style_names)

    # 在这里你可以使用content_images, style_images, content_names, style_names
    # 进行你的训练或者其他操作