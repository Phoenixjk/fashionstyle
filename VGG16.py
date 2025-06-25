import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from datasets import ImageDataset
from CrossAttention import CrossAttention
import torch.nn as nn
import torch.nn.functional as F


# def resize(output_features):
#     interpolate_transform = transforms.Resize((77, 768), interpolation=transforms.InterpolationMode.BICUBIC)
#     return interpolate_transform(output_features)


def resize(output_features): #[8,512,256]

    linear_layer = nn.Sequential(
        nn.Conv1d(512, 77, kernel_size=1, padding='same'),
        nn.Linear(256, 768)
    )

    conv_transpose = nn.ConvTranspose2d(77, 77, kernel_size=256, stride=256, padding=0, output_padding=0, groups=77)
    output_features = output_features.unsqueeze(3)  # 增加一个维度，用于分组卷积
    output_features = conv_transpose(output_features)
    output_features = output_features.squeeze(3)  # 移除分组维度



# 加载预训练的VGG16模型，去除全连接层，并使用最新的参数命名
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features  # 或使用 IMAGENET1K_V1
vgg16.eval()
# 将模型转移到GPU上（如果有的话），并设置为评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)  # 设置为评估模式


transforms_2 = [transforms.Resize(int(512 * 1.12), transforms.InterpolationMode.BICUBIC),  # 调整输入图片的大小
                    #transforms.RandomCrop(opt.size),  # 随机裁剪256*256
                    transforms.Resize(512),
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 归一化，这两行不能颠倒顺序呢，归一化需要用到tensor型

dataloader = DataLoader(ImageDataset('datasets/edges2shoes', transforms_2=transforms_2, unaligned=True),  #数据集
                            batch_size=8, shuffle=False, num_workers=0, drop_last=True)



for batch in dataloader:
    neirong_images = batch['neirong'].to(device)     #3*512*512
    fengge_images = batch['fengge'].to(device)
    with torch.no_grad():  #提取特征
        neirong_features = vgg16(neirong_images)   #内容特征
        fengge_features = vgg16(fengge_images)     #风格特征  512*16*16
        # 打印特征的形状，以确保提取正确
    # print(f"内容图片张量形状: {neirong_images.shape}")
    # print(f"风格图片张量形状: {fengge_images.shape}")
    # print("------------------------------------------")
    # print(f"内容图片特征形状: {neirong_features.shape}")
    # print(f"风格图片特征形状: {fengge_features.shape}")
    # print("\n")


    # 使用CrossAttention类
    # 假设 content_features 和 style_features 是从 VGG 模型提取的特征
    # content_features: [batch_size, C_content, H, W]
    # style_features: [batch_size, C_style, H', W']
    # 将特征展平，以适应CrossAttention模型
    neirong_features_flat = neirong_features.view(neirong_features.size(0), -1, neirong_features.size(2) * neirong_features.size(3))
    fengge_features_flat = fengge_features.view(fengge_features.size(0), -1, fengge_features.size(2) * fengge_features.size(3))  #512*256
    # print(neirong_features_flat.size(1), fengge_features_flat.size(1))
    # 初始化交叉注意力模块
    #[512]
    a = neirong_features_flat.size(2)   #256
    b = fengge_features_flat.size(2)
    cross_attention = CrossAttention(query_dim=neirong_features_flat.size(2), context_dim=fengge_features_flat.size(2), heads=8, dim_head=64, dropout=0.).to(device)
    # print(neirong_features_flat.shape, fengge_features_flat.shape)
    # 应用交叉注意力
    output_features = cross_attention(neirong_features_flat, fengge_features_flat)   #交叉注意力输出的融合特征   [8,512,256]
    final_feature = output_features + neirong_features_flat + fengge_features_flat
    print(output_features.shape)
    # output_features = resize(output_features.to(device))
 #--------------------------------------
    output_features = resize(output_features)
    print(output_features.shape)
    # 现在X_reshaped就是目标尺寸的张量
    # 检查输出张量的形状
    #该往扩散模型里移了


