import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(anchor_features, positive_features, negative_features, temperature=0.5):
    # 展平特征张量以计算相似度
    batch_size = anchor_features.size(0)
    anchor_flat = anchor_features.view(batch_size, -1).float()
    positive_flat = positive_features.view(batch_size, -1).float()
    negative_flat = negative_features.view(batch_size, -1).float()

    # 计算锚点和正样本的相似度
    pos_similarity = (anchor_flat * positive_flat).sum(dim=1) / temperature

    # 计算锚点和负样本的相似度
    neg_similarity = (anchor_flat * negative_flat).sum(dim=1) / temperature

    # 将相似度合并为一个批次的logits
    logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity.unsqueeze(1)], dim=1)

    # 应用 softmax 函数获取归一化的概率分布
    softmax_scores = F.softmax(logits, dim=1)

    # 计算损失，我们只关心正样本的损失
    loss = -torch.log(softmax_scores[:, 0] + 1e-9)  # 添加一个小的值以避免 log(0)

    return loss.mean()


def diffusion_model_loss(model, content_images, condition_features, steps, strength, device):
    """
    计算扩散模型的损失函数，包括重构损失和条件控制损失。

    参数:
    - model: 预训练的扩散模型
    - content_images: 原始内容图像
    - condition_features: 条件控制特征
    - steps: 去噪步骤数
    - strength: 去噪强度
    - device: 设备，用于确保数据在正确的设备上
    """
    # 确保模型处于训练模式
    model.train()

    # 前向扩散过程：将原始图像编码为噪声
    noised_images = model.forward_diffusion(content_images, steps, strength)

    # 去噪过程：使用条件特征和噪声图像重构图像
    reconstructed_images = model.reverse_diffusion(condition_features, noised_images, steps, strength)

    # 计算重构损失：原始图像与重构图像之间的差异
    reconstruction_loss = F.mse_loss(reconstructed_images, content_images)

    # [可选] 计算条件控制损失：模型输出与条件特征的一致性
    # 这里需要根据模型的实现细节来定义如何计算条件控制损失
    # 例如，可以使用条件特征与重构图像特征的相似度作为损失
    condition_control_loss = F.mse_loss(model.extract_features(reconstructed_images), condition_features)

    # 总损失是重构损失和条件控制损失的加权和
    total_loss = reconstruction_loss + condition_control_loss

    # 返回总损失
    return total_loss

def diff_extract_features(grid_feature,vgg16,device):   #输入是图片1 3 512 512  变成1 77 768的特征
    linear_layer = nn.Sequential(
        nn.Conv1d(512, 77, kernel_size=1),
        nn.Linear(256, 768)
    ).to(device)
    grid_features = vgg16(grid_feature)  # 内容特征
    grid_features_flat = grid_features.view(grid_features.size(0), -1,grid_features.size(2) * grid_features.size(3)).to(device)
    grid_features_flat = linear_layer(grid_features_flat)
    return grid_features_flat
