import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

def default(val, dflt):
    return val if val is not None else dflt

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):   #query_dim=256, context_dim=256
        super().__init__()
        inner_dim = dim_head * heads   #512
        context_dim = default(context_dim, query_dim)  #256

        self.scale = dim_head ** -0.5
        self.heads = heads
        # 为内容特征和风格特征分别定义线性层
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 输出层，将注意力加权的特征映射回原始特征维度
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, content_features, style_features, mask=None):   #content_features, style_features  [8,512,64]
        h = self.heads
        # 将内容特征和风格特征分别映射到查询（Q）、键（K）和值（V）表示
        query = self.to_q(content_features)
        key = self.to_k(style_features)
        value = self.to_v(style_features)

        # 调整形状以匹配多头注意力机制
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (query, key, value))

        # 计算查询和键之间的相似度
        sim = einsum('b i d, b j d -> b i j', query, key) * self.scale

        # 如果提供了掩码，将其应用到相似度矩阵上
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # 应用softmax获取注意力权重
        attn = F.softmax(sim, dim=-1)

        # 使用注意力权重加权值
        out = einsum('b i j, b j d -> b i d', attn, value)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # 通过输出层将融合特征映射回原始特征维度
        return self.to_out(out)



