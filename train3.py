import argparse, os, sys, glob
import PIL
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from DBlearning import contrastive_loss,diffusion_model_loss,diff_extract_features
from idea import extract_clothing_shape
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import DataLoader
from datasets import ImageDataset
from CrossAttention import CrossAttention
import torch.nn.functional as F
from model import MyModel #我的模型
from datasets2 import ImageDataset2
sys.path.append(os.path.dirname(sys.path[0]))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import CLIPProcessor, CLIPModel


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def resize(output_features):
    interpolate_transform = transforms.Resize((77, 768), interpolation=transforms.InterpolationMode.BICUBIC)
    return interpolate_transform(output_features)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    # model.eval()
    model.train()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.






config = "configs/stable-diffusion/v1-inference.yaml"
ckpt = "models/sd/sd-v1-4.ckpt"
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}")
sampler = DDIMSampler(model)


for param in model.parameters():
    if not param.requires_grad:
        param.requires_grad = True
    if param.dtype == torch.float16:
        param.data = param.data.float()

def main(tag, style_names ,content_names, prompt, content_image, style_image ,c_feature, ddim_steps=50, strength=0.5, model=None):
    ddim_eta = 0.0
    n_iter = 1

    C = 4
    f = 8
    n_samples = 1
    n_rows = 0
    scale = 10.0

    precision = "autocast"
    outdir = "outputs/img2img-samples/test"
    # seed_everything(seed)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) + 10

    # style_image = load_img(style_dir).to(device)
    # style_image = repeat(style_image, '1 ... -> b ...', b=batch_size)
    # style_latent = model.get_first_stage_encoding(model.encode_first_stage(style_image))  # move to latent space   593  855

    # content_name = content_dir.split('/')[-1].split('.')[0]
    # content_image = load_img(content_dir).to(device)
    # content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)
    # content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))    #8*4*64*64
    # init_latent = content_latent

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = c_feature
                            # uc = resize(c_feature)
                            # uc = model.get_learned_conditioning(batch_size * [""], style_image)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = c_feature
                        # c = model.get_learned_conditioning(prompts, style_image)  # 条件控制   可能应该从这入手
                        # c = resize(c_feature)

                        # img2img

                        # stochastic encode
                        # z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))

                        # stochastic inversion
                        t_enc = int(strength * 1000)
                        x_noisy = model.q_sample(x_start=init_latent, t=torch.tensor([t_enc] * batch_size).to(device))  #8 4 64 64
                        model_output = model.apply_model(x_noisy, torch.tensor([t_enc] * batch_size).to(device), c)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device), \
                                                          noise=model_output, use_original_steps=True)

                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc, )
                        # print(z_enc.shape, uc.shape, t_enc)

                        # txt2img
                        #          noise  =torch.randn_like(content_latent)
                        #          samples, intermediates =sampler.sample(ddim_steps,1,(4,512,512),c,verbose=False, eta=1.,x_T = noise,
                        # unconditional_guidance_scale=scale,
                        # unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid_feature = grid   #1 3 512 512
                # grid_feature.size(0)
                for i in range(grid_feature.size(0)):
                    grid = make_grid(grid_feature[i], nrow=n_rows)
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    output = Image.fromarray(grid.astype(np.uint8))
                    output_path = 'outputs/img2img-samples/test/{}+{}+{}.jpg'.format(i, content_names, style_names)
                    output.save(output_path)  # [512,512]
                    all_items = os.listdir(folder_path)
                    image_names = [item for item in all_items if item.endswith('.jpg')]
                    extract_clothing_shape(
                        output_path,
                        './Images/fuzhuang/{}'.format(image_names[tag]),
                        output_path
                    )
                    grid_count += 1
                    # toc = time.time()
                # X = model.extract_features(grid_feature)
                # grid_feature_1 = grid[0]
                # grid_feature_2 = grid[1]
                # grid_feature_1 = make_grid(grid_feature_1, nrow=n_rows)
                # grid_feature_2 = make_grid(grid_feature_2, nrow=n_rows)
                # # to image
                # grid_feature_1 = 255. * rearrange(grid_feature_1, 'c h w -> h w c').cpu().numpy()
                # grid_feature_2 = 255. * rearrange(grid_feature_2, 'c h w -> h w c').cpu().numpy()
                # output1 = Image.fromarray(grid_feature_1.astype(np.uint8))
                # output = Image.fromarray(grid_feature.astype(np.uint8))
                # output.save(os.path.join(outpath, content_name+'-'+prompt+f'-{grid_count:04}.png'))
                # for i in range(grid.size(0)):
                # output1.save(os.path.join('outputs/img2img-samples/test/{}+{}style.jpg'.format(content_names, style_names)))  # [512,512]
                # output.save(os.path.join('outputs/img2img-samples/test/{}+{}2.jpg'.format(content_names, style_names)))  # [512,512]
                # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
    reconstruction_loss = F.mse_loss(grid_feature.float(), content_image)
    condition_control_loss = F.mse_loss(diff_extract_features(grid_feature.float(),vgg16,device), c_feature)
    loss = reconstruction_loss + condition_control_loss
    grid_feature.requires_grad_(True)
    # print("grid_feature",grid_feature.requires_grad)
    return grid_feature,loss





# model = model.to(device)
# model = model.eval()
# model.train()  # 将模型设置为训练模式

# for i in range(6227):
#     contentdir = "./comparison2/" + str(i) + ".jpg"
#     main(prompt = '*', content_dir = contentdir, style_dir = contentdir, ddim_steps = 50, strength = 0.7, seed=42, model = model)
# contentdir = "./images/" + "style.jpg"
# styledir = "./images/" + "style.jpg"
# main(prompt='', content_dir=contentdir, style_dir=styledir, ddim_steps=50, strength=0.7,
#      seed=42, model=model)
# 加载预训练的VGG16模型，去除全连接层，并使用最新的参数命名
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features  # 或使用 IMAGENET1K_V1
vgg16.eval()
# 将模型转移到GPU上（如果有的话），并设置为评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)  # 设置为评估模式
transforms_2 = [transforms.Resize(int(512 * 1.12), transforms.InterpolationMode.BICUBIC),  # 调整输入图片的大小
                    #transforms.RandomCrop(opt.size),  # 随机裁剪256*256
                    transforms.Resize(512),
                    # transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 归一化，这两行不能颠倒顺序呢，归一化需要用到tensor型

dataloader = DataLoader(ImageDataset('datasets/edges2shoes', transforms_2=transforms_2, unaligned=True),  #数据集
                            batch_size=2, shuffle=False, num_workers=0, drop_last=True)





model_me = MyModel(device).to(device)
for param in model.parameters():
    if not param.requires_grad:
        param.requires_grad = True
    if param.dtype == torch.float16:
        param.data = param.data.float()
model_me.train()
# 假设 'layer_name' 是你想要更新的层的名称
optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-4)
for group in optimizer_model.param_groups:
    for param in group['params']:
        param.data = param.data.float()  # 确保优化器中的参数是float32
optimizer_me = torch.optim.Adam(model_me.parameters(), lr=1e-4)
for group in optimizer_me.param_groups:
    for param in group['params']:
        param.data = param.data.float()  # 确保优化器中的参数是float32
n_epoch = 10
for i in range(n_epoch):
    folder_path = './Images/fuzhuang'
    all_items = os.listdir(folder_path)
    changdu = len(all_items)
    tag = 0
    for batch in dataloader:
        neirong_images = batch['neirong'].to(device)
        fengge_images = batch['fengge'].to(device)   #1 3 512 512
        content_names = batch['content_name']
        style_names = batch['style_name']
        # 使用列表推导式移除 '.jpg' 并合并为单一字符串
        content_names = ''.join([file_name[:-4] for file_name in content_names])
        style_names = ''.join([file_name[:-4] for file_name in style_names])
        neirong_features = vgg16(neirong_images).float()   #内容特征
        fengge_features = vgg16(fengge_images).float()     #风格特征
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
        neirong_features_flat = neirong_features.view(neirong_features.size(0), -1, neirong_features.size(2) * neirong_features.size(3)).to(device)
        fengge_features_flat = fengge_features.view(fengge_features.size(0), -1, fengge_features.size(2) * fengge_features.size(3)).to(device)
        # print(neirong_features_flat.size(1), fengge_features_flat.size(1))
        # 初始化交叉注意力模块
        #[512]

        # print(neirong_features_flat.shape, fengge_features_flat.shape)
        # 应用交叉注意力
        output_features = model_me(neirong_features_flat.float(),fengge_features_flat.float())  #交叉注意力
        # output_features = model_me2(fengge_features_flat)
        neirong_images.requires_grad_(True)
        fengge_images.requires_grad_(True)
        output,loss_diff = main(tag ,style_names=style_names, content_names=content_names, prompt='*', content_image=neirong_images.float(),
                      style_image=fengge_images.float(), c_feature=output_features.float(), ddim_steps=10, strength=0.1, model=model)
        # print("grid_feature", output.requires_grad)
        # output_f = vgg16(output.float().to(device))
        # output_f.requires_grad_(True)
        # loss_nce1 = contrastive_loss(output_f,neirong_features,fengge_features)
        # loss_nce2 = contrastive_loss(output_f, fengge_features, neirong_features)
        # loss_all = loss_nce1 + loss_nce2 + loss_diff
        loss_all = loss_diff.float()
        # print(f"Loss requires_grad: {loss_all.requires_grad}")
        # print(f"Loss_all has grad_fn: {loss_all.grad_fn is not None}")
        # if loss_all.grad_fn is not None:
        #     print(f"Grad_fn name: {loss_all.grad_fn.__class__.__name__}")
        # print('loss_nce1:',loss_nce1, 'loss_nce1:', loss_nce2,'loss_diff:', loss_diff)
        # logging.warning('loss_nce1:',loss_nce1, 'loss_nce1:', loss_nce2,'loss_diff:', loss_diff)
        with open('loss.txt', 'a') as f:
            f.write(f'Epoch {i}: loss_diff={loss_all}\n')
            # f.write(f'Epoch {i}: loss_nce1 = {loss_nce1},loss_nce2={loss_nce2},loss_diff={loss_diff}\n')
        # 清除梯度
        optimizer_model.zero_grad()
        optimizer_me.zero_grad()
        loss_all.backward()
        # 更新模型参数
        optimizer_model.step()  # 更新 main 函数中的模型参数
        optimizer_me.step()    # 更新交叉注意力模型参数
        tag += 1
        if tag >= changdu:
            break;

    if i % 10 == 0 :
        torch.save(model_me.state_dict(), 'outputs/model_me.pth')
        torch.save(model.state_dict(), 'outputs/model.pth')