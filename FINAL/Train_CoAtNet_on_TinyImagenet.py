
#region Import
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset 
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchvision.transforms.autoaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import math
import numpy as np
from tqdm import tqdm
import os
import copy
import glob 
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image as ToPILImage
from PIL import Image, ImageFile
from einops import rearrange
from einops.layers.torch import Rearrange
import warnings
warnings.filterwarnings("ignore")
#endregion

#region Setup device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

num_classes = 200
num_workers = 2
#endregion

#region 1. KONFIGURASI HYPERPARAMETER
CONFIG = {
    # Hyperparameter Preprocessing dan Augmentasi Data
    "center_crop": True,                     # Crop gambar dari tengah
    "randaugment_n": 2,                      # Jumlah operasi RandAugment
    "randaugment_m": 15,                     # Magnitude RandAugment
    "mixup_alpha": 0.8,                      # Parameter alpha untuk Mixup
    
    # Hyperparameter Fungsi Loss
    "loss_type": "Softmax",                  # Jenis fungsi loss
    "label_smoothing": 0.1,                  # Smoothing factor untuk label
    
    # Hyperparameter Training
    "train_epochs": 300,                     # Jumlah epoch
    "train_batch_size": 20,                # Ukuran batch
    
    # Hyperparameter Optimisasi
    "optimizer_type": "AdamW",               # Jenis optimizer
    "peak_lr": 1e-3,                         # Learning rate maksimum
    "min_lr": 1e-5,                          # Learning rate minimum
    "warmup_steps": 10000,                   # Jumlah langkah warm-up
    "lr_decay_schedule": "Cosine",           # Jadwal penurunan learning rate
    "weight_decay_rate": 0.05,               # Rate weight decay
    "gradient_clip": 1.0,                    # Nilai maksimum gradien
    "ema_decay_rate": None                   # EMA dinonaktifkan
}
#endregion

#region 2. DEFINISI ARSITEKTUR MODEL (CoAtNet)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEfficientSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return MemoryEfficientSwish.apply(x)


class MemoryEfficientMish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        v = 1. + i.exp()
        h = v.log() 
        grad_gh = 1./h.cosh().pow_(2) 
        grad_hx = i.sigmoid()
        grad_gx = grad_gh *  grad_hx
        grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx 
        return grad_output * grad_f 


class Mish(nn.Module):
    def forward(self, x):
        return MemoryEfficientMish.apply(x)



import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

act_fn_map = {
    'swish': 'silu'
}

memory_efficient_map = {
    'swish': Swish,
    'mish': Mish
}


def get_act_fn(act_fn, prefer_memory_efficient=True):
    if isinstance(act_fn, str):
        if prefer_memory_efficient and act_fn in memory_efficient_map:
            return memory_efficient_map[act_fn]()
        if act_fn in act_fn_map:
            act_fn = act_fn_map[act_fn]
        return getattr(F, act_fn)
    return act_fn


def drop_connect(x, drop_ratio, training=True):
    if not training or drop_ratio == 0:
        return x
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules:
            print("{}: {}".format(n, num_params))
        total_num_params += num_params
    print("-" * 50)
    print("Total number of parameters: {}".format(total_num_params))


def weight_init(m):
    '''
    Usage:
        model.apply(weight_init)
    '''
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        init.normal_(m.weight)
        if m.bias is not None:
            init.normal_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param)
            else:
                init.normal_(param)



import torch.nn as nn
import math


class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, ksize, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=ksize, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            half_pad_h = pad_h >> 1
            half_pad_w = pad_w >> 1
            self.static_padding = nn.ZeroPad2d((half_pad_w, pad_w - half_pad_w, half_pad_h, pad_h - half_pad_h))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, bias=True, momentum=0.1, eps=1e-5, act_fn=None, static_padding=False, image_size=None):
        super().__init__()
        if static_padding:
            self.conv = Conv2dStaticSamePadding(in_channels, out_channels, ksize, image_size=image_size, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum, eps=eps)
        self.act_fn = get_act_fn(act_fn)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


class DepthwiseConv2dBlock(nn.Module):
    def __init__(self, in_channels, ksize, stride=1, bias=False, momentum=0.1, eps=1e-5, act_fn=None, static_padding=False, image_size=None):
        super().__init__()
        if static_padding:
            self.conv = Conv2dStaticSamePadding(in_channels, in_channels, ksize, image_size=image_size, stride=stride, groups=in_channels, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=ksize, stride=stride, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum, eps=eps)
        self.act_fn = get_act_fn(act_fn)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x
    


import torch.nn as nn
import numpy as np


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25, bias=True, act_fn='mish', static_padding=False, image_size=None):
        super().__init__()
        squeezed_channels = max(int(in_channels * se_ratio), 1)
        if static_padding:
            self.reduce = Conv2dStaticSamePadding(in_channels, squeezed_channels, 1, image_size=image_size, bias=bias)
        else:
            self.reduce = nn.Conv2d(in_channels, squeezed_channels, kernel_size=1, bias=bias)
        self.act_fn = get_act_fn(act_fn)
        if static_padding:
            self.expand = Conv2dStaticSamePadding(squeezed_channels, in_channels, 1, image_size=image_size, bias=bias)
        else:
            self.expand = nn.Conv2d(squeezed_channels, in_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self.reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.expand(x_se)
        x = x * x_se.sigmoid()
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 momentum=0.1, eps=1e-5, ksize=3,
                 stride=1, expand_ratio=1, se_ratio=0.25,
                 drop_ratio=0.2, act_fn='mish', image_size=224):
        super().__init__()
        self.skip_connect = stride == 1 and in_channels == out_channels
        self.drop_ratio = drop_ratio
        self.act_fn = get_act_fn(act_fn)
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        expand_channels = in_channels * expand_ratio
        if not isinstance(image_size, int):
            image_size = np.array(image_size)
        if expand_ratio != 1:
            self.expand_conv = Conv2dBlock(in_channels, expand_channels, 1, bias=False, momentum=momentum, eps=eps, act_fn=self.act_fn, static_padding=True, image_size=image_size)
        self.depthwise_conv = DepthwiseConv2dBlock(expand_channels, ksize, stride=stride, momentum=momentum, eps=eps, act_fn=self.act_fn, static_padding=True, image_size=image_size)
        image_size = np.ceil(image_size / stride).astype(int)
        if se_ratio is not None:
            self.se = SqueezeExcitation(expand_channels, se_ratio=se_ratio, act_fn=self.act_fn, static_padding=True, image_size=image_size)
        self.project_conv = Conv2dBlock(expand_channels, out_channels, 1, bias=False, momentum=momentum, eps=eps, static_padding=True, image_size=image_size)

    def forward(self, x):
        x_in = x
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.se_ratio is not None:
            x = self.se(x)
        x = self.project_conv(x)
        if self.skip_connect:
            x = drop_connect(x, self.drop_ratio, training=self.training) + x_in
        return x


class MBConvForRelativeAttention(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, out_channels,
                 momentum=0.1, eps=1e-5, ksize=3, expand_ratio=1,
                 se_ratio=0.25, drop_ratio=0.1, act_fn='mish',
                 use_downsampling=False, **kwargs):
        super().__init__()
        self.use_downsampling = use_downsampling
        self.drop_ratio = drop_ratio
        self.norm = nn.BatchNorm2d(in_channels)
        self.mbconv = MBConv(in_channels, out_channels, momentum=momentum, eps=eps,
                             ksize=ksize, stride=2 if use_downsampling else 1,
                             expand_ratio=expand_ratio, se_ratio=se_ratio,
                             drop_ratio=drop_ratio, act_fn=act_fn, image_size=(inp_h, inp_w))
        if use_downsampling:
            self.pool = nn.MaxPool2d((2, 2))
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.use_downsampling:
            x_downsample = self.pool(x)
            x_downsample = self.conv(x_downsample)
        else:
            x_downsample = x
        x = self.norm(x)
        x = self.mbconv(x)
        x = drop_connect(x, self.drop_ratio, training=self.training)
        x = x_downsample + x
        return x
      


import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeAttention(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, n_head, d_k, d_v, out_channels, attn_dropout=0.1, ff_dropout=0.1, attn_bias=False):
        super().__init__()
        self.inp_h = inp_h
        self.inp_w = inp_w
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.Q = nn.Linear(in_channels, n_head * d_k, bias=attn_bias)
        self.K = nn.Linear(in_channels, n_head * d_k, bias=attn_bias)
        self.V = nn.Linear(in_channels, n_head * d_v, bias=attn_bias)
        self.ff = nn.Linear(n_head * d_v, out_channels)
        self.attn_dropout = nn.Dropout2d(attn_dropout)
        self.ff_dropout = nn.Dropout(ff_dropout)
        self.relative_bias = nn.Parameter(
            torch.randn(n_head, ((inp_h << 1) - 1) * ((inp_w << 1) - 1)),
            requires_grad=True
        )
        self.register_buffer('relative_indices', self._get_relative_indices(inp_h, inp_w))

    def _get_relative_indices(self, height, width):
        ticks_y, ticks_x = torch.arange(height), torch.arange(width)
        grid_y, grid_x = torch.meshgrid(ticks_y, ticks_x)
        area = height * width
        out = torch.empty(area, area).fill_(float('nan'))
        for idx_y in range(height):
            for idx_x in range(width):
                rel_indices_y = grid_y - idx_y + height
                rel_indices_x = grid_x - idx_x + width
                flatten_indices = (rel_indices_y * width + rel_indices_x).view(-1)
                out[idx_y * width + idx_x] = flatten_indices
        assert not out.isnan().any(), '`relative_indices` have blank indices'
        assert (out >= 0).all(), '`relative_indices` have negative indices'
        return out.long()

    def _interpolate_relative_bias(self, height, width):
        relative_bias = self.relative_bias.view(1, self.n_head, (self.inp_h << 1) - 1, -1)
        relative_bias = F.interpolate(relative_bias, size=((height << 1) - 1, (width << 1) - 1), mode='bilinear', align_corners=True)
        return relative_bias.view(self.n_head, -1)

    def update_relative_bias_and_indices(self, height, width):
        self.relative_indices = self._get_relative_indices(height, width)
        self.relative_bias = self._interpolate_relative_bias(height, width)

    def forward(self, x):
        b, c, H, W, h = *x.shape, self.n_head
    
        len_x = H * W
        x = x.view(b, c, len_x).transpose(-1, -2)
        q = self.Q(x).view(b, len_x, self.n_head, self.d_k).transpose(1, 2)
        k = self.K(x).view(b, len_x, self.n_head, self.d_k).transpose(1, 2)
        v = self.V(x).view(b, len_x, self.n_head, self.d_v).transpose(1, 2)

        if H == self.inp_h and W == self.inp_w:
            relative_indices = self.relative_indices
            relative_bias = self.relative_bias
        else:
            relative_indices = self._get_relative_indices(H, W).to(x.device)
            relative_bias = self._interpolate_relative_bias(H, W)

        relative_indices = relative_indices.view(1, 1, *relative_indices.size()).expand(b, h, -1, -1)
        relative_bias = relative_bias.view(1, relative_bias.size(0), 1, relative_bias.size(1)).expand(b, -1, len_x, -1)
        relative_biases = relative_bias.gather(dim=-1, index=relative_indices)

        similarity = torch.matmul(q, k.transpose(-1, -2)) + relative_biases
        similarity = similarity.softmax(dim=-1)
        similarity = self.attn_dropout(similarity)
        
        out = torch.matmul(similarity, v)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.n_head * self.d_v)
        out = self.ff(out)
        out = self.ff_dropout(out)
        out = out.transpose(-1, -2).view(b, -1, H, W)
        return out


class FeedForwardRelativeAttention(nn.Module):
    def __init__(self, in_dim, expand_dim, drop_ratio=0.1, act_fn='gelu'):
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, expand_dim, kernel_size=1)
        self.act_fn = get_act_fn(act_fn)
        self.fc2 = nn.Conv2d(expand_dim, in_dim, kernel_size=1)
        self.drop_ratio = drop_ratio

    def forward(self, x):
        x_in = x
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = drop_connect(x, drop_ratio=self.drop_ratio, training=self.training) + x_in
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn='mish', ff_dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.act_fn = get_act_fn(act_fn)
        self.dropout = nn.Dropout(ff_dropout)
        self.norm = nn.LayerNorm(in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class TransformerWithRelativeAttention(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, n_head, d_k=None, d_v=None,
                 out_channels=None, attn_dropout=0.1, ff_dropout=0.1,
                 act_fn='gelu', attn_bias=False, expand_ratio=4,
                 use_downsampling=False, **kwargs):
        super().__init__()
        self.use_downsampling = use_downsampling
        self.dropout = ff_dropout
        out_channels = out_channels or in_channels
        d_k = d_k or out_channels // n_head
        d_v = d_v or out_channels // n_head
        if use_downsampling:
            self.pool = nn.MaxPool2d((2, 2))
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(in_channels)
        self.attention = RelativeAttention(inp_h, inp_w, in_channels, n_head, d_k, d_v, out_channels, attn_dropout=attn_dropout, ff_dropout=ff_dropout, attn_bias=attn_bias)
        self.ff = FeedForwardRelativeAttention(out_channels, out_channels * expand_ratio, drop_ratio=ff_dropout, act_fn=act_fn)

    def forward(self, x):
        if self.use_downsampling:
            x_stem = self.pool(x)
            x_stem = self.conv(x_stem)
        else:
            x_stem = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        if self.use_downsampling:
            x = self.pool(x)
        x = self.attention(x)
        x = drop_connect(x, self.dropout, training=self.training)
        x = x_stem + x
        x_attn = x
        x = self.ff(x)
        x = drop_connect(x, self.dropout, training=self.training)
        x = x_attn + x
        return x
    

import torch.nn as nn
import torch.nn.functional as F

configs = {
    'coatnet-0': {
        'num_blocks': [2, 2, 3, 5, 2],
        'num_channels': [64, 96, 192, 384, 768],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-1': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [64, 96, 192, 384, 768],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-2': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [128, 128, 256, 512, 1026],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-3': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [192, 192, 384, 768, 1536],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-4': {
        'num_blocks': [2, 2, 12, 28, 2],
        'num_channels': [192, 192, 384, 768, 1536],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-5': {
        'num_blocks': [2, 2, 12, 28, 2],
        'num_channels': [192, 256, 512, 1280, 2048],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 64,
        'block_types': ['C', 'C', 'T', 'T']
    },
    # Something's not right with this one
    # 'coatnet-6': {
    #     'num_blocks': [2, 2, 4, [8, 42], 2],
    #     'num_channels': [192, 192, 384, [768, 1536], 2048],
    #     'expand_ratio': [4, 4, 4, 4, 4],
    #     'n_head': 128,
    #     'block_types': ['C', 'C', 'C-T', 'T']
    # },
    # 'coatnet-7': {
    #     'num_blocks': [2, 2, 4, [8, 42], 2],
    #     'num_channels': [192, 256, 512, [1024, 2048], 3072],
    #     'expand_ratio': [4, 4, 4, 4, 4],
    #     'n_head': 128,
    #     'block_types': ['C', 'C', 'C-T', 'T']
    # }
}

blocks = {
    'C': MBConvForRelativeAttention,
    'T': TransformerWithRelativeAttention
}


class CoAtNet(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, config='coatnet-0', num_classes=None, head_act_fn='mish', head_dropout=0.1):
        super().__init__()
        self.config = configs[config]
        block_types = self.config['block_types']
        self.s0 = self._make_stem(in_channels)
        self.s1 = self._make_block(block_types[0], inp_h >> 2, inp_w >> 2,
                                   self.config['num_channels'][0],
                                   self.config['num_channels'][1],
                                   self.config['num_blocks'][1],
                                   self.config['expand_ratio'][0])
        self.s2 = self._make_block(block_types[1], inp_h >> 3, inp_w >> 3,
                                   self.config['num_channels'][1],
                                   self.config['num_channels'][2],
                                   self.config['num_blocks'][2],
                                   self.config['expand_ratio'][1])
        self.s3 = self._make_block(block_types[2], inp_h >> 4, inp_w >> 4,
                                   self.config['num_channels'][2],
                                   self.config['num_channels'][3],
                                   self.config['num_blocks'][3],
                                   self.config['expand_ratio'][2])
        self.s4 = self._make_block(block_types[3], inp_h >> 5, inp_w >> 5,
                                   self.config['num_channels'][3],
                                   self.config['num_channels'][4],
                                   self.config['num_blocks'][4],
                                   self.config['expand_ratio'][3])
        self.include_head = num_classes is not None
        if self.include_head:
            if isinstance(num_classes, int):
                self.single_head = True
                num_classes = [num_classes]
            else:
                self.single_head = False
            self.heads = nn.ModuleList([ProjectionHead(self.config['num_channels'][-1], nc, act_fn=head_act_fn, ff_dropout=head_dropout) for nc in num_classes])

    def _make_stem(self, in_channels):
        return nn.Sequential(*[
            nn.Conv2d(
                in_channels if i == 0 else self.config['num_channels'][0],
                self.config['num_channels'][0], kernel_size=3, padding=1,
                stride=2 if i == 0 else 1
            ) for i in range(self.config['num_blocks'][0])
        ])

    def _make_block(self, block_type, inp_h, inp_w, in_channels, out_channels, depth, expand_ratio):
        block_list = []
        if not isinstance(in_channels, int):
            in_channels = in_channels[-1]
        if block_type in blocks:
            block_cls = blocks[block_type]
            block_list.extend([
                block_cls(
                    inp_h, inp_w, in_channels if i == 0 else out_channels,
                    n_head=self.config['n_head'], out_channels=out_channels,
                    expand_ratio=expand_ratio, use_downsampling=i == 0
                ) for i in range(depth)
            ])
        else:
            for i, _block_type in enumerate(block_type.split('-')):
                block_cls = blocks[_block_type]
                block_list.extend(
                    block_cls(
                        inp_h, inp_w, in_channels if i == 0 and j == 0 else out_channels[i - 1] if j == 0 else out_channels[i],
                        n_head=self.config['n_head'], out_channels=out_channels[i],
                        expand_ratio=expand_ratio, use_downsampling=j == 0
                    ) for j in range(depth[i])
                )
        return nn.Sequential(*block_list)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        if self.include_head:
            if self.single_head:
                return self.heads[0](x)
            return [head(x) for head in self.heads]
        return x

#endregion

#region 3. KONFIGURASI DATA PIPELINE
train_transform = transforms.Compose([
    RandAugment(num_ops=CONFIG['randaugment_n'], magnitude=CONFIG['randaugment_m']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = glob.glob("/home/tasi2425111/resized-tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform

        self.class_to_idx = {}
        class_folders = set([path.split('/')[5] for path in self.filenames])
        for i, class_name in enumerate(sorted(class_folders)):
            self.class_to_idx[class_name] = i

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)

        class_name = img_path.split('/')[5]
        label = self.class_to_idx[class_name]

        image = ToPILImage(image)
        if self.transform:
            image = self.transform(image)
        return image, label

class ValTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = glob.glob("/home/tasi2425111/resized-tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform

        train_files = glob.glob("/home/tasi2425111/resized-tiny-imagenet-200/train/*/*/*.JPEG")
        class_folders = set([path.split('/')[5] for path in train_files])
        self.class_to_idx = {}
        for i, class_name in enumerate(sorted(class_folders)):
            self.class_to_idx[class_name] = i

        self.cls_dic = {}
        for i, line in enumerate(open('/home/tasi2425111/resized-tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.class_to_idx[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)

        label = self.cls_dic[img_path.split('/')[-1]]
        image = ToPILImage(image)
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = TrainTinyImageNetDataset(transform = train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG["train_batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True)

val_dataset = ValTinyImageNetDataset(transform = val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG["train_batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)

#endregion

#region 4. KONFIGURASI MIXUP
def get_mixup_fn():
    mixup_fn = Mixup(
        mixup_alpha=CONFIG["mixup_alpha"],
        label_smoothing=CONFIG["label_smoothing"],
        num_classes=num_classes  # Sesuaikan dengan dataset Anda
    )
    return mixup_fn
#endregion

#region 5. KONFIGURASI LOSS FUNCTION
def get_loss_fn():
    if CONFIG["loss_type"] == "Softmax":
        # CrossEntropyLoss sudah mencakup softmax
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    else:
        # Tambahkan loss function lain jika diperlukan
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    
    return criterion
#endregion

#region 6. KONFIGURASI OPTIMIZER
def get_optimizer(model):
    if CONFIG["optimizer_type"] == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["peak_lr"],
            weight_decay=CONFIG["weight_decay_rate"]
        )
    else:
        # Fallback ke Adam jika AdamW tidak tersedia
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG["peak_lr"],
            weight_decay=CONFIG["weight_decay_rate"]
        )
    
    return optimizer
#endregion

#region 7. KONFIGURASI LEARNING RATE SCHEDULER
def get_lr_scheduler(optimizer, num_training_steps):
    if CONFIG["lr_decay_schedule"] == "Cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_training_steps,  # Total langkah training
            lr_min=CONFIG["min_lr"],
            warmup_t=CONFIG["warmup_steps"],
            warmup_lr_init=CONFIG["min_lr"],
            cycle_limit=1
        )
    else:
        # Fallback ke StepLR
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return lr_scheduler
#endregion

#region 8. FUNGSI HELPER UNTUK EMA (EXPONENTIAL MOVING AVERAGE)
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        
        # Fix EMA. https://github.com/pytorch/vision/pull/2591
        for param in self.ema.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            
            for k, v in esd.items():
                if needs_module:
                    k = 'module.' + k
                
                model_v = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + (1. - self.decay) * model_v)
                else:
                    v.copy_(model_v)
#endregion

#region 9. FUNGSI TRAINING
def train_model():    
    # Get model using coatnet_3 function

    image_size = (3, 224, 224)
    config=f'coatnet-3'
    model = CoAtNet(image_size[1], image_size[2], image_size[0], config=config, num_classes=num_classes)
    
    # Disable if start from scratch
    # checkpoint = torch.load("/home/tasi2425111/for_hpc/baru/ti_co/7_continue_fr_6/best_model.pth", map_location=device)
    # model.load_state_dict(checkpoint["state_dict"])

    model = model.to(device)
    
    mixup_fn = get_mixup_fn()
    criterion = get_loss_fn()
    optimizer = get_optimizer(model)
    
    # Calculate total number of training steps
    total_steps = len(train_loader) * CONFIG["train_epochs"]
    lr_scheduler = get_lr_scheduler(optimizer, total_steps)
    
    # Setup EMA if enabled
    ema_model = None
    if CONFIG["ema_decay_rate"] is not None:
        ema_model = ModelEMA(model, decay=CONFIG["ema_decay_rate"])
    
    # Setup gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Early stopping variables
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    # Lists to store metrics for summary
    val_losses = []
    val_accs = []
    
    # Training loop
    for epoch in range(CONFIG["train_epochs"]):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['train_epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply mixup if enabled
            if mixup_fn is not None:
                inputs, targets = mixup_fn(inputs, targets)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if CONFIG["gradient_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA model if enabled
            if ema_model is not None:
                ema_model.update(model)
            
            # Update learning rate
            lr_scheduler.step_update(epoch * len(train_loader) + batch_idx)
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            if isinstance(targets, torch.Tensor) and targets.dim() == 1:
                correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': train_loss/(batch_idx+1), 
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Validate after each epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Use EMA model for validation if available
                if ema_model is not None:
                    outputs = ema_model.ema(inputs)
                else:
                    outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1} - Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Check if this is the best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            
            # Save best model state
            if ema_model is not None:
                best_model_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                    'val_loss': epoch_val_loss,
                    'val_acc': epoch_val_acc,
                }
            else:
                best_model_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                    'val_loss': epoch_val_loss,
                    'val_acc': epoch_val_acc,
                }
                
            # Save the best model checkpoint
            torch.save(best_model_state, 'best_model.pth')
            print(f"Saved new best model with validation loss: {epoch_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
        # Check if early stopping criteria is met
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
        'ema': ema_model.ema.state_dict() if ema_model is not None else None,
        'val_acc': epoch_val_acc,
    }
    
    torch.save(checkpoint, 'final_model.pth')
    
    # Print training summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total epochs: {epoch+1}")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_model_state['epoch']})")
    print(f"Final metrics:")
    print(f"  Validation Loss: {val_losses[-1]:.4f}")
    print(f"  Validation Accuracy: {val_accs[-1]:.2f}%")
    print("="*50)
#endregion

train_model()