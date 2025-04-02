num_classes = 200


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


#region 9. FUNGSI TRAINING
#@title 9. FUNGSI TRAINING

""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""

#region Import
import argparse
import copy
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')
#endregion

#region Hyperparameter

class Args:
    ### CUSTOM ###
    # Load this checkpoint as if they were the pretrained weights (with adaptation) (default: None).
    pretrained_path = None
    # Resume full model and optimizer state from checkpoint (default: '')
    resume = ''
    # path to dataset (root dir)
    data_dir = '/home/tasi2425111/restructured-resized-tiny-imagenet-200'  #Disesuaikan dengan kebutuhan
    # number of label classes (Model default if None)
    num_classes = 200  #Disesuaikan dengan kebutuhan
    # Name of model to train (default: "resnet50")
    model = 'coatnet_3' #Coatnet_3  #Disesuaikan dengan kebutuhan
    # Device (accelerator) to use.
    device = 'cuda:0'
    # Input image center crop percent (for validation only)
    crop_pct = None ## Tidak diikutkan karena sudah diresize
    # Use AutoAugment policy. "v0" or "original". (default: None)
    aa = 'rand-m15-n2' ## Operation = 2 , Magnitude = 15
    # mixup alpha, mixup enabled if > 0. (default: 0.)
    mixup = 0.8
    #loss_type = Softmax #Sudah default di code training
    # Label smoothing (default: 0.1)
    smoothing = 0.1
    # number of epochs to train (default: 300)
    epochs = 300
    # Input batch size for training (default: 128)
    batch_size = 20
    # Validation batch size override (default: None)
    validation_batch_size = 20
    # Optimizer (default: "sgd")
    opt = 'AdamW'
    # learning rate, overrides lr-base if set (default: None)
    lr = 1e-3
    # lower lr bound for cyclic schedulers that hit 0 (default: 0)
    min_lr = 1e-5
    # Learning rate scheduler (default: "cosine")
    sched = 'cosine'
    # epochs to warmup LR, if scheduler supports
    warmup_epochs = 10000
    # weight decay (default: 2e-5)
    weight_decay = 0.05
    # Clip gradient norm (default: None, no clipping)
    clip_grad = 1.0
    # Decay factor for model weights moving average (default: 0.9998)
    model_ema_decay = None
    # Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty
    input_size = (3, 224, 224)
    # Override mean pixel value of dataset
    mean = (0.485, 0.456, 0.406)
    # Override std deviation of dataset
    std = (0.229, 0.224, 0.225)


    ### DEFAULT ###
    # path to dataset (positional is *deprecated*, use --data-dir)
    data = None
    # dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)
    dataset = ''
    # dataset train split (default: train)
    train_split = 'train'
    # dataset validation split (default: validation)
    val_split = 'validation'
    # Manually specify num samples in train split, for IterableDatasets.
    train_num_samples = None
    # Manually specify num samples in validation split, for IterableDatasets.
    val_num_samples = None
    # Allow download of dataset for torch/ and tfds/ datasets that support it.
    dataset_download = False
    # path to class to idx mapping file (default: "")
    class_map = ''
    # Dataset image conversion mode for input images.
    input_img_mode = None
    # Dataset key for input images.
    input_key = None
    # Dataset key for target labels.
    target_key = None
    # Allow huggingface dataset import to execute code downloaded from the dataset's repo.
    dataset_trust_remote_code = False
    # Start with pretrained version of specified network (if avail)
    pretrained = False
    # Load this checkpoint into model after initialization (default: none)
    initial_checkpoint = ''
    # prevent resume of optimizer state when resuming model
    no_resume_opt = False
    # Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.
    gp = None
    # Image size (default: None => model default)
    img_size = None
    # Image input channels (default: None => 3)
    in_chans = None
    # Image resize interpolation type (overrides model)
    interpolation = ''
    # Use channels_last memory layout
    channels_last = False
    # Select jit fuser. One of ('', 'te', 'old', 'nvfuser')
    fuser = ''
    # The number of steps to accumulate gradients (default: 1)
    grad_accum_steps = 1
    # Enable gradient checkpointing through model blocks/stages
    grad_checkpointing = False
    # enable experimental fast-norm
    fast_norm = False
    # Model kwargs
    model_kwargs = {}
    # Head initialization scale
    head_init_scale = None
    # Head initialization bias value
    head_init_bias = None
    # torch.compile mode (default: None).
    torchcompile_mode = None
    # torch.jit.script the full model , action='store_true'
    torchscript = False
    # Enable compilation w/ specified backend (default: inductor). const='inductor'
    torchcompile = None
    # use NVIDIA Apex AMP or Native AMP for mixed precision training
    amp = False
    # lower precision AMP dtype (default: float16)
    amp_dtype = 'float16'
    # AMP impl to use, "native" or "apex" (default: native)
    amp_impl = 'native'
    # Model dtype override (non-AMP) (default: float32)
    model_dtype = None
    # Force broadcast buffers for native DDP to off.
    no_ddp_bb = False
    # torch.cuda.synchronize() end of each step
    synchronize_step = False
    # Local rank for distributed training
    local_rank = 0
    # Python imports for device backend modules.
    device_modules = None
    # Optimizer Epsilon (default: None, use opt default)
    opt_eps = None
    # Optimizer Betas (default: None, use opt default)
    opt_betas = None
    # Optimizer momentum (default: 0.9)
    momentum = 0.9
    # Gradient clipping mode. One of ("norm", "value", "agc")
    clip_mode = 'norm'
    # layer-wise learning rate decay (default: None)
    layer_decay = None
    # action=utils.ParseKwargs
    opt_kwargs = {}
    # Apply LR scheduler step on update instead of epoch end.
    sched_on_updates = False
    # base learning rate: lr = lr_base * global_batch_size / base_size
    lr_base = 0.1
    # base learning rate batch size (divisor, default: 256).
    lr_base_size = 256
    # base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)
    lr_base_scale = ''
    # learning rate noise on/off epoch percentages
    lr_noise = None
    # learning rate noise limit percent (default: 0.67)
    lr_noise_pct = 0.67
    # learning rate noise std-dev (default: 1.0)
    lr_noise_std = 1.0
    # learning rate cycle len multiplier (default: 1.0)
    lr_cycle_mul = 1.0
    # amount to decay each learning rate cycle (default: 0.5)
    lr_cycle_decay = 0.5
    # learning rate cycle limit, cycles enabled if > 1
    lr_cycle_limit = 1
    # learning rate k-decay for cosine/poly (default: 1.0)
    lr_k_decay = 1.0
    # warmup learning rate (default: 1e-5)
    warmup_lr = 1e-5
    # epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).
    epoch_repeats = 0.
    # manual epoch number (useful on restarts)
    start_epoch = None
    # list of decay epoch indices for multistep lr. must be increasing
    decay_milestones = [90, 180, 270]
    # epoch interval to decay LR
    decay_epochs = 90
    # Exclude warmup period from decay schedule.
    warmup_prefix = False
    # epochs to cooldown LR at min_lr, after cyclic schedule ends
    cooldown_epochs = 0
    # patience epochs for Plateau LR scheduler (default: 10)
    patience_epochs = 10
    # LR decay rate (default: 0.1)
    decay_rate = 0.1
    # Disable all training augmentation, override other train aug args
    no_aug = False
    # Crop-mode in train
    train_crop_mode = None
    # Random resize scale (default: 0.08 1.0)
    scale = [0.08, 1.0]
    # Random resize aspect ratio (default: 0.75 1.33)
    ratio = [3. / 4., 4. / 3.]
    # Horizontal flip training aug probability
    hflip = 0.5
    # Vertical flip training aug probability
    vflip = 0.
    # Color jitter factor (default: 0.4)
    color_jitter = 0.4
    # Probability of applying any color jitter.
    color_jitter_prob = None
    # Probability of applying random grayscale conversion.
    grayscale_prob = None
    # Probability of applying gaussian blur.
    gaussian_blur_prob = None
    # Number of augmentation repetitions (distributed training only) (default: 0)
    aug_repeats = 0
    # Number of augmentation splits (default: 0, valid: 0 or >=2)
    aug_splits = 0
    # Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.
    jsd_loss = False
    # Enable BCE loss w/ Mixup/CutMix use.
    bce_loss = False
    # Sum over classes when using BCE loss.
    bce_sum = False
    # Threshold for binarizing softened BCE targets (default: None, disabled).
    bce_target_thresh = None
    # Positive weighting for BCE loss.
    bce_pos_weight = None
    # Random erase prob (default: 0.)
    reprob = 0.
    # Random erase mode (default: "pixel")
    remode = 'pixel'
    # Random erase count (default: 1)
    recount = 1
    # Do not random erase first (clean) augmentation split
    resplit = False
    # cutmix alpha, cutmix enabled if > 0. (default: 0.)
    cutmix = 0.0
    # cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
    cutmix_minmax = None
    # Probability of performing mixup or cutmix when either/both is enabled
    mixup_prob = 1.0
    # Probability of switching to cutmix when both mixup and cutmix enabled
    mixup_switch_prob = 0.5
    # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
    mixup_mode = 'batch'
    # Turn off mixup after this epoch, disabled if 0 (default: 0)
    mixup_off_epoch = 0
    # Training interpolation (random, bilinear, bicubic default: "random")
    train_interpolation = 'random'
    # Dropout rate (default: 0.)
    drop = 0.0
    # Drop connect rate, DEPRECATED, use drop-path (default: None)
    drop_connect = None
    # Drop path rate (default: None)
    drop_path = None
    # Drop block rate (default: None)
    drop_block = None
    # BatchNorm momentum override (if not None)
    bn_momentum = None
    # BatchNorm epsilon override (if not None)
    bn_eps = None
    # Enable NVIDIA Apex or Torch synchronized BatchNorm.
    sync_bn = False
    # Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")
    dist_bn = 'reduce'
    # Enable separate BN layers per augmentation split.
    split_bn = False
    # Enable tracking moving average of model weights.
    model_ema = False
    # Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.
    model_ema_force_cpu = False
    # Enable warmup for model EMA decay.
    model_ema_warmup = False
    # Random seed (default: 42)
    seed = 42
    # worker seed mode (default: all)
    worker_seeding = 'all'
    # how many batches to wait before logging training status
    log_interval = 50
    # how many batches to wait before writing recovery checkpoint
    recovery_interval = 0
    # number of checkpoints to keep (default: 10)
    checkpoint_hist = 10
    # how many training processes to use (default: 4)
    workers = 4
    # save images of input batches every log interval for debugging
    save_images = False
    # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
    pin_mem = False
    # disable fast prefetcher
    no_prefetcher = False
    # path to output folder (default: none, current dir)
    output = ''
    # name of train experiment, name of sub-folder for output
    experiment = ''
    # Best metric (default: "top1")
    eval_metric = 'top1'
    # Test/inference time augmentation (oversampling) factor. 0=None (default: 0)
    tta = 0
    # use the multi-epochs-loader to save time at the beginning of every epoch
    use_multi_epochs_loader = False
    # log training and validation metrics to wandb
    log_wandb = False
    # wandb project name
    wandb_project = None
    # wandb tags
    wandb_tags = []
    # If resuming a run, the id of the run in wandb
    wandb_resume_id = ''

#endregion

def main():
    utils.setup_default_logging()
    args  = Args()

    # model = globals()[args.model]()

    image_size = (3, 224, 224)
    config=f'coatnet-3'
    model = CoAtNet(image_size[1], image_size[2], image_size[0], config=config, num_classes=num_classes)

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)
        if model_dtype == torch.float16:
            _logger.warning('float16 is not recommended for training, for half precision bfloat16 is recommended.')

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32, 'float32 model dtype must be used with AMP'
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1,  # force head adaptation
        )

    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device, dtype=model_dtype)  # FIXME move model device & dtype into create_model
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )
    if utils.is_primary(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults['weight_decay'] = args.weight_decay  # this isn't stored in optimizer.defaults
        defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
        logging.info(
            f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
        )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type in ('cuda',) and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler(device=device.type)
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info(f'AMP not enabled. Training in {model_dtype or torch.float32}.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device='cpu' if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
        if args.torchcompile:
            model_ema = torch.compile(model_ema, backend=args.torchcompile)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile, mode=args.torchcompile_mode)

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.train_num_samples,
        trust_remote_code=args.dataset_trust_remote_code,
    )

    if args.val_split:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            input_img_mode=input_img_mode,
            input_key=args.input_key,
            target_key=args.target_key,
            num_samples=args.val_num_samples,
            trust_remote_code=args.dataset_trust_remote_code,
        )

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support de-interleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = None
    if args.val_split:
        eval_workers = args.workers
        if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            eval_workers = min(2, args.workers)
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=eval_workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            img_dtype=model_dtype or torch.float32,
            device=device,
            use_prefetcher=args.prefetcher,
        )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric if loader_eval is not None else 'loss'
    decreasing_metric = eval_metric == 'loss'
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist
        )
        # with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        #     f.write(args_text)

        if args.log_wandb:
            if has_wandb:
                assert not args.wandb_resume_id or args.resume
                wandb.init(
                    project=args.wandb_project,
                    name=exp_name,
                    config=args,
                    tags=args.wandb_tags,
                    resume="must" if args.wandb_resume_id else None,
                    id=args.wandb_resume_id if args.wandb_resume_id else None,
                )
            else:
                _logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        if args.warmup_prefix:
            sched_explain = '(warmup_epochs + epochs + cooldown_epochs). Warmup added to total when warmup_prefix=True'
        else:
            sched_explain = '(epochs + cooldown_epochs). Warmup within epochs when warmup_prefix=False'
        _logger.info(
            f'Scheduled epochs: {num_epochs} {sched_explain}. '
            f'LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_dtype=model_dtype,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                num_updates_total=num_epochs * updates_per_epoch,
            )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if loader_eval is not None:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                    model_dtype=model_dtype,
                )

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                    ema_eval_metrics = validate(
                        model_ema,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        device=device,
                        amp_autocast=amp_autocast,
                        log_suffix=' (EMA)',
                    )
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            else:
                latest_metric = train_metrics[eval_metric]

            if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, latest_metric)

            latest_results = {
                'epoch': epoch,
                'train': train_metrics,
            }
            if eval_metrics is not None:
                latest_results['validation'] = eval_metrics
            results.append(latest_results)

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        # log best metric as tracked by checkpoint saver
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    if utils.is_primary(args):
        # for parsable results display, dump top-10 summaries to avoid excess console spam
        display_results = sorted(
            results,
            key=lambda x: x.get('validation', x.get('train')).get(eval_metric, 0),
            reverse=decreasing_metric,
        )
        print(f'--result\n{json.dumps(display_results[-10:], indent=4)}')


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_dtype=None,
        model_ema=None,
        mixup_fn=None,
        num_updates_total=None,
  ):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    # -- Tambahan: rekam waktu mulai epoch
    epoch_start_time = time.time()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // args.grad_accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetcher:
            input, target = input.to(device=device, dtype=model_dtype), target.to(device=device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'npu':
                torch.npu.synchronize()

        time_now = time.time()
        update_time_m.update(time_now - update_start_time)
        update_start_time = time_now

        # -- Log progress, termasuk estimasi waktu
        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = losses_m.avg, losses_m.val
            if args.distributed:
                # synchronize current step and avg loss, each process keeps its own running avg
                loss_avg = utils.reduce_tensor(loss.new([loss_avg]), args.world_size).item()
                loss_now = utils.reduce_tensor(loss.new([loss_now]), args.world_size).item()
                update_sample_count *= args.world_size

            # -- Hitung waktu yang telah berjalan & sisa waktu (ETA)
            waktu_terpakai = time.time() - epoch_start_time
            progress = (update_idx + 1) / updates_per_epoch  # 0.0 - 1.0
            if progress > 0:
                estimasi_total = waktu_terpakai / progress
                estimasi_sisa = estimasi_total - waktu_terpakai
            else:
                estimasi_sisa = 0.0

            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                    f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})  '
                    f'Elapsed/ETA: {waktu_terpakai:.1f}s / {estimasi_sisa:.1f}s'
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, f'train-batch-{batch_idx}.jpg'),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    loss_avg = losses_m.avg
    if args.distributed:
        # synchronize avg loss, each process keeps its own running avg
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = utils.reduce_tensor(loss_avg, args.world_size).item()

    return OrderedDict([('loss', loss_avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        model_dtype=None,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == "npu":
                torch.npu.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


main()


#endregion