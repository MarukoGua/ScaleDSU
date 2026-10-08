from typing import Tuple
from math import log

import torch
from torch import nn, einsum
from einops import rearrange
from timm.layers import trunc_normal_
import torch.nn.functional as F


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}

def exists(x):
    return x is not None

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample(nn.Module):
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = F.pad(hidden_states, (self.pad,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros([hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]])
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        weight[indices, indices] = kernel

        return F.conv1d(hidden_states, weight, stride=2)


class Upsample(nn.Module):
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros([hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]])
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        weight[indices, indices] = kernel
        return F.conv_transpose1d(hidden_states, weight, stride=2, padding=self.pad * 2 + 1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = cond_emb.unsqueeze(-1)
            scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)    


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_conv = nn.Conv1d(config.in_channels, config.block_out_channels[0], config.patch_size, padding=(config.patch_size-1)//2)
        
        dim = config.block_out_channels[0]
        self.max_stride = 2 ** (len(config.block_out_channels) - 2)

        # time_dim = dim * 4
        time_dim = 0
        # sinu_pos_emb = SinusoidalPosEmb(dim)
        # fourier_dim = dim
        # self.time_mlp = nn.Sequential(
        #     sinu_pos_emb,
        #     nn.Linear(fourier_dim, time_dim),
        #     nn.GELU(),
        #     nn.Linear(time_dim, time_dim)
        # )
        
        num_classes = config.num_labels
        self.classes_emb = nn.Embedding(num_classes, dim)
        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        self.downs = nn.ModuleList([])

        dims = config.block_out_channels
        in_out = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                    ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample() if not is_last else nn.Conv1d(dim_out, dim_out, 3, padding = 1)
                ])
                )
        
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_out, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                ResnetBlock(dim_in + dim_out, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample() if not is_last else nn.Conv1d(dim_in, dim_in, 3, padding = 1)
            ]))
        
        self.out_dim = config.in_channels
        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def pad_to(self, x):
        length = x.shape[-1]
        if length % self.max_stride > 0 :
            new_length = length + self.max_stride - length % self.max_stride
        else:
            new_length = length
        
        lpad, rpad = int((new_length-length)/2), int(new_length-length)-int((new_length-length)/2)
        pad = (lpad, rpad)
        
        x = F.pad(x, pad, "constant", 0)
        return x, pad
    
    def unpad(self, x, pad):
        if pad[0]+pad[1] > 0:
            x = x[:, :, pad[0]:-pad[1]]
        return x

    # def forward(self, x, time, classes):
    def forward(self, x, classes=None):
        x, pad = self.pad_to(x)
        x = self.init_conv(x)
        r = x.clone()

        # t = self.time_mlp(time)

        h = []
        c = None
        if classes is not None:
            classes_emb = self.classes_emb(classes)
            classes_emb = torch.reshape(classes_emb, (classes_emb.shape[0], -1))
            c = self.classes_mlp(classes_emb)

        for block1, block2, attn, downsample in self.downs:
            # x = block1(x, t, c)
            x = block1(x, c)
            h.append(x)
            # x = block2(x, t, c)
            x = block2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        # x = self.mid_block1(x, t, c)
        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        # x = self.mid_block2(x, t, c)
        x = self.mid_block2(x, c)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            # x = block1(x, t, c)
            x = block1(x, c)
            x = torch.cat((x, h.pop()), dim=1)
            # x = block2(x, t, c)
            x = block2(x, c)
            x = attn(x)
            x = upsample(x, )
        x = torch.cat((x, r), dim=1)
        x = self.unpad(x, pad)
        # x = self.final_res_block(x, t, c)
        x = self.final_res_block(x, c)
        return self.final_conv(x)


class UNetConfig:

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            block_out_channels: Tuple = (8, 16, 32),
            patch_size: int = 7,
            use_timestep_embedding: bool = False,
            time_embedding_type: str = "positional", 
            num_labels: int=256,
            **kwargs
    ):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.patch_size = patch_size

        self.time_embedding_type = time_embedding_type
        self.use_timestep_embedding = use_timestep_embedding
        
        self.num_labels = num_labels
        self.kwargs = kwargs

        
