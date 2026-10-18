from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from functools import partial
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    conv_nd,
    linear,
    SiLU,
    avg_pool_nd,
    zero_module,
    normalization,
    checkpoint,
)

class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == th.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(th.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class PositionalEncod(nn.Module):
    def __init__(self, PosEnc=2, device='cuda'):
        super().__init__()
        self.PEnc = PosEnc
        self.k_pi = (th.tensor(np.pi)*(2**th.arange(self.PEnc))).reshape(-1, self.PEnc).to(device)
        self.k_pi = self.k_pi.T

    def forward(self, input):
        tmp = th.cat([th.sin(self.k_pi*input), th.cos( self.k_pi*input)], axis=0)

        return th.cat([input,tmp.T],-1)

class TimeEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = th.outer(x * self.scale, emb) 
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MeanEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of Mean value.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to Mean value. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = th.outer(x * self.scale, emb) # 计算两个一维张量的外积
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class StdEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of stanard value.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to stanard value. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = th.outer(x * self.scale, emb)
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class VelEmbedding(nn.Module):
    __doc__ = r"""Computes a velocity embedding.

    Input:
        x: tensor of shape (B, Nz)
    Output:
        tensor of shape (B, C, Nt)
    Args:
        dim_depth (int): depth dim
        dim_time (int): time dim
    """

    def __init__(self, dim, dim_sample=20000):
        super().__init__()
        self.dim = dim
        self.dim_sample = dim_sample
        self.vel_mlp  = nn.Sequential(
            nn.Linear(dim_sample, 8*dim),
            nn.GELU(),
            nn.Linear(8*dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
            nn.GELU(),
        )

    def forward(self, x):
        b, h = x.shape

        if h != self.dim_sample:
            raise ValueError("velocity sample points should be 1000")

        emb = self.vel_mlp(x)

        return emb

class WellVelEmbedding(nn.Module):
    __doc__ = r"""Computes a well velocity embedding.

    Input:
        x: tensor of shape (B, C, Nz)
    Output:
        tensor of shape (B, C, Nt)
    Args:
        dim_depth (int): depth dim
        dim_time (int): time dim
    """

    def __init__(self, inp_channels, out_channels, dim_depth, dim_time, scale = 0.0002):
        super().__init__()
        self.inp_channels = inp_channels
        self.dim_depth = dim_depth
        self.dim_time = dim_time
        self.scale = scale
        self.vel_mlp  = nn.Sequential(
            linear(dim_depth, dim_time),
            nn.SiLU()
        )

        self.conv = nn.Conv1d(inp_channels, out_channels, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        if h != self.dim_depth:
            raise ValueError("velocity height should be consistent with setting")
        if c != self.inp_channels:
            raise ValueError("input channel should be consistent with setting")

        x = self.scale * x.permute(0, 1, 3, 2).reshape(b * c * w, h)
        x = self.vel_mlp(x)
        x = x.view(b, c, w, self.dim_time).permute(0, 1, 3, 2)
        emb = self.conv(x)

        return emb

class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)

class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [nn.Conv1d(channel, channel, 4, stride=2, padding=1)]

        super().__init__(*layers)

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param time_emb_channels: the number of timestep embedding channels.
    :param sx_emb_channels: the number of source location embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_channels,
        coords_emb_channels=None,
        cls_emb_channels=None,
        dropout=0,
        use_conv=False,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_channels = time_emb_channels
        self.coords_emb_channels = coords_emb_channels
        self.cls_emb_channels = cls_emb_channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            WeightStandardizedConv1d(self.in_channels, self.out_channels, 3, padding=1),
            normalization(self.out_channels),
            nn.SiLU(),
        )

        # time embedding
        if coords_emb_channels is not None:
            self.time_emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    time_emb_channels + 2 * coords_emb_channels,
                    4 * self.out_channels,
                ),
                nn.SiLU(),
                linear(
                    4 * self.out_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        else:
            self.time_emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    time_emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )

        if cls_emb_channels is not None:
            self.cls_emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    cls_emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        else:
            self.cls_emb_layers = None

        self.out_layers = nn.Sequential(
            WeightStandardizedConv1d(self.out_channels, self.out_channels, 3, padding=1),
            normalization(self.out_channels),
            nn.SiLU(),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, x, time_emb, cls_emb=None):
        h = self.in_layers(x)
        time_emb_out = self.time_emb_layers(time_emb).type(h.dtype)
        while len(time_emb_out.shape) < len(h.shape):
            time_emb_out = time_emb_out[..., None]

        cls_emb_out = None
        if self.cls_emb_layers is not None:
            cls_emb = list(filter(lambda x: x is not None, [cls_emb]))
            cls_emb = th.cat(cls_emb, dim = -1)
            cls_emb_out = self.cls_emb_layers(cls_emb).type(h.dtype)
            while len(cls_emb_out.shape) < len(h.shape):
                cls_emb_out = cls_emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            if cls_emb_out is None:
                scale, shift = th.chunk(time_emb_out, 2, dim=1)
            else:
                scale, shift = th.chunk(time_emb_out + cls_emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if cls_emb_out is None:
                h = h + time_emb_out + cls_emb_out
            else:
                h = h + time_emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()

        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])

class ResBlockWithAttention(nn.Module):
    def __init__(self, in_channel, out_channel, time_embed_dim, coords_embed_dim, cls_embed_dim,
        dropout, num_heads, use_scale_shift_norm=True, use_attention=False):
        super().__init__()

        self.resblocks = ResBlock(in_channel, out_channel, 
            time_embed_dim, coords_embed_dim, cls_embed_dim, dropout)

        if use_attention:
            self.attention = AttentionBlock(out_channel, num_heads)

        else:
            self.attention = None

    def forward(self, input, time, cls):
        out = self.resblocks(input, time, cls)

        if self.attention is not None:
            out = self.attention(out)

        return out

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        time_emb_scale=1.0,
        coords_cond=True,
        num_classes=None,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        PosEnc=2,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.coords_cond = coords_cond
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.padder_size = 2 ** len(channel_mult)
        self.PosEnc = PosEnc

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels, time_emb_scale),
            linear(model_channels, time_embed_dim),
            nn.GELU(),
            linear(time_embed_dim, time_embed_dim),
        )

        coords_embed_dim = None
        v_embed_dim = None
        if self.coords_cond:
            coords_embed_dim = model_channels * 4
            self.coords_embed = nn.Sequential(
                    linear(20000*3, 8*coords_embed_dim),
                    nn.GELU(),
                    linear(8*coords_embed_dim, 4*coords_embed_dim),
                    nn.GELU(),
                    linear(4*coords_embed_dim, 2*coords_embed_dim),
                    nn.GELU(),
                    linear(2*coords_embed_dim, coords_embed_dim),
            )
            self.v_embed = VelEmbedding(coords_embed_dim)

        cls_embed_dim = None
        if self.num_classes is not None:
            cls_embed_dim = model_channels * 4
            self.cls_embed = nn.Sequential(
                nn.Embedding(num_classes, model_channels),
                nn.Linear(model_channels, cls_embed_dim),
                nn.GELU(),
                nn.Linear(cls_embed_dim, cls_embed_dim)
            )

        n_block = len(channel_mult)
        down_layers = [nn.Conv1d(in_channels, model_channels, 3, padding=1)]
        feat_channels = [model_channels]
        in_ch = model_channels
        for i in range(n_block):
            for _ in range(num_res_blocks):
                out_ch = model_channels * channel_mult[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_ch,
                        out_ch,
                        time_embed_dim,
                        coords_embed_dim,
                        cls_embed_dim,
                        dropout,
                        num_heads,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_attention=2 ** i in attention_resolutions,
                    )
                )

                feat_channels.append(out_ch)
                in_ch = out_ch

            if i != n_block - 1:
                down_layers.append(Downsample(in_ch))
                feat_channels.append(in_ch)

        self.down = nn.ModuleList(down_layers)

        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_ch,
                    in_ch,
                    time_embed_dim,
                    coords_embed_dim,
                    cls_embed_dim,
                    dropout,
                    num_heads,
                    use_scale_shift_norm=use_scale_shift_norm,
                    use_attention=True,
                ),
                ResBlockWithAttention(
                    in_ch,
                    in_ch,
                    time_embed_dim,
                    coords_embed_dim,
                    cls_embed_dim,
                    dropout,
                    num_heads,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(num_res_blocks + 1):
                out_ch = model_channels * channel_mult[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_ch + feat_channels.pop(),
                        out_ch,
                        time_embed_dim,
                        coords_embed_dim,
                        cls_embed_dim,
                        dropout,
                        num_heads,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_attention=2 ** i in attention_resolutions,
                    )
                )

                in_ch = out_ch

            if i != 0:
                up_layers.append(Upsample(in_ch))

        self.up = nn.ModuleList(up_layers)

        out_layers = []
        out_layers.append(
                ResBlockWithAttention(
                    in_ch,
                    model_channels,
                    time_embed_dim,
                    coords_embed_dim,
                    cls_embed_dim,
                    dropout,
                    num_heads,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
        )
        out_layers.append(nn.Conv1d(model_channels, out_channels, 3, padding=1))
        self.out = nn.ModuleList(out_layers)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.down.apply(convert_module_to_f16)
        self.mid.apply(convert_module_to_f16)
        self.up.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.down.apply(convert_module_to_f32)
        self.mid.apply(convert_module_to_f32)
        self.up.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.down.parameters()).dtype

    def forward(self, input, time, x_coords, z_coords, sx_coords, v, cls=None, cond=None):
        _, _, h = input.shape
        out = self.check_input_size(input)

        time_embed = self.time_embed(time)

        # coords_embed = None
        # vel_embed = None
        if self.coords_cond:
            coords = th.cat((x_coords, z_coords, sx_coords), dim=-1)
            coords_embed = self.coords_embed(coords)
            v_embed = self.v_embed(v)
            time_embed = th.cat((time_embed, coords_embed, v_embed), dim=-1)

        cls_embed = None
        if self.num_classes is not None:
            cls_embed = self.cls_embed(cls.squeeze(1))

        feats = []
        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed, cls_embed)

            else:
                out = layer(out)

            feats.append(out)

        for layer in self.mid:
            out = layer(out, time_embed, cls_embed)

        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(th.cat((out, feats.pop()), 1), time_embed, cls_embed)

            else:
                out = layer(out)

        for layer in self.out:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed, cls_embed)

            else:
                out = layer(out)

        return out[:, :, :h]

    def check_input_size(self, input):
        _, _, h = input.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        input = F.pad(input, (0, mod_pad_h), mode='replicate')
        return input


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, x_coords, z_coords, v, cls, low_res=None, **kwargs):
        _, _, new_height = x.shape
        upsampled = F.interpolate(low_res, (new_height), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, x_coords, z_coords, v, cls, **kwargs)

if __name__ == '__main__':
    model = UNetModel(        
        in_channels = 1,
        model_channels = 32,
        out_channels = 1,
        num_res_blocks = 2,
        attention_resolutions = (1, 2, 4, 8)
    )
    print(model)
    test = th.randn(3, 1, 128).cuda()
