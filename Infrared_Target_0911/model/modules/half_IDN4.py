import torch
from torch import nn
from .Style import StyleLayer
from utils.utils import Uhalf_ResnetBlocWithAttn, Downsample
from .SwinTransformer import SwinTransformer
from .BasicBlock import  ScaleEncoder, Fusion, BasicConv, InputEncoder, MiddleEncoder, MiddleDecoder, OutputDecoder,ResBlock1,ResBlock2,ResBlock3
from utils.wtconv2d import WTConv2d
from mamba_ssm import Mamba
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import einops
from einops import rearrange
#from visualizer import get_local
import numpy as np
from model.modules import blocks
from mamba_ssm import Mamba

m = None


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class Downsamples(nn.Module):
    def __init__(self, n_feat):
        super(Downsamples, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat //2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)  ##返回所有元素的方差
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x, ill_map, mask=None):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh w) h dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        ill = reshape(ill_map)
        v = v * ill
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac

        # 置为-1e9 因为softmax中0输出为1，我们想要输出为0就是负无穷大
        if mask is not None:
            mask = reshape(mask)
            mask = torch.matmul(mask, mask.transpose(-2, -1)) * self.fac
            res = res.masked_fill(mask == 0, -1e9)
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh w) h dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res


class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor, ill_map_height, ill_map_width, mask=None):
        assert len(x.size()) == 4
        x = self.row_att(x, ill_map_width, mask=mask)
        x = x.transpose(-2, -1)
        if mask is not None:
            x = self.col_att(x, ill_map_height.transpose(-2, -1), mask=mask.transpose(-2, -1), )
        else:
            x = self.col_att(x, ill_map_height.transpose(-2, -1), mask=mask)
        x = x.transpose(-2, -1)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.rep_conv1 = Conv2d_BN(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.rep_conv2 = Conv2d_BN(hidden_features, hidden_features, 1, 1, 0, groups=hidden_features)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        identity = x
        x = self.project_in(x)
        x1 = x + self.rep_conv1(x) + self.rep_conv2(x)
        x2 = self.dwconv(x)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x + identity

    @torch.no_grad()
    def fuse(self):
        conv = self.rep_conv1.fuse()  ##Conv_BN
        conv1 = self.rep_conv2.fuse()  ##Conv_BN

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class WM(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.convb = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=1, padding=1)
        )
        self.model1 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c,  # Model dimension d_model
            d_state=32,  # SSM state expansion factor
            d_conv=3,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c,  # Model dimension d_model
            d_state=32,  # SSM state expansion factor
            d_conv=3,  # Local convolution width
            expand=9,  # Block expansion factor
        )
        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(normalized_shape=c)
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.convb(x) + x
        x = self.ln(x.reshape(b, -1, c)) #reshape后是3n H/2*W/2 c
       
        y = self.model1(x).permute(0, 2, 1) 
        output = y.reshape(b, c, h, w)
        return self.smooth(output)


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class WMB(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(WMB, self).__init__()
        self.DWT = blocks.DWT()
        self.IWT = blocks.IWT()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.illu = Illumination_Estimator(dim, n_fea_in=dim+1, n_fea_out=dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.mb = WM(dim)

    def forward(self, input_):
        global m
        x = input_
        n, c, h, w = x.shape
        x = self.norm1(x)
        x = data_transform(x)
        input_dwt = self.DWT(x)
        # input_LL=A [B,C,H/2,W/2]   input_high0={V,H,D} [3B,C,H/2,W/2]
        input_LL, input_high = input_dwt[:n, ...], input_dwt[n:, ...]
        input_LL, input_image = self.illu(input_LL)
        input_high = self.mb(input_high)

        output = self.IWT(torch.cat((input_LL, input_high), dim=0))
        output = inverse_data_transform(output)

        x = x + output
        x = x + self.ffn(self.norm2(x))
        return x
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


# self.encoder = half_IDN(in_channel=3, norm_groups=32, inner_channel=64, 
# channel_mults=[1,2,4,8,16], attn_res=[32], res_blocks=2, dropout=0.2, image_size=512)
class half_IDN(nn.Module):
    def __init__(self, in_channel=3, inner_channel=64, norm_groups=32, channel_mults=(1, 2, 4, 8, 16), num_blocks=[1, 2, 4,8,16],
                 heads=[1, 2, 4, 8,16,32],attn_res=(32), res_blocks=2, dropout=0.2, image_size=512,LayerNorm_type = 'WithBias'):
        super().__init__()
        #self.SwinTransformer = SwinTransformer()
        #swin主路
        # self.Fusion_2 = Fusion(128, 96)
        # self.Fusion_4 = Fusion(256, 192)
        # self.Fusion_8 = Fusion(512, 384)
        # self.Fusion_16 = Fusion(1024, 768)
        

        # self.Fusion_2 = Fusion(96, 128)
        # self.Fusion_4 = Fusion(192, 256)
        # self.Fusion_8 = Fusion(384, 512)
        # self.Fusion_16 = Fusion(768, 1024)
        self.Fusion_0 = Fusion(64,64)
        self.Fusion_2 = Fusion(128, 128)
        self.Fusion_4 = Fusion(256, 256)
        self.Fusion_8 = Fusion(512, 512)
        self.Fusion_16 = Fusion(1024, 1024)
        
        
        # self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv0 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        
        
        # self.WTconv2d = WTConv2d(3,3,stride=1, wt_levels=2)
        # self.WTconv2d_0 = WTConv2d(64,64,stride=1, wt_levels=2)
        # self.WTconv2d_1 = WTConv2d(128,128,stride=2, wt_levels=2)
        # self.WTconv2d_2 = WTConv2d(256,256,stride=2, wt_levels=2)
        # self.WTconv2d_3 = WTConv2d(512,512,stride=2, wt_levels=2)
        # self.WTconv2d_4 = WTConv2d(1024,1024,stride=2, wt_levels=2)
        
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=1)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv1_3 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv1_4 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv1_5 = nn.Conv2d(512, 1024, kernel_size=1)
        
        
        # self.conv2_1 = nn.Conv2d(96,128,kernel_size=1)
        # self.conv2_2 = nn.Conv2d(192,256,kernel_size=1)
        # self.conv2_3 = nn.Conv2d(384,512,kernel_size=1)
        # self.conv2_4 = nn.Conv2d(768,1024,kernel_size=1)
        
        self.estimator = Illumination_Estimator(inner_channel)
        self.decoder_level0_0 = nn.Sequential(*[WMB(dim=int(int(inner_channel )), num_heads=heads[0],
                                                    ffn_expansion_factor=2.66,
                                                    bias=False, LayerNorm_type=LayerNorm_type)
                                                for i in range(num_blocks[0])])
        self.decoder_level1_0 = nn.Sequential(*[WMB(dim=int(int(inner_channel * 2)), num_heads=heads[0],
                                                    ffn_expansion_factor=2.66,
                                                    bias=False, LayerNorm_type=LayerNorm_type)
                                                for i in range(num_blocks[0])])
        
        self.decoder_level2_0 = nn.Sequential(*[
            WMB(dim=int(int(inner_channel * 2 * 2)), num_heads=heads[0],
                ffn_expansion_factor=2.66, bias=False,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.decoder_level3_0 = nn.Sequential(*[
            WMB(dim=int(int(inner_channel * 2 * 4)), num_heads=heads[0],
                ffn_expansion_factor=2.66, bias=False,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level4_0 = nn.Sequential(*[
            WMB(dim=int(int(inner_channel * 2 * 8)), num_heads=heads[0],
                ffn_expansion_factor=2.66, bias=False,
                LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.decoder_level5_0 = nn.Sequential(*[
        #     WMB(dim=int(int(inner_channel * 2 * 16)), num_heads=heads[0],
        #         ffn_expansion_factor=2.66, bias=False,
        #         LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        # self.latent = blocks.FFAB(int(int(norm_groups * 2 * 4)))
        self.down_1 = Downsamples(int(inner_channel))  # From Level 1 to Level 2  通道变为输入通道两倍，图像下采样 宽高/2
        self.down_2 = Downsamples(int(inner_channel * 2))  # From Level 1 to Level 2
        self.down_3 = Downsamples(int(inner_channel * 4))  # From Level 1 to Level 2
        self.down_4 = Downsamples(int(inner_channel * 8))  # From Level 1 to Level 2
        self.down_5 = Downsamples(int(inner_channel * 16))  # From Level 1 to Level 2
        self.patch_embed = OverlapPatchEmbed(3, 32)
        
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        self.conv_body_first = StyleLayer(3, pre_channel, 3, bias=True, activate=True)
        self.conv_body_down = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            self.conv_body_down.append(StyleLayer(pre_channel, channel_mult, 3, downsample=True))
            self.condition_shift.append(StyleLayer(pre_channel, channel_mult, 3, bias=True, activate=False))
            for _ in range(0, res_blocks):
                downs.append(Uhalf_ResnetBlocWithAttn(pre_channel, channel_mult, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)
        
        self.FS = ResBlock1(1024,1024)
        self.SS0 = ResBlock2(64,64)
        self.SS1 = ResBlock2(128,128)
        self.SS2 = ResBlock2(256,256)
        self.SS3 = ResBlock2(512,512)
        self.SS4 = ResBlock2(1024,1024)
        
    # 8 重制版 3_20
    
    def forward(self, img): # B,3,512,512
        img = self.conv1_1(img)  # 64 512 512
        out_enc_level0_0 = self.decoder_level0_0(img)
        inp_enc_level1_0 = self.down_1(out_enc_level0_0) # 128 256 256
        out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0) # 128 256 256
        
        inp_enc_level2_0 = self.down_2(out_enc_level1_0) #256 128 128
        out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0) # 256 128 128
        
        inp_enc_level3_0 = self.down_3(out_enc_level2_0) # 512 64 64
        out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0) # 512 64 64
        
        inp_enc_level4_0 = self.down_4(out_enc_level3_0) # 1024 32 32
        out_enc_level4_0 = self.decoder_level4_0(inp_enc_level4_0) # 1024 32 32
        
        # out_enc_level5_0 = self.FS (out_enc_level4_0) # 1024 32 32
        
        
        f0 = self.SS0(img)
        f1 = self.SS1(inp_enc_level1_0)
        f2 = self.SS2(inp_enc_level2_0)
        f3 = self.SS3(inp_enc_level3_0)
        f4 = self.SS4(inp_enc_level4_0)
        
        
        
        f0_1 = self.downs[1](img) # torch.Size([2, 64, 512, 512])
        f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
        f0 = f0_2 + img
        
        f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
        f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
        f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
        f1 = f1_3
        f1 = self.SS1(f1)
        
        f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
        f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
        f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
        f2 = f2_3
        f2 = self.SS2(f2)
        
        f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
        f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
        f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
        f3 =  f3_3
        f3 = self.SS3(f3)
        
        f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
        f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
        f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
        f4 = self.SS4(f4_3)
        
        # f4 = self.SS(f4)
        #fusion0 = self.Fusion_0(out_enc_level0_0,f0)
        fusion1 = self.Fusion_2(out_enc_level1_0,f1)
        fusion2 = self.Fusion_4(out_enc_level2_0,f2)
        fusion3 = self.Fusion_8(out_enc_level3_0,f3)
        fusion4 = self.Fusion_16(out_enc_level4_0,f4)
        #return [fusion0,fusion1,fusion2,fusion3] 
        return [fusion1,fusion2,fusion3,fusion4] 
    
    
    
    
    
        
    # 7final WT+Mamba+FAS+CFAS+FAI
    # def forward(self, img): # B,3,512,512
    #     img = self.conv1_1(img)  # 64 512 512
    #     # out_enc_level0_0 = self.decoder_level0_0(img)
    #     # inp_enc_level1_0 = self.down_1(out_enc_level0_0) # 128 256 256
    #     # out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0) # 128 256 256
        
    #     # inp_enc_level2_0 = self.down_2(out_enc_level1_0) #256 128 128
    #     # out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0) # 256 128 128
        
    #     # inp_enc_level3_0 = self.down_3(out_enc_level2_0) # 512 64 64
    #     # out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0) # 512 64 64
        
        
    #     # out_enc_level5_0 = self.FS (out_enc_level4_0) # 1024 32 32
        
        
        
    #     inp_enc_level1_0 = self.down_1(img) # 128 256 256
    #     out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0) # 128 256 256
        
    #     inp_enc_level2_0 = self.down_2(out_enc_level1_0) #256 128 128
    #     out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0) # 256 128 128
        
    #     inp_enc_level3_0 = self.down_3(out_enc_level2_0) # 512 64 64
    #     out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0) # 512 64 64
        
    #     inp_enc_level4_0 = self.down_4(out_enc_level3_0) # 1024 32 32
    #     out_enc_level4_0 = self.decoder_level4_0(inp_enc_level4_0) # 1024 32 32
        
        
        
        
        
    #     # f0 = self.SS0(out_enc_level0_0)
    #     # f1 = self.SS1(inp_enc_level1_0)
    #     # f2 = self.SS2(inp_enc_level2_0)
    #     # f3 = self.SS3(inp_enc_level3_0)
        
        
    #     # f1 = self.SS1(inp_enc_level1_0)
    #     # f2 = self.SS2(inp_enc_level2_0)
    #     # f3 = self.SS3(inp_enc_level3_0)
        
    #     # f4 = self.SS4(inp_enc_level4_0)
        
        
        
    #     f0_1 = self.downs[1](img) # torch.Size([2, 64, 512, 512])
    #     f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
    #     f0 = f0_2 + img
        
    #     f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
    #     f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
    #     f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
    #     f1 = self.SS1(f1_3)
        
    #     f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
    #     f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
    #     f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
    #     f2 = self.SS2(f2_3)
        
    #     f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
    #     f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
    #     f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
    #     f3 = self.SS3(f3_3)
        
    #     f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
    #     f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
    #     f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
    #     f4 = self.SS4(f4_3)
        
    #     # f4 = self.SS(f4)
    #     # fusion0 = self.Fusion_0(out_enc_level0_0,f0)
    #     # fusion1 = self.Fusion_2(out_enc_level1_0,f1)
    #     # fusion2 = self.Fusion_4(out_enc_level2_0,f2)
    #     # fusion3 = self.Fusion_8(out_enc_level3_0,f3)
    #     # return [fusion0,fusion1,fusion2,fusion3] 
        
        
        
        
    #     fusion1 = self.Fusion_2(out_enc_level1_0,f1)
    #     fusion2 = self.Fusion_4(out_enc_level2_0,f2)
    #     fusion3 = self.Fusion_8(out_enc_level3_0,f3)
    #     fusion4 = self.Fusion_16(out_enc_level4_0,f4) 
    #     return [fusion1,fusion2,fusion3,fusion4] 
        
    #     ###
    #     # fusion1 = self.Fusion_2(f1,out_enc_level1_0)
    #     # fusion2 = self.Fusion_4(f2,out_enc_level2_0)
    #     # fusion3 = self.Fusion_8(f3,out_enc_level3_0)
    #     # fusion4 = self.Fusion_16(f4,out_enc_level4_0) 
    #     # return [fusion1,fusion2,fusion3,fusion4] 
        
    #     ####
        
        
        # SwinTouts = self.SwinTransformer(img)
        # # img = self.WTconv2d(img)
        # w = self.conv1_1(img)
        # w1 = self.WTconv2d_0(w)

        # feat = self.downs[0](img) # torch.Size([2, 64, 512, 512])
        # f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
        # f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
        # f0 = w1 + f0_2
        
        # w2_1 = self.conv1_2(f0)
        # w2 = self.WTconv2d_1(w2_1)
        # f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
        # f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
        # f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
        # f1 = w2 + f1_3
        
        # w3_1 = self.conv1_3(f1)
        # w3 = self.WTconv2d_2(w3_1)
        # f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
        # f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
        # f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
        # f2 = w3 + f2_3
        
        # w4_1 = self.conv1_4(f2)
        # w4 = self.WTconv2d_3(w4_1)
        # f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
        # f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
        # f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
        # f3 = w4 + f3_3
        
        # w5_1 = self.conv1_5(f3)
        # w5 = self.WTconv2d_4(w5_1)
        # f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
        # f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
        # f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
        # f4 = w5 + f4_3
        # fusion1 = self.Fusion_2(SwinTouts[0],f1)
        # fusion2 = self.Fusion_4(SwinTouts[1],f2)
        # fusion3 = self.Fusion_8(SwinTouts[2],f3)
        # fusion4 = self.Fusion_16(SwinTouts[3],f4)
       
        # m1 = torch.cat([fusion1, f1], dim=1)
        # m2 = torch.cat([fusion2, f2], dim=1)
        # m3 = torch.cat([fusion3, f3], dim=1)
        # m4 = torch.cat([fusion4, f4], dim=1)
        # return [m1, m2, m3, m4]
    # # new_4_2 WMB+down+swin+fusion
    # def forward(self, img): # B,3,512,512
    #     SwinTouts = self.SwinTransformer(img)
    #     illu_fea, illu_map = self.estimator(img)
    #     img = img * illu_map + img
    #     inp_enc_encoder1 = self.patch_embed(img)

    #     inp_enc_level1_0 = self.down_1(inp_enc_encoder1)
    #     out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0)

    #     inp_enc_level2_0 = self.down_2(out_enc_level1_0)
    #     out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0)

    #     inp_enc_level3_0 = self.down_3(out_enc_level2_0)
    #     out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0)
        
    #     inp_enc_level4_0 = self.down_3(out_enc_level3_0)
    #     out_enc_level4_0 = self.decoder_level3_0(inp_enc_level4_0)
    #     # img = self.WTconv2d(img)
    #     w = self.conv1_1(img)
    #     w1 = self.WTconv2d_0(w)
    #     feat = self.downs[0](img) # torch.Size([2, 64, 512, 512] 
    #     # f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
    #     # f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
    #     f0_1 = self.down_1(feat)
    #     w1 = self.decoder_level1_0(f0_1)
    #     f0 = w1 + f0_1
        
    #     # w2_1 = self.conv1_2(f0)
    #     # w2 = self.WTconv2d_1(w2_1)
    #     f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
    #     f1_2 = self.down_2(f1_1)
    #     # f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
    #     #f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
    #     w2 = self.decoder_level2_0(f1_2)
    #     f1 = w2 + f1_2
        
    #     # w3_1 = self.conv1_3(f1)
    #     # w3 = self.WTconv2d_2(w3_1)
    #     f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
    #     # f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
    #     f2_2 = self.down_3(f2_1)
    #     #f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
    #     w3 = self.decoder_level3_0(f2_2)
    #     f2 = w3 + f2_2
        
    #     # w4_1 = self.conv1_4(f2)
    #     # w4 = self.WTconv2d_3(w4_1)
    #     f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
    #     # f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
    #     f3_2 = self.down_3(f3_1)
    #     #f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
    #     w4 = self.decoder_level4_0(f3_2)
    #     f3 = w4 + f3_2
        
    #     # w5_1 = self.conv1_5(f3)
    #     # w5 = self.WTconv2d_4(w5_1)
    #     f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
    #     f4_2 = self.down_4(f4_1)
    #     # f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
    #     w5 = self.decoder_level5_0(f4_2)
    #     # f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
    #     # w5 = self.decoder_level5_0(f4_2)
    #     f4 =  w5 + f4_2
    #     fusion1 = self.Fusion_2(SwinTouts[0],f1)
    #     fusion2 = self.Fusion_4(SwinTouts[1],f2)
    #     fusion3 = self.Fusion_8(SwinTouts[2],f3)
    #     fusion4 = self.Fusion_16(SwinTouts[3],f4)
       
    #     m1 = torch.cat([fusion1, f1], dim=1)
    #     m2 = torch.cat([fusion2, f2], dim=1)
    #     m3 = torch.cat([fusion3, f3], dim=1)
    #     m4 = torch.cat([fusion4, f4], dim=1)
    #     return [m1, m2, m3, m4]
        
        
        
    
    # new_4_2 WMB+down+swin+fusion
    # def forward(self, img): # B,3,512,512
    #     SwinTouts = self.SwinTransformer(img)
    #     illu_fea, illu_map = self.estimator(img)
    #     img = img * illu_map + img
    #     inp_enc_encoder1 = self.patch_embed(img)

    #     inp_enc_level1_0 = self.down_1(inp_enc_encoder1)
    #     out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0)

    #     inp_enc_level2_0 = self.down_2(out_enc_level1_0)
    #     out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0)

    #     inp_enc_level3_0 = self.down_3(out_enc_level2_0)
    #     out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0)
        
    #     inp_enc_level4_0 = self.down_3(out_enc_level3_0)
    #     out_enc_level4_0 = self.decoder_level3_0(inp_enc_level4_0)
        # # img = self.WTconv2d(img)
        # w = self.conv1_1(img)
        # w1 = self.WTconv2d_0(w)
        # feat = self.downs[0](img) # torch.Size([2, 64, 512, 512] 
        # # f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
        # # f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
        # f0_1 = self.down_1(feat)
        # w1 = self.decoder_level1_0(f0_1)
        # f0 = w1 + f0_1
        
        # # w2_1 = self.conv1_2(f0)
        # # w2 = self.WTconv2d_1(w2_1)
        # f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
        # f1_2 = self.down_2(f1_1)
        # # f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
        # #f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
        # w2 = self.decoder_level2_0(f1_2)
        # f1 = w2 + f1_2
        
        # # w3_1 = self.conv1_3(f1)
        # # w3 = self.WTconv2d_2(w3_1)
        # f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
        # # f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
        # f2_2 = self.down_3(f2_1)
        # #f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
        # w3 = self.decoder_level3_0(f2_2)
        # f2 = w3 + f2_2
        
        # # w4_1 = self.conv1_4(f2)
        # # w4 = self.WTconv2d_3(w4_1)
        # f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
        # # f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
        # f3_2 = self.down_3(f3_1)
        # #f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
        # w4 = self.decoder_level4_0(f3_2)
        # f3 = w4 + f3_2
        
        # # w5_1 = self.conv1_5(f3)
        # # w5 = self.WTconv2d_4(w5_1)
        # f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
        # f4_2 = self.down_4(f4_1)
        # # f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
        # w5 = self.decoder_level5_0(f4_2)
        # # f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
        # # w5 = self.decoder_level5_0(f4_2)
        # f4 =  w5 + f4_2
        # fusion1 = self.Fusion_2(SwinTouts[0],f1)
        # fusion2 = self.Fusion_4(SwinTouts[1],f2)
        # fusion3 = self.Fusion_8(SwinTouts[2],f3)
        # fusion4 = self.Fusion_16(SwinTouts[3],f4)
       
        # m1 = torch.cat([fusion1, f1], dim=1)
        # m2 = torch.cat([fusion2, f2], dim=1)
        # m3 = torch.cat([fusion3, f3], dim=1)
        # m4 = torch.cat([fusion4, f4], dim=1)
        # return [m1, m2, m3, m4]
    
        
    # # new_4 WMB + CNN + fusion +swin
    # def forward(self, img): # B,3,512,512
    #     SwinTouts = self.SwinTransformer(img)
    #     illu_fea, illu_map = self.estimator(img)
    #     img = img * illu_map + img
        
    #     # # img = self.WTconv2d(img)
    #     # w = self.conv1_1(img)
    #     # w1 = self.WTconv2d_0(w)
    #     feat = self.downs[0](img) # torch.Size([2, 64, 512, 512] 
    #     f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
    #     # f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
    #     w1 = self.decoder_level1_0(f0_1)
    #     f0 = w1 + f0_1
        
    #     # w2_1 = self.conv1_2(f0)
    #     # w2 = self.WTconv2d_1(w2_1)
    #     f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
    #     f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
    #     #f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
    #     w2 = self.decoder_level2_0(f1_2)
    #     f1 = w2 + f1_2
        
    #     # w3_1 = self.conv1_3(f1)
    #     # w3 = self.WTconv2d_2(w3_1)
    #     f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
    #     f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
    #     #f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
    #     w3 = self.decoder_level3_0(f2_2)
    #     f2 = w3 + f2_2
        
    #     # w4_1 = self.conv1_4(f2)
    #     # w4 = self.WTconv2d_3(w4_1)
    #     f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
    #     f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
    #     #f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
    #     w4 = self.decoder_level4_0(f3_2)
    #     f3 = w4 + f3_2
        
    #     # w5_1 = self.conv1_5(f3)
    #     # w5 = self.WTconv2d_4(w5_1)
    #     f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
    #     f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
    #     f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
    #     # w5 = self.decoder_level5_0(f4_2)
    #     f4 =  f4_3
    #     fusion1 = self.Fusion_2(SwinTouts[0],f1_2)
    #     fusion2 = self.Fusion_4(SwinTouts[1],f2_2)
    #     fusion3 = self.Fusion_8(SwinTouts[2],f3_2)
    #     fusion4 = self.Fusion_16(SwinTouts[3],f4_3)
       
    #     m1 = torch.cat([fusion1, f1_2], dim=1)
    #     m2 = torch.cat([fusion2, f2_2], dim=1)
    #     m3 = torch.cat([fusion3, f3_2], dim=1)
    #     m4 = torch.cat([fusion4, f4_3], dim=1)
    #     return [m1, m2, m3, m4]
    
    
    # # new_1_2 cnn加上WT +swint  
    # def forward(self, img): # B,3,512,512
        
    #     # SwinTouts = self.SwinTransformer(img)
    #     # img = self.WTconv2d(img)
    #     w = self.conv1_1(img)
    #     w1 = self.WTconv2d_0(w)

    #     feat = self.downs[0](img) # torch.Size([2, 64, 512, 512])
    #     f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
    #     f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
    #     f0 = w1 + f0_2
        
    #     w2_1 = self.conv1_2(f0)
    #     w2 = self.WTconv2d_1(w2_1)
    #     f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
    #     f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
    #     f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
    #     f1 = w2 + f1_3
        
    #     w3_1 = self.conv1_3(f1)
    #     w3 = self.WTconv2d_2(w3_1)
    #     f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
    #     f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
    #     f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
    #     f2 = w3 + f2_3
        
    #     w4_1 = self.conv1_4(f2)
    #     w4 = self.WTconv2d_3(w4_1)
    #     f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
    #     f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
    #     f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
    #     f3 = w4 + f3_3
        
    #     w5_1 = self.conv1_5(f3)
    #     w5 = self.WTconv2d_4(w5_1)
    #     f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
    #     f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
    #     f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
    #     f4 = w5 + f4_3 # 1024 32 32
    #     # fusion1 = self.Fusion_2(SwinTouts[0],f1)
    #     # # fusion1 = self.conv2_1(fusion1)
    #     # fusion2 = self.Fusion_4(SwinTouts[1],f2)
    #     # # fusion2 = self.conv2_2(fusion2)
    #     # fusion3 = self.Fusion_8(SwinTouts[2],f3)
    #     # # fusion3 = self.conv2_3(fusion3)
    #     # fusion4 = self.Fusion_16(SwinTouts[3],f4)
    #     # fusion4 = self.conv2_4(fusion4)
       
    #     # m1 = torch.cat([fusion1, f1], dim=1)
    #     # m2 = torch.cat([fusion2, f2], dim=1)
    #     # m3 = torch.cat([fusion3, f3], dim=1)
    #     # m4 = torch.cat([fusion4, f4], dim=1)
    #     return [f1, f2, f3, f4]
    
    # # new_3 cnn加上WT +swint  fusion 改变位置 以swin为主路
    # def forward(self, img): # B,3,512,512
        
    #     SwinTouts = self.SwinTransformer(img)
    #     # img = self.WTconv2d(img)
    #     w = self.conv1_1(img)
    #     w1 = self.WTconv2d_0(w)

    #     feat = self.downs[0](img) # torch.Size([2, 64, 512, 512])
    #     f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
    #     f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
    #     f0 = w1 + f0_2
        
    #     w2_1 = self.conv1_2(f0)
    #     w2 = self.WTconv2d_1(w2_1)
    #     f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
    #     f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
    #     f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
    #     f1 = w2 + f1_3
        
    #     w3_1 = self.conv1_3(f1)
    #     w3 = self.WTconv2d_2(w3_1)
    #     f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
    #     f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
    #     f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
    #     f2 = w3 + f2_3
        
    #     w4_1 = self.conv1_4(f2)
    #     w4 = self.WTconv2d_3(w4_1)
    #     f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
    #     f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
    #     f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
    #     f3 = w4 + f3_3
        
    #     w5_1 = self.conv1_5(f3)
    #     w5 = self.WTconv2d_4(w5_1)
    #     f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
    #     f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
    #     f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
    #     f4 = w5 + f4_3
    #     fusion1 = self.Fusion_2(f1,SwinTouts[0])
    #     fusion1 = self.conv2_1(fusion1)
    #     fusion2 = self.Fusion_4(f2,SwinTouts[1])
    #     fusion2 = self.conv2_2(fusion2)
    #     fusion3 = self.Fusion_8(f3,SwinTouts[2])
    #     fusion3 = self.conv2_3(fusion3)
    #     fusion4 = self.Fusion_16(f4,SwinTouts[3])
    #     fusion4 = self.conv2_4(fusion4)
       
    #     m1 = torch.cat([fusion1, f1], dim=1)
    #     m2 = torch.cat([fusion2, f2], dim=1)
    #     m3 = torch.cat([fusion3, f3], dim=1)
    #     m4 = torch.cat([fusion4, f4], dim=1)
    #     return [m1, m2, m3, m4]
    
    
    # new_2 cnn加上WT +swint 
    # def forward(self, img): # B,3,512,512
        
    #     SwinTouts = self.SwinTransformer(img)
    #     # img = self.WTconv2d(img)
    #     w = self.conv1_1(img)
    #     w1 = self.WTconv2d_0(w)

    #     feat = self.downs[0](img) # torch.Size([2, 64, 512, 512])
    #     f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
    #     f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
    #     f0 = w1 + f0_2
        
    #     w2_1 = self.conv1_2(f0)
    #     w2 = self.WTconv2d_1(w2_1)
    #     f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
    #     f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
    #     f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
    #     f1 = w2 + f1_3
        
    #     w3_1 = self.conv1_3(f1)
    #     w3 = self.WTconv2d_2(w3_1)
    #     f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
    #     f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
    #     f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
    #     f2 = w3 + f2_3
        
    #     w4_1 = self.conv1_4(f2)
    #     w4 = self.WTconv2d_3(w4_1)
    #     f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
    #     f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
    #     f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
    #     f3 = w4 + f3_3
        
    #     w5_1 = self.conv1_5(f3)
    #     w5 = self.WTconv2d_4(w5_1)
    #     f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
    #     f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
    #     f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
    #     f4 = w5 + f4_3
    #     fusion1 = self.Fusion_2(SwinTouts[0],f1)
    #     fusion2 = self.Fusion_4(SwinTouts[1],f2)
    #     fusion3 = self.Fusion_8(SwinTouts[2],f3)
    #     fusion4 = self.Fusion_16(SwinTouts[3],f4)
       
    #     m1 = torch.cat([fusion1, f1], dim=1)
    #     m2 = torch.cat([fusion2, f2], dim=1)
    #     m3 = torch.cat([fusion3, f3], dim=1)
    #     m4 = torch.cat([fusion4, f4], dim=1)
    #     return [m1, m2, m3, m4]
        
        
    # # new_1 cnn加上WT 不加transformer 一种是cotact自己 另一种是contact cnn+wt contact cnn(还没试11.19回去试)
    # def forward(self, img): # B,3,512,512
        
    #     # SwinTouts = self.SwinTransformer(img)
    #     # img = self.WTconv2d(img)
    #     w = self.conv1_1(img)
    #     w1 = self.WTconv2d_0(w)

    #     feat = self.downs[0](img) # torch.Size([2, 64, 512, 512])
    #     f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
    #     f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
    #     f0 = w1 + f0_2
        
    #     w2_1 = self.conv1_2(f0)
    #     w2 = self.WTconv2d_1(w2_1)
    #     f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
    #     f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
    #     f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
    #     f1 = w2 + f1_3
        
    #     w3_1 = self.conv1_3(f1)
    #     w3 = self.WTconv2d_2(w3_1)
    #     f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
    #     f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
    #     f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
    #     f2 = w3 + f2_3
        
    #     w4_1 = self.conv1_4(f2)
    #     w4 = self.WTconv2d_3(w4_1)
    #     f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
    #     f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
    #     f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
    #     f3 = w4 + f3_3
        
    #     w5_1 = self.conv1_5(f3)
    #     w5 = self.WTconv2d_4(w5_1)
    #     f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
    #     f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
    #     f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
    #     f4 = w5 + f4_3
    #     # fusion1 = self.Fusion_2(SwinTouts[0],f1)
    #     # fusion2 = self.Fusion_4(SwinTouts[1],f2)
    #     # fusion3 = self.Fusion_8(SwinTouts[2],f3)
    #     # fusion4 = self.Fusion_16(SwinTouts[3],f4)
       
    #     m1 = torch.cat([f1_3, f1], dim=1)
    #     m2 = torch.cat([f2_3, f2], dim=1)
    #     m3 = torch.cat([f3_3, f3], dim=1)
    #     m4 = torch.cat([f4_3, f4], dim=1)
    #     return [m1, m2, m3, m4]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    # # 4
    # def forward(self, img): # B,3,512,512
        
    #     SwinTouts = self.SwinTransformer(img)
    #     img = self.WTconv2d(img)
    #     feat = self.downs[0](img) # torch.Size([2, 64, 512, 512])
    #     f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
    #     f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
    #     f0 = self.WTconv2d_0(f0_2)
    #     f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
    #     f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
    #     f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
    #     f1 = self.WTconv2d_1(f1_3)
    #     f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
    #     f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
    #     f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
    #     f2 = self.WTconv2d_2(f2_3)
    #     f3_1 = self.downs[9](f2_3) # torch.Size([2, 256, 64, 64])
    #     f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
    #     f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
    #     f3 = self.WTconv2d_3(f3_3)
    #     f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
    #     f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
    #     f4 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
    #     f4 = self.WTconv2d_4(f4)
    #     fusion1 = self.Fusion_2(SwinTouts[0],f1)
    #     fusion2 = self.Fusion_4(SwinTouts[1],f2)
    #     fusion3 = self.Fusion_8(SwinTouts[2],f3)
    #     fusion4 = self.Fusion_16(SwinTouts[3],f4)
       
    #     m1 = torch.cat([f1, fusion1], dim=1)
    #     m2 = torch.cat([f2, fusion2], dim=1)
    #     m3 = torch.cat([f3, fusion3], dim=1)
    #     m4 = torch.cat([f4, fusion4], dim=1)
    #     return [m1, m2, m3, m4]
    
    #3
    # def forward(self, img): # B,3,512,512
    #     SwinTouts = self.SwinTransformer(img)
    #     img = self.WTconv2d(img)
    #     feat = self.downs[0](img) # torch.Size([2, 64, 512, 512])
    #     f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
    #     f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
    #     f0 = self.WTconv2d_0(f0_2)
    #     f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
    #     f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
    #     f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
    #     f1 = self.WTconv2d_1(f1_3)
    #     f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
    #     f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
    #     f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
    #     f2 = self.WTconv2d_2(f2_3)
    #     f3_1 = self.downs[9](f2_3) # torch.Size([2, 256, 64, 64])
    #     f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
    #     f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
    #     f3 = self.WTconv2d_3(f3_3)
    #     f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
    #     f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
    #     f4 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
    #     f4 = self.WTconv2d_4(f4)
    #     # feat = self.downs[0](img) # 64,512,512
    #     # f0_1 = self.downs[4](feat) # 128,512,512
    #     # f1 = self.WTconv2d_0(f0_1) # 128,256,256
    #     # f1_1 = self.downs[7](f1) # 256,256,256
    #     # f2 = self.WTconv2d_1(f1_1)  # 256,128,128
    #     # f2_1 = self.downs[10](f2) # 512,128,128
    #     # f3 = self.WTconv2d_2(f2_1)  # 512,64,64
    #     # f3_1 = self.downs[13](f3) # 1024,64,64
    #     # f4 = self.WTconv2d_3(f3_1)  # 1024,32,32
    #     fusion1 = self.Fusion_2(SwinTouts[0],f1)
    #     fusion2 = self.Fusion_4(SwinTouts[1],f2)
    #     fusion3 = self.Fusion_8(SwinTouts[2],f3)
    #     fusion4 = self.Fusion_16(SwinTouts[3],f4)
       
    #     m1 = torch.cat([f1, fusion1], dim=1)
    #     m2 = torch.cat([f2, fusion2], dim=1)
    #     m3 = torch.cat([f3, fusion3], dim=1)
    #     m4 = torch.cat([f4, fusion4], dim=1)
    #     return [m1, m2, m3, m4]
        # print(f1.shape)
        
        
        # # fusion1 = self.Fusion_2(SwinTouts[0],f1)
        # # m3 = torch.cat([f3, fusion3], dim=1)
        # # m4 = torch.cat([f4, fusion4], dim=1)
        # return [m1, m2, m3, m4]
    # 1
    # def forward(self, img): # B,3,512,512
    #     SwinTouts = self.SwinTransformer(img) 
    #     # 96,256,256
    #     # 192,128,128
    #     # 384,64,64
    #     # 768,32,32
    #     feat = self.conv_body_first(img) # 512,32,32
    #     # g0 = self.condition_shift[0](feat)
    #     feat = self.conv_body_down[0](feat) # 512,32,32
    #     g1 = self.condition_shift[1](feat) # 128,256,256
    #     feat = self.conv_body_down[1](feat) # 512,32,32
    #     g2 = self.condition_shift[2](feat) # 256,128,128
    #     feat = self.conv_body_down[2](feat) # 512,32,32
    #     g3 = self.condition_shift[3](feat) # 512,64,64
    #     feat = self.conv_body_down[3](feat) # 512,32,32
    #     g4 = self.condition_shift[4](feat) # 1024,32,32
    #     fusion1 = self.Fusion_2(SwinTouts[0],g1)
    #     fusion2 = self.Fusion_4(SwinTouts[1],g2)
    #     fusion3 = self.Fusion_8(SwinTouts[2],g3)
    #     fusion4 = self.Fusion_16(SwinTouts[3],g4)
        # feat = self.downs[0](img) # torch.Size([2, 64, 512, 512])
        # f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
        # f0 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
        # f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
        # f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
        # f1 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
        # f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
        # f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
        # f2 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
        # f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
        # f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
        # f3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
        # f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
        # f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
        # f4 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
        # m1 = torch.cat([f1, f1], dim=1) # torch.Size([2, 256, 256, 256])
        # m2 = torch.cat([f2, f2], dim=1)
        # m3 = torch.cat([f3, f3], dim=1)
        # m4 = torch.cat([f4, f4], dim=1)
        # m1 = torch.cat([f1, f1], dim=1)
        # m2 = torch.cat([f2, f2], dim=1)
        # m3 = torch.cat([f3, f3], dim=1)
        # m4 = torch.cat([f4, f4], dim=1)
    #     # m1 = torch.cat([g1, g1], dim=1)
    #     # m2 = torch.cat([g2, g2], dim=1)
    #     # m3 = torch.cat([g3, g3], dim=1)
    #     # m4 = torch.cat([g4, g4], dim=1)
    #     m1 = torch.cat([g1, fusion1], dim=1)
    #     m2 = torch.cat([g2, fusion2], dim=1)
    #     m3 = torch.cat([g3, fusion3], dim=1)
    #     m4 = torch.cat([g4, fusion4], dim=1)
        # return [m1, m2, m3, m4]
    #     # return [f1, f2, f3, f4]
        
        
    # 2
    # def forward(self, img): # B,3,512,512
    #     SwinTouts = self.SwinTransformer(img) 
    #     # 96,256,256
    #     # 192,128,128
    #     # 384,64,64
    #     # 768,32,32
    #     # feat = self.conv_body_first(img) # 512,32,32
    #     # # g0 = self.condition_shift[0](feat)
    #     # feat = self.conv_body_down[0](feat) # 512,32,32
    #     # g1 = self.condition_shift[1](feat) # 128,256,256
    #     # feat = self.conv_body_down[1](feat) # 512,32,32
    #     # g2 = self.condition_shift[2](feat) # 256,128,128
    #     # feat = self.conv_body_down[2](feat) # 512,32,32
    #     # g3 = self.condition_shift[3](feat) # 512,64,64
    #     # feat = self.conv_body_down[3](feat) # 512,32,32
    #     # g4 = self.condition_shift[4](feat) # 1024,32,32
    #     feat = self.downs[0](img)
    #     f0_1 = self.downs[1](feat)
    #     f0 = self.downs[2](f0_1)
    #     f1_1 = self.downs[3](f0)
    #     f1_2 = self.downs[4](f1_1)
    #     f1 = self.downs[5](f1_2)
    #     f2_1 = self.downs[6](f1)
    #     f2_2 = self.downs[7](f2_1)
    #     f2 = self.downs[8](f2_2)
    #     f3_1 = self.downs[9](f2)
    #     f3_2 = self.downs[10](f3_1)
    #     f3 = self.downs[11](f3_2)
    #     f4_1 = self.downs[12](f3)
    #     f4_2 = self.downs[13](f4_1)
    #     f4 = self.downs[14](f4_2)
    #     fusion1 = self.Fusion_2(SwinTouts[0],f1)
    #     fusion2 = self.Fusion_4(SwinTouts[1],f2)
    #     fusion3 = self.Fusion_8(SwinTouts[2],f3)
    #     fusion4 = self.Fusion_16(SwinTouts[3],f4)
       
    #     m1 = torch.cat([f1, fusion1], dim=1)
    #     m2 = torch.cat([f2, fusion2], dim=1)
    #     m3 = torch.cat([f3, fusion3], dim=1)
    #     m4 = torch.cat([f4, fusion4], dim=1)
    #     return [m1, m2, m3, m4]


