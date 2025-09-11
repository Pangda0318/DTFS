import torch
from torch import nn
from .Style import StyleLayer
from utils.utils import Uhalf_ResnetBlocWithAttn, Downsample
from .SwinTransformer import SwinTransformer
from .BasicBlock import Fusion
from utils.wtconv2d import WTConv2d
# from wtconv import WTConv2d

# conv_dw = WTConv2d(32, 32, kernel_size=5, wt_levels=3)
# self.encoder = half_IDN(in_channel=3, norm_groups=32, inner_channel=64, 
# channel_mults=[1,2,4,8,16], attn_res=[32], res_blocks=2, dropout=0.2, image_size=512)
class half_IDN(nn.Module):
    def __init__(self, in_channel=3, inner_channel=64, norm_groups=32, channel_mults=(1, 2, 4, 8, 16), attn_res=(32), res_blocks=2, dropout=0.2, image_size=512):
        super().__init__()
        self.SwinTransformer = SwinTransformer()
        self.Fusion_2 = Fusion(128, 96)
        self.Fusion_4 = Fusion(256, 192)
        self.Fusion_8 = Fusion(512, 384)
        self.Fusion_16 = Fusion(1024, 768)
        
        # #swin主路
        # self.Fusion_2 = Fusion(96, 128)
        # self.Fusion_4 = Fusion(192, 256)
        # self.Fusion_8 = Fusion(384, 512)
        # self.Fusion_16 = Fusion(768, 1024)
        
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv0 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        
        
        self.WTconv2d = WTConv2d(3,3,stride=1, wt_levels=3)
        self.WTconv2d_0 = WTConv2d(64,64,stride=1, wt_levels=3)
        self.WTconv2d_1 = WTConv2d(128,128,stride=2, wt_levels=3)
        self.WTconv2d_2 = WTConv2d(256,256,stride=2, wt_levels=3)
        self.WTconv2d_3 = WTConv2d(512,512,stride=2, wt_levels=3)
        self.WTconv2d_4 = WTConv2d(1024,1024,stride=2, wt_levels=3)
        
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=1)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv1_3 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv1_4 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv1_5 = nn.Conv2d(512, 1024, kernel_size=1)
        
        
        self.conv2_1 = nn.Conv2d(96,128,kernel_size=1)
        self.conv2_2 = nn.Conv2d(192,256,kernel_size=1)
        self.conv2_3 = nn.Conv2d(384,512,kernel_size=1)
        self.conv2_4 = nn.Conv2d(768,1024,kernel_size=1)
        
        
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
        
    # new_4 cnn加上WT +swint 
    def forward(self, img): # B,3,512,512
        
        SwinTouts = self.SwinTransformer(img)
        # img = self.WTconv2d(img)
        w = self.conv1_1(img)
        w1 = self.WTconv2d_0(w)

        feat = self.downs[0](img) # torch.Size([2, 64, 512, 512])
        f0_1 = self.downs[1](feat) # torch.Size([2, 64, 512, 512])
        f0_2 = self.downs[2](f0_1) # torch.Size([2, 64, 512, 512])
        f0 = w1 + f0_2
        
        w2_1 = self.conv1_2(f0)
        w2 = self.WTconv2d_1(w2_1)
        f1_1 = self.downs[3](f0) # torch.Size([2, 64, 256, 256])
        f1_2 = self.downs[4](f1_1)# torch.Size([2, 128, 256, 256])
        f1_3 = self.downs[5](f1_2) # torch.Size([2, 128, 256, 256])
        f1 = w2 + f1_3
        
        w3_1 = self.conv1_3(f1)
        w3 = self.WTconv2d_2(w3_1)
        f2_1 = self.downs[6](f1) # torch.Size([2, 128, 128, 128])
        f2_2 = self.downs[7](f2_1) # torch.Size([2, 256, 128, 128])
        f2_3 = self.downs[8](f2_2) # torch.Size([2, 256, 128, 128])
        f2 = w3 + f2_3
        
        w4_1 = self.conv1_4(f2)
        w4 = self.WTconv2d_3(w4_1)
        f3_1 = self.downs[9](f2) # torch.Size([2, 256, 64, 64])
        f3_2 = self.downs[10](f3_1) # torch.Size([2, 512, 64, 64])
        f3_3 = self.downs[11](f3_2) # torch.Size([2, 512, 64, 64])
        f3 = w4 + f3_3
        
        w5_1 = self.conv1_5(f3)
        w5 = self.WTconv2d_4(w5_1)
        f4_1 = self.downs[12](f3) # torch.Size([2, 512, 32, 32])
        f4_2 = self.downs[13](f4_1) # torch.Size([2, 1024, 32, 32])
        f4_3 = self.downs[14](f4_2) # torch.Size([2, 1024, 32, 32])
        f4 = w5 + f4_3
        fusion1 = self.Fusion_2(SwinTouts[0],f1)
        fusion2 = self.Fusion_4(SwinTouts[1],f2)
        fusion3 = self.Fusion_8(SwinTouts[2],f3)
        fusion4 = self.Fusion_16(SwinTouts[3],f4)
       
        m1 = torch.cat([fusion1, f1], dim=1)
        m2 = torch.cat([fusion2, f2], dim=1)
        m3 = torch.cat([fusion3, f3], dim=1)
        m4 = torch.cat([fusion4, f4], dim=1)
        return [m1, m2, m3, m4]
    
    
    
    
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
    
    
    # # new_2 cnn加上WT +swint 
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


