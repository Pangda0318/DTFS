import torch.nn as nn
from .modules.SFFI import SFFI
from .modules.MIFA import MIFA
from torch.nn import functional as F
from .modules.half_IDN_abation_noWT import half_IDN
from .modules.SwinTransformer import SwinTransformer
from .modules.BasicBlock import ScaleEncoder, Fusion, BasicConv, InputEncoder, MiddleEncoder, MiddleDecoder, OutputDecoder,ResBlock1,ResBlock2,ResBlock3




# all have 版本
class ISDTDNet(nn.Module):

    def __init__(self, args):
        super(ISDTDNet, self).__init__()
        self.encoder = half_IDN(in_channel=3, norm_groups=32, inner_channel=64, channel_mults=[1,2,4,8,16], attn_res=[32], res_blocks=2, dropout=0.2, image_size=args.base_size)
        self.MIFA = MIFA()
        self.SFFI = SFFI(input_dim=3, hidden_dim=64)
        self.Resblock = ResBlock3(64, 64)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=1)
        self.conv1_2 = nn.Conv2d(64, 3, kernel_size=1)
        #self.conv1_3 = nn.Conv2d(3,1,kernel_size = 1)
        
    def forward(self, img):
        _, _, h, w = img.size()
        feat= self.encoder(img)
        out = self.MIFA(feat)
        resize_out = F.upsample(input=out, size=(h, w), mode='bilinear', align_corners=True)
        mask = self.SFFI(resize_out)
        
        return mask
