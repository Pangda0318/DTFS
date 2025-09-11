import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_, DropPath



class SwinTransformer(nn.Module): # in_channels=4
    def __init__(self, img_size=512, patch_size=2, in_channels=3, embed_dim=96, ape=False, drop_rate=0, depths=[2, 2, 2, 2], window_size=7, num_heads=[3, 6, 12, 24], drop_path_rate=0.2):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.ape = ape
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in range(self.num_layers):
            layer = BasicLayer(window_size=window_size, 
                               input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                               dim=embed_dim * 2 ** i_layer, 
                               depth=depths[i_layer], 
                               num_heads=num_heads[i_layer],  
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])])
            self.layers.append(layer)
            norm_layer = nn.LayerNorm(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, norm_layer)


    def forward(self, x):
        x = self.patch_embed(x) # x:[B, 128*128, 96]
        if self.ape:
            x = x + self.absolute_pos_embed # still [B, 128*128, 96]
        x = self.pos_drop(x) # x [B, 128*128, 96]
        SwinTouts=[]
        for i in range(self.num_layers):
            input_resolution=(self.patches_resolution[0] // (2 ** i), self.patches_resolution[1] // (2 ** i))
            H, W = input_resolution
            layer = self.layers[i]
            out, x = layer(x) # [B, 128*128, 96] [B, 64*64, 96]
            norm_layer = getattr(self, f'norm{i}')
            out = norm_layer(out)
            out = out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()   # (B, num_features[i], H, W)
            SwinTouts.append(out)
        return tuple(SwinTouts)


        

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, L, C = x.shape # x [B, 128*128, 96]
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C) # [B, 128, 128, 96]
        pad_input = (H % 2 == 1 ) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C stride=2 [B, 64, 64, 96]
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C [B, 64, 64, 4*96]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C [B, 64*64, 4*96]
        x = self.norm(x)
        x = self.reduction(x)
        return x # [B, 64*64, 2*96]


        


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=2, in_channels=4, embed_dim=96):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        img_size = to_2tuple(img_size)
        patches_resolution = [img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]] # [128,128]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

    

    def forward(self, x):
        _, _, H, W = x.shape # x [B, 4, 256, 256]
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0]- H % self.patch_size[0]))
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        x = self.proj(x) # x [B, 96, 128, 128]
        # H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2) # x [B, 128*128, 96]
        x = self.norm(x) # x [B, 128*128, 96]
        # x = x.transpose(1, 2).view(-1, self.embed_dim, H, W)
        return x
    


def window_partition(x, window_size):
    B, H, W, C = x.shape # [1, 133, 133, 1] # x [B, 128, 128, 96] or [1, 128, 128, 1]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) # [1, 19, 7, 19 , 7, 1] # # x [B, 18, 7, 18, 7, 96]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) # windows: (num_windows*B, window_size, window_size, C) [361, 7, 7, 1] [324*B,7,7,96] [324, 7, 7, 1]
    return windows  # [361, 7, 7, 1]

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size)) # windows: [361*B, 7, 7, 96]  361*B / (133*13/7/7)
    # B = windows.shape[0] / (H // window_size) / (W // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1) # [B, 19, 19, 7, 7, 96]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1) # [B, 133, 133, 96]
    return x 

class BasicLayer(nn.Module):
    def __init__(self, window_size, input_resolution, dim, depth, num_heads, downsample, drop_path):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList([SwinTransformerBlock(input_resolution=input_resolution, dim=dim, shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                          window_size=window_size, num_heads=num_heads, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                          )
                                     for i in range(depth)])
        self.input_resolution = input_resolution
        if downsample is not None:
            self.downsample = downsample(input_resolution=input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x) # 0 [B, 128*128, 96]
        if self.downsample is not None:
            x_down = self.downsample(x) # [B, 64*64, 2*96]
            return x, x_down
        else:
            return x, x



class SwinTransformerBlock(nn.Module):
    def __init__(self, input_resolution, dim, shift_size, window_size, num_heads, drop_path):
        super().__init__()
        self.input_resolution = input_resolution
        self.norm1 = nn.LayerNorm(dim)
        self.shift_size = shift_size
        self.window_size = window_size
        if self.shift_size > 0:
            H, W = 512,512
            # fly-modify resolution
            Hp = int(np.ceil(H / self.window_size)) * self.window_size # 19 * 7 = 133
            Wp = int(np.ceil(W / self.window_size)) * self.window_size # 19 * 7 = 133
            img_mask = torch.zeros((1, Hp, Wp, 1)) # [1, 133, 133, 1]
            # img_mask = torch.zeros((1, H, W, 1)) # [1, 128, 128, 1]
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] =cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size) # [361, 7, 7, 1]
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [361, 7*7]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)# attention (nW*C, 1, window_size*window_size)-(nW*C, window_size*window_size, 1)
                                                                            # = (nW*C, window_size*window_size, window_size*window_size) [361, 7*7, 7*7]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) # !=0 set as -100 [361, 7*7, 7*7]
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads, window_size=to_2tuple(window_size))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        B, L, C = x.shape # x [B, 128*128, 96]
        H, W = self.input_resolution # 128, 128
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x) # x [B, 128*128, 96] # nn.LayerNorm automatically determines based on input size
        x = x.view(B, H, W, C) # x [B, 128, 128, 96]
        # fly-modify resolution
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size # 5
        pad_b = (self.window_size - H % self.window_size) % self.window_size # 5
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b)) # [B, 133, 133, 96]
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # x [B, 133, 133, 96]
            x_windows = window_partition(shifted_x, self.window_size) # [361*B,7,7,96]
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size) # [361*B,7,7,96]

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # [361*B, 7*7, 96]
        attn_windows = self.attn(x_windows, attn_mask=self.attn_mask) # [361*B, 7*7, 96]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # [361*B, 7, 7, 96]
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp) # [B, 133, 133, 96]
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp) # [B, 133, 133, 96]
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C) # [B, 128*128, 96]
        x = shortcut + self.drop_path(x)
        return x # [B, 128*128, 96]
        



class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias = True)
        self.num_heads = num_heads
        head_dim = dim // num_heads # 48
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH [13*13, 2]
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, 7, 7]
        coords_flatten = torch.flatten(coords, 1)  # [2, 7*7]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, 7*7, 7*7]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [7*7, 7*7, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [7*7, 7*7]
        self.register_buffer("relative_position_index", relative_position_index)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)

        
    def forward(self, x, attn_mask=None):
        B_, N, C =x.shape # [324*B, 7*7, 96]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [3, 324*B, 2, 7*7, 48]
        q, k, v = qkv[0], qkv[1], qkv[2] # [324*B, 2, 7*7, 48]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # @:Matrix multiplication [324*B, 2, 7*7, 7*7]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # [7*7,7*7,2]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [2, 7*7, 7*7]
        attn = attn + relative_position_bias.unsqueeze(0) # boardcast [324*B, 2, 7*7, 7*7]
        if attn_mask is not None:
            nW = attn_mask.shape[0] #[324, 7*7, 7*7]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0) # [B, 324, 2, 7*7, 7*7]+[1,324,1,7*7,7*7]=[B,324,2,7*7,7*7]
            attn = attn.view(-1, self.num_heads, N, N) # [324*B, 2, 7*7, 7*7]
            attn = self.softmax(attn) # [324*B, 2, 7*7, 7*7]
        else:
            attn = self.softmax(attn) # [324*B, 2, 7*7, 7*7]
        attn = self.attn_drop(attn) # [324*B, 2, 7*7, 7*7]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) # @:[324*B, 2, 7*7, 48] transpose:[324*B, 7*7, 2, 48] reshape:[324*B, 7*7, 96]
        x = self.proj(x) # [324*B, 7*7, 96]
        x = self.proj_drop(x) # [324*B, 7*7, 96]
        return x # [324*B, 7*7, 96]





if __name__ == "__main__":
    net = SwinTransformer()
    input_tensor = torch.randn(1,4,256,256)
    output_tensor = net(input_tensor)
    print(output_tensor[0].shape) # [1,96,128,128]
    print(output_tensor[1].shape) # [1,192,64,64]
    print(output_tensor[2].shape) # [1,384,32,32]







