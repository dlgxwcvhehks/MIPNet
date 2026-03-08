import torch
import torch.nn as nn
import torch.nn.functional as F
from .NAFNet.arch_util import LayerNorm2d
from .NAFNet.local_arch import Local_Base
from .depth_anything_v2.dpt import DepthAnythingV2
import os

class DepthModel:
    def __init__(self, weight_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_configs = {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384]
        }

        self.model = DepthAnythingV2(**model_configs).to(self.device)

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"file not found: {weight_path}")

        try:
            state_dict = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise e

        self.model.eval()

    def get_depth_map(self, input_tensor):
        """
        Processes the input tensor and generates a depth map.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: The generated depth map.
        """
        input_tensor = input_tensor.to(self.device)

        patch_size = 14
        _, _, H, W = input_tensor.shape

        new_H = (H + patch_size - 1) // patch_size * patch_size
        new_W = (W + patch_size - 1) // patch_size * patch_size

        input_tensor = F.interpolate(input_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)

        with torch.no_grad():
            depth_map = self.model(input_tensor)
            depth_map = depth_map.unsqueeze(1)

        depth_map = F.interpolate(depth_map, size=(H, W), mode='bilinear', align_corners=False)

        min_val = depth_map.min()
        max_val = depth_map.max()
        depth_map = (depth_map - min_val) / (max_val - min_val + 1e-8)
        return depth_map



class GAFE(nn.Module):
    def __init__(self, width, relu_slope=0.2, use_HIN=True):
        super(GAFE, self).__init__()

        self.conv_R = nn.Conv2d(width, width, kernel_size=1, bias=True)

        self.conv_1 = nn.Conv2d(1, width // 2, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.identity = nn.Conv2d(width, width, 1, 1, 0)

        self.gamma_conv = nn.Conv2d(width, width, kernel_size=1, bias=True)
        self.beta_conv = nn.Conv2d(width, width, kernel_size=1, bias=True)


        if use_HIN:
            self.norm = nn.InstanceNorm2d(width//4, affine=True)
        self.use_HIN = use_HIN


        nn.init.constant_(self.gamma_conv.weight, 0)
        nn.init.constant_(self.gamma_conv.bias, 0)
        nn.init.constant_(self.beta_conv.weight, 0)
        nn.init.constant_(self.beta_conv.bias, 0)

    def forward(self, x, x_depth):

        x = self.conv_R(x)  # (B, C, H, W)

        depth_features = self.conv_1(x_depth)
        if self.use_HIN:
            depth_features_1, depth_features_2 = torch.chunk(depth_features, 2, dim=1)
            depth_features = torch.cat([self.norm(depth_features_1), depth_features_2], dim=1)
        depth_features = self.relu_1(depth_features)
        depth_features = self.relu_2(self.conv_2(depth_features))
        depth_features = depth_features + self.identity(depth_features)

        gamma = torch.sigmoid(self.gamma_conv(depth_features))  # (B, C, H, W)
        beta = self.beta_conv(depth_features)  # (B, C, H, W)

        out = gamma * x + beta  # (B, C, H, W)

        return out



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class PhaseFeatureExtractorWithIFFT(nn.Module):
    def __init__(self, nc):
        super(PhaseFeatureExtractorWithIFFT, self).__init__()
        self.nc = nc
        self.gate = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))  # 高频注入初值
        self.image_index = 0  # 初始化图像索引
    def _process_single_scale(self, x_single: torch.Tensor):

        B, C, H, W = x_single.shape
        x_fp32 = x_single.to(torch.float32)

        fft = torch.fft.fft2(x_fp32)
        fft_shift = torch.fft.fftshift(fft)
        phase = torch.angle(fft_shift)

        u = torch.arange(W, device=x_single.device) - W / 2
        v = torch.arange(H, device=x_single.device) - H / 2
        U, V = torch.meshgrid(u, v, indexing='xy')
        D = torch.sqrt(U ** 2 + V ** 2).to(x_single.device)  # (H,W)
        D = D.unsqueeze(0).unsqueeze(0)  # (1,1,H,W) broadcast到(B,C,H,W)

        low_radius = min(H, W) * 0.1
        high_mask = (D > low_radius).float()
        low_mask = 1.0 - high_mask

        real_high = torch.cos(phase) * high_mask
        imag_high = torch.sin(phase) * high_mask
        fft_high = torch.complex(real_high, imag_high)
        fft_high = torch.fft.ifftshift(fft_high)
        img_high = torch.real(torch.fft.ifft2(fft_high))

        real_low = torch.cos(phase) * low_mask
        imag_low = torch.sin(phase) * low_mask
        fft_low = torch.complex(real_low, imag_low)
        fft_low = torch.fft.ifftshift(fft_low)
        img_low = torch.real(torch.fft.ifft2(fft_low))

        gate = torch.sigmoid(self.gate)
        img_combined = img_low + gate * img_high
        return img_combined


    def forward(self, x):
        B, C, H, W = x.shape

        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)


        feat_orig = self._process_single_scale(x)           # (B,C,H,W)
        feat_half = self._process_single_scale(x_half)      # (B,C,H/2,W/2)
        feat_quarter = self._process_single_scale(x_quarter)  # (B,C,H/4,W/4)


        feat_half_up = F.interpolate(feat_half, size=(H, W), mode='bilinear', align_corners=False)
        feat_quarter_up = F.interpolate(feat_quarter, size=(H, W), mode='bilinear', align_corners=False)


        feat_multi_scale = torch.cat([feat_orig, feat_half_up, feat_quarter_up], dim=1)  # (B, C*3, H, W)

        return feat_multi_scale


class NAFNet(nn.Module):

    def __init__(self, weight_path,  img_channel=3, width=8, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.phase_feature = PhaseFeatureExtractorWithIFFT(width)
        self.phase_fusions = nn.ModuleList()
        self.sam12 = GAFE(width)
        self.depth_model = DepthModel(weight_path)

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.phase_fusions = nn.ModuleList([
            nn.Conv2d(chan * 3, chan * 8, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(chan * 3, chan * 4, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(chan * 3, chan * 2, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(chan * 3, chan, kernel_size=3, padding=1, bias=True),
        ])

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        depth_map = self.depth_model.get_depth_map(inp)
        x = self.intro(inp)
        x = self.sam12(x, depth_map)
        phase_features = self.phase_feature(x)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)


        x = self.middle_blks(x)

        for decoder, up, enc_skip, phase_fusion in zip(self.decoders, self.ups, encs[::-1], self.phase_fusions):
            x = up(x)

            phase_resized = F.interpolate(phase_features, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + 0.1 * phase_fusion(phase_resized)
            x = x + enc_skip
            x = decoder(x)


        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W], depth_map, phase_features


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    enc_blks = [1, 1, 1, 1]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                 enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)


