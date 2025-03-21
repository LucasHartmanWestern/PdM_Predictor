import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, dil):
        super().__init__()
        pad = int((ksize - 1) / 2) * dil
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=pad, dilation=dil, bias=False)

    def forward(self, x):
        out = self.conv(x)
        return out


class BNPReLU(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        out = self.bn(x)
        out = self.act(out)
        return out


class ConvBNPReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride):
        super().__init__()
        pad = int((ksize - 1) / 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        n = int(out_ch / 2)
        self.conv_start = ConvBNPReLU(in_ch, n, ksize=3, stride=2)
        self.conv3x3 = Conv(n, n, ksize=3, stride=1, dil=1)
        self.conv3x3_dilated = Conv(n, n, ksize=3, stride=1, dil=2)
        self.bnprelu = BNPReLU(out_ch)

    def forward(self, x):
        out = self.conv_start(x)
        out1 = self.conv3x3(out)
        out2 = self.conv3x3_dilated(out)
        out = self.bnprelu(torch.cat([out1, out2], dim=1))
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBNPReLU(in_ch, out_ch, ksize=3, stride=1),
            ConvBNPReLU(out_ch, out_ch, ksize=3, stride=1)
        )

    def forward(self, x):
        out = self.double_conv(x)
        return out


# TODO:
#   - test new model
class NickNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256, 512, 1024]

        self.start_block = ConvBNPReLU(input_channels, nb_filter[0], ksize=3, stride=2)   # 512 -> 256
        self.down_block_2 = DownBlock(nb_filter[0], nb_filter[1])  # 256 -> 128
        self.down_block_3 = DownBlock(nb_filter[1], nb_filter[2])  # 128 -> 64
        self.down_block_4 = DownBlock(nb_filter[2], nb_filter[3])  # 64 -> 32
        self.down_block_5 = DownBlock(nb_filter[3], nb_filter[4])  # 32 -> 16
        self.down_block_6 = DownBlock(nb_filter[4], nb_filter[5])  # 16 -> 8
        self.down_block_7 = DownBlock(nb_filter[5], nb_filter[6])  # 8 -> 4
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block_6 = UpBlock(nb_filter[6] + nb_filter[5], nb_filter[5])
        self.up_block_5 = UpBlock(nb_filter[5] + nb_filter[4], nb_filter[4])
        self.up_block_4 = UpBlock(nb_filter[4] + nb_filter[3], nb_filter[3])
        self.up_block_3 = UpBlock(nb_filter[3] + nb_filter[2], nb_filter[2])
        self.up_block_2 = UpBlock(nb_filter[2] + nb_filter[1], nb_filter[1])
        self.up_block_1 = UpBlock(nb_filter[1] + nb_filter[0], nb_filter[0])
        self.final_block = Conv(nb_filter[0], num_classes, ksize=3, stride=1, dil=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.start_block(x)  # C:16, S:256
        x2 = self.down_block_2(x1)  # C:32, S:128
        x3 = self.down_block_3(x2)  # C:64, S:64
        x4 = self.down_block_4(x3)  # C:128, S:32
        x5 = self.down_block_5(x4)  # C:256, S:16
        x6 = self.down_block_6(x5)  # C:512, S:8
        x7 = self.down_block_7(x6)  # C:1024, S:4

        x6 = self.up_block_6(torch.cat([x6, self.up(x7)], 1))  # C:512, S:8
        x5 = self.up_block_5(torch.cat([x5, self.up(x6)], 1))  # C:256, S:16
        x4 = self.up_block_4(torch.cat([x4, self.up(x5)], 1))  # C:128, S:32
        x3 = self.up_block_3(torch.cat([x3, self.up(x4)], 1))  # C:64, S:64
        x2 = self.up_block_2(torch.cat([x2, self.up(x3)], 1))  # C:32, S:128
        x1 = self.up_block_1(torch.cat([x1, self.up(x2)], 1))  # C:16, S:256

        out = self.final_block(self.up(x1))  # C:1, S:512
        out = self.sigmoid(out)
        return out
