import torch
from torch import nn


class DSConvComplex(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, dilation=1):
        super().__init__()
        pad = int((ksize - 1) / 2) * dilation
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
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


class BNPReLU(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        out = self.bn(x)
        out = self.act(out)
        return out


class QuickBlock(nn.Module):
    def __init__(self, in_ch, out_ch, direction):
        super().__init__()
        assert direction in ['up', 'down'], 'ERROR: incorrect direction parameter for QuickBlock'
        n = int(out_ch / 2)
        if direction == 'up':
            self.conv_start = ConvBNPReLU(in_ch, n, ksize=1, stride=1)
        else:
            self.conv_start = ConvBNPReLU(in_ch, n, ksize=3, stride=2)
        self.conv3x3 = DSConvComplex(n, n, ksize=3, stride=1, dilation=1)
        self.conv3x3_dilated = DSConvComplex(n, n, ksize=3, stride=1, dilation=2)
        self.bnprelu = BNPReLU(out_ch)

    def forward(self, x):
        out = self.conv_start(x)
        out1 = self.conv3x3(out)
        out2 = self.conv3x3_dilated(out)
        out = self.bnprelu(torch.cat([out1, out2], dim=1))
        return out


# TODO:
#   - testing results were only 0.68 (avg jaccard index) :(
class QuickNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256, 512, 1024]

        self.l0_down = QuickBlock(input_channels, nb_filter[0], direction='down')  # 512 -> 256
        self.l1_down = QuickBlock(nb_filter[0], nb_filter[1], direction='down')  # 256 -> 128
        self.l2_down = QuickBlock(nb_filter[1], nb_filter[2], direction='down')  # 128 -> 64
        self.l3_down = QuickBlock(nb_filter[2], nb_filter[3], direction='down')  # 64 -> 32
        self.l4_down = QuickBlock(nb_filter[3], nb_filter[4], direction='down')  # 32 -> 16
        self.l5_down = QuickBlock(nb_filter[4], nb_filter[5], direction='down')  # 16 -> 8
        self.l6_down = QuickBlock(nb_filter[5], nb_filter[6], direction='down')  # 8 -> 4

        self.l5_up = QuickBlock(nb_filter[6] + nb_filter[5], nb_filter[5], direction='up')
        self.l4_up = QuickBlock(nb_filter[5] + nb_filter[4], nb_filter[4], direction='up')
        self.l3_up = QuickBlock(nb_filter[4] + nb_filter[3], nb_filter[3], direction='up')
        self.l2_up = QuickBlock(nb_filter[3] + nb_filter[2], nb_filter[2], direction='up')
        self.l1_up = QuickBlock(nb_filter[2] + nb_filter[1], nb_filter[1], direction='up')
        self.l0_up = QuickBlock(nb_filter[1] + nb_filter[0], nb_filter[0], direction='up')
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # down
        x_l0 = self.l0_down(x)  # C:16, S:256
        x_l1 = self.l1_down(x_l0)  # C:32, S:128
        x_l2 = self.l2_down(x_l1)  # C:64, S:64
        x_l3 = self.l3_down(x_l2)  # C:128, S:32
        x_l4 = self.l4_down(x_l3)  # C:256, S:16
        x_l5 = self.l5_down(x_l4)  # C:512, S:8
        x_l6 = self.l6_down(x_l5)  # C:1024, S:4
        # up
        x_l5 = self.l5_up(torch.cat([x_l5, self.up(x_l6)], 1))  # C:512, S:8
        x_l4 = self.l4_up(torch.cat([x_l4, self.up(x_l5)], 1))  # C:256, S:16
        x_l3 = self.l3_up(torch.cat([x_l3, self.up(x_l4)], 1))  # C:128, S:32
        x_l2 = self.l2_up(torch.cat([x_l2, self.up(x_l3)], 1))  # C:64, S:64
        x_l1 = self.l1_up(torch.cat([x_l1, self.up(x_l2)], 1))  # C:32, S:128
        x_l0 = self.l0_up(torch.cat([x_l0, self.up(x_l1)], 1))  # C:16, S:256
        # out
        out = self.final(self.up(x_l0))  # C:1, S:512
        out = self.sigmoid(out)
        return out
