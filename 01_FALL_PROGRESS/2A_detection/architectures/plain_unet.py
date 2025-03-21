import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on code obtained from:
# https://github.com/milesial/Pytorch-UNet/tree/master


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super(UNet, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        self.inc = (DoubleConv(input_channels, nb_filter[0]))  # in: 3, out: 64
        self.down1 = (Down(nb_filter[0], nb_filter[1]))  # in: 64, out: 128
        self.down2 = (Down(nb_filter[1], nb_filter[2]))  # in: 128, out: 256
        self.down3 = (Down(nb_filter[2], nb_filter[3]))  # in: 256, out: 512
        self.down4 = (Down(nb_filter[3], nb_filter[4]))  # in: 512, out: 1024
        self.up1 = (Up(nb_filter[4], nb_filter[3]))  # in: 1024, out: 512
        self.up2 = (Up(nb_filter[3], nb_filter[2]))  # in: 512, out: 256
        self.up3 = (Up(nb_filter[2], nb_filter[1]))  # in: 256, out: 128
        self.up4 = (Up(nb_filter[1], nb_filter[0]))  # in: 128, out: 64
        self.outc = (OutConv(nb_filter[0], num_classes))  # in: 64, out: 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.sigmoid(x)
