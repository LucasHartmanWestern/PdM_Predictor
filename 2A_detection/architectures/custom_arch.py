import torch
from torch import nn


class DSConvSimple(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=ksize, stride=1, padding=pad, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


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


class FastVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(FastVGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = DSConvSimple(in_channels, middle_channels, ksize=3, pad=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = DSConvSimple(middle_channels, out_channels, ksize=3, pad=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
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
#   - current design is mega lightweight so maybe go deeper?
class QuickNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.l0_down = QuickBlock(input_channels, nb_filter[0], direction='down')  # 512 -> 256
        self.l1_down = QuickBlock(nb_filter[0], nb_filter[1], direction='down')  # 256 -> 128
        self.l2_down = QuickBlock(nb_filter[1], nb_filter[2], direction='down')  # 128 -> 64
        self.l3_down = QuickBlock(nb_filter[2], nb_filter[3], direction='down')  # 64 -> 32
        self.l4_down = QuickBlock(nb_filter[3], nb_filter[4], direction='down')  # 32 -> 16

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
        # up
        x_l3 = self.l3_up(torch.cat([x_l3, self.up(x_l4)], 1))  # C:128, S:32
        x_l2 = self.l2_up(torch.cat([x_l2, self.up(x_l3)], 1))  # C:64, S:64
        x_l1 = self.l1_up(torch.cat([x_l1, self.up(x_l2)], 1))  # C:32, S:128
        x_l0 = self.l0_up(torch.cat([x_l0, self.up(x_l1)], 1))  # C:16, S:256
        # out
        out = self.final(self.up(x_l0))  # C:1, S:512
        out = self.sigmoid(out)
        return out


class FastUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()

        # nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = FastVGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = FastVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = FastVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = FastVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = FastVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = FastVGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = FastVGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = FastVGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = FastVGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = self.sigmoid(output)
        return output


class FastNestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super(FastNestedUNet, self).__init__()

        # nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = FastVGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = FastVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = FastVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = FastVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = FastVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = FastVGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = FastVGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = FastVGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = FastVGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = FastVGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = FastVGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = FastVGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = FastVGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = FastVGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = FastVGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = DSConvSimple(nb_filter[0], num_classes, ksize=1, pad=0)
            self.final2 = DSConvSimple(nb_filter[0], num_classes, ksize=1, pad=0)
            self.final3 = DSConvSimple(nb_filter[0], num_classes, ksize=1, pad=0)
            self.final4 = DSConvSimple(nb_filter[0], num_classes, ksize=1, pad=0)
        else:
            self.final = DSConvSimple(nb_filter[0], num_classes, ksize=1, pad=0)

        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return output
