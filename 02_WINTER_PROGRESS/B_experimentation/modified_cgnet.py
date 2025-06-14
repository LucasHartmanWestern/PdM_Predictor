###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn
# Copyright (c) 2018
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_processing import preprocess_sample

__all__ = ["Context_Guided_Network"]

# ---------- Base Modules ---------- #

class ConvBNPReLU(nn.Module):
	def __init__(self, nIn, nOut, kSize, stride=1):
		super().__init__()
		padding = int((kSize - 1) / 2)
		self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
		self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
		self.act = nn.PReLU(nOut)

	def forward(self, input):
		output = self.conv(input)
		output = self.bn(output)
		output = self.act(output)
		return output

class BNPReLU(nn.Module):
	def __init__(self, nOut):
		super().__init__()
		self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
		self.act = nn.PReLU(nOut)

	def forward(self, input):
		output = self.bn(input)
		output = self.act(output)
		return output

class ConvBN(nn.Module):
	def __init__(self, nIn, nOut, kSize, stride=1):
		super().__init__()
		padding = int((kSize - 1) / 2)
		self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
		self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

	def forward(self, input):
		output = self.conv(input)
		output = self.bn(output)
		return output

class Conv(nn.Module):
	def __init__(self, nIn, nOut, kSize, stride=1):
		super().__init__()
		padding = int((kSize - 1) / 2)
		self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

	def forward(self, input):
		output = self.conv(input)
		return output

class ChannelWiseConv(nn.Module):
	def __init__(self, nIn, nOut, kSize, stride=1):
		super().__init__()
		padding = int((kSize - 1) / 2)
		self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn,
		                      bias=False)

	def forward(self, input):
		output = self.conv(input)
		return output

class DilatedConv(nn.Module):
	def __init__(self, nIn, nOut, kSize, stride=1, d=1):
		super().__init__()
		padding = int((kSize - 1) / 2) * d
		self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

	def forward(self, input):
		output = self.conv(input)
		return output

class ChannelWiseDilatedConv(nn.Module):
	def __init__(self, nIn, nOut, kSize, stride=1, d=1):
		super().__init__()
		padding = int((kSize - 1) / 2) * d
		self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False, dilation=d)

	def forward(self, input):
		output = self.conv(input)
		return output

class FGlo(nn.Module):
	def __init__(self, channel, reduction=16):
		super(FGlo, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y

# ---------- Block Modules ---------- #

class ContextGuidedBlock_Down(nn.Module):
	def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
		super().__init__()
		self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)  # size/2, channel: nIn--->nOut
		self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
		self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)
		self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
		self.act = nn.PReLU(2 * nOut)
		self.reduce = Conv(2 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut
		self.F_glo = FGlo(nOut, reduction)

	def forward(self, input):
		output = self.conv1x1(input)
		loc = self.F_loc(output)
		sur = self.F_sur(output)
		joi_feat = torch.cat([loc, sur], 1)  # the joint feature
		joi_feat = self.bn(joi_feat)
		joi_feat = self.act(joi_feat)
		joi_feat = self.reduce(joi_feat)  # channel= nOut
		output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
		return output

class ContextGuidedBlock(nn.Module):
	def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
		super().__init__()
		n = int(nOut / 2)
		self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)  # 1x1 Conv is employed to reduce the computation
		self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
		self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate)  # surrounding context
		self.bn_prelu = BNPReLU(nOut)
		self.add = add
		self.F_glo = FGlo(nOut, reduction)

	def forward(self, input):
		output = self.conv1x1(input)
		loc = self.F_loc(output)
		sur = self.F_sur(output)
		joi_feat = torch.cat([loc, sur], 1)
		joi_feat = self.bn_prelu(joi_feat)
		output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
		# if residual version
		if self.add:
			output = input + output
		return output

class InputInjection(nn.Module):
	def __init__(self, downsamplingRatio):
		super().__init__()
		self.pool = nn.ModuleList()
		for i in range(0, downsamplingRatio):
			self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

	def forward(self, input):
		for pool in self.pool:
			input = pool(input)
		return input

# ---------- Network Module ---------- #

class Modified_CGNet(nn.Module):
	def __init__(self, num_classes, use_rois, resize_shape=None):
		"""
		args:
		  num_classes: number of classes in the dataset. Default is 19 for the cityscapes
		  M: the number of blocks in stage 2
		  N: the number of blocks in stage 3
		"""
		super().__init__()
		self.use_rois = use_rois
		self.resize_shape = resize_shape
		self.num_classes = num_classes

		# hard-coded parameters
		M = 3
		N = 21
		dropout_flag = False

		# define layers
		self.level1_0 = ConvBNPReLU(3, 32, 3, 2)  # feature map size divided 2, 1/2
		self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
		self.level1_2 = ConvBNPReLU(32, 32, 3, 1)

		self.sample1 = InputInjection(1)  # down-sample for Input Injection, factor=2
		self.sample2 = InputInjection(2)  # down-sample for Input Injection, factor=4

		self.b1 = BNPReLU(32 + 3)

		# stage 2
		self.level2_0 = ContextGuidedBlock_Down(32 + 3, 64, dilation_rate=2, reduction=8)
		self.level2 = nn.ModuleList()
		for i in range(0, M - 1):
			self.level2.append(ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8))  # CG block
		self.bn_prelu_2 = BNPReLU(128 + 3)

		# stage 3
		self.level3_0 = ContextGuidedBlock_Down(128 + 3, 128, dilation_rate=4, reduction=16)
		self.level3 = nn.ModuleList()
		for i in range(0, N - 1):
			self.level3.append(ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16))  # CG block
		self.bn_prelu_3 = BNPReLU(256)

		if dropout_flag:
			print("have droput layer")
			self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), Conv(256, num_classes, 1, 1))
		else:
			self.classifier = nn.Sequential(Conv(256, num_classes, 1, 1))

		# init weights
		for m in self.modules():
			classname = m.__class__.__name__
			if classname.find('Conv2d') != -1:
				nn.init.kaiming_normal_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()
				elif classname.find('ConvTranspose2d') != -1:
					nn.init.kaiming_normal_(m.weight)
					if m.bias is not None:
						m.bias.data.zero_()

	def forward(self, input):
		"""
		args:
			input: Receives the input RGB image
			return: segmentation map
		"""
		# pre-process input
		input = preprocess_sample(input, self.use_rois, self.resize_shape)

		# stage 1
		output0 = self.level1_0(input)
		output0 = self.level1_1(output0)
		output0 = self.level1_2(output0)
		inp1 = self.sample1(input)
		inp2 = self.sample2(input)

		# stage 2
		output0_cat = self.b1(torch.cat([output0, inp1], 1))
		output1_0 = self.level2_0(output0_cat)  # down-sampled

		for i, layer in enumerate(self.level2):
			if i == 0:
				output1 = layer(output1_0)
			else:
				output1 = layer(output1)

		output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

		# stage 3
		output2_0 = self.level3_0(output1_cat)  # down-sampled
		for i, layer in enumerate(self.level3):
			if i == 0:
				output2 = layer(output2_0)
			else:
				output2 = layer(output2)

		output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))

		# classifier
		classifier = self.classifier(output2_cat)

		# upsample segmenation map ---> the input image size
		# out = F.upsample(classifier, input.size()[2:], mode='bilinear', align_corners=False)  # Upsample score map, factor=8 (deprecated method)
		out = F.interpolate(classifier, input.size()[2:], mode='bilinear', align_corners=False)  # Upsample score map, factor=8

		# post-process output
		# out = postprocess_sample(out, self.use_rois, self.num_classes)

		return out
