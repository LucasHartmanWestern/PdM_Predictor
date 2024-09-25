import torch
import torch.nn as nn
import numpy as np

# obtained from:
# https://github.com/daduquea/powerLosses/tree/main


class PowerJaccard(nn.Module):
	def __init__(self, p_value=2, smooth=10):
		super(PowerJaccard, self).__init__()
		self.p_value = p_value
		self.smooth = smooth

	def forward(self, inputs, targets):
		y_true_f = torch.flatten(targets)
		y_pred_f = torch.flatten(inputs)
		intersection = np.sum(y_true_f * y_pred_f)
		term_true = torch.sum(torch.pow(y_true_f, self.p_value))
		term_pred = torch.sum(torch.pow(y_pred_f, self.p_value))
		union = term_true + term_pred - intersection
		return 1 - ((intersection + self.smooth) / (union + self.smooth))


class PowerDice(nn.Module):
	def __init__(self, p_value=2, smooth=10):
		super(PowerDice, self).__init__()
		self.p_value = p_value
		self.smooth = smooth

	def forward(self, inputs, targets):
		y_true = torch.flatten(targets)
		y_pred = torch.flatten(inputs)
		numerator = torch.sum(2 * (y_true * y_pred))
		y_true = torch.pow(y_true, self.p_value)
		y_pred = torch.pow(y_pred, self.p_value)
		denominator = torch.sum(y_true) + torch.sum(y_pred)
		return 1 - ((numerator + self.smooth) / (denominator + self.smooth))
