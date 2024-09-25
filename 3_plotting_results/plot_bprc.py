import os

import torch
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryPrecisionRecallCurve

if __name__ == '__main__':
	# hyperparameters
	fig_size = (4.0, 2.5)  # size of other plots in paper
	model_names = ['old_unet', 'nested_unet_512', 'cgnet']
	base_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor'

	# -- create plot -- #
	plt.clf()
	plt.figure(figsize=fig_size)
	for m in range(len(model_names)):
		data_path = os.path.join(base_path, '2A_detection/experiments/{}/testing/bprc.pth'.format(model_names[m]))
		bprc = BinaryPrecisionRecallCurve(thresholds=1000)
		bprc.load_state_dict(torch.load(data_path, map_location='cpu'))
		bprc.plot(score=True, ax=plt.gca())
		bprc.reset()
	plt.title('')
	plt.tight_layout()
	plt.savefig(os.path.join(base_path, '3_plotting_results/plots/combined_bprc.png'))
