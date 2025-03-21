import os

import torch
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryPrecisionRecallCurve

f_name1 = 'CYPRESS_exp1.png'
f_name2 = 'CYPRESS_exp2.png'

base_path1 = '/Users/nick_1/PycharmProjects/Western Summer Research/UWO_Maitenance_Predictor/past_experiments'
base_path2 = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/2A_detection/experiments'

experiment1 = 'optimizer experiment'
experiment2 = '100 epoch training'

models1 = ['model_1 (SGD)', 'model_2 (Adam)', 'model_3 (AdamW)']
models2 = ['old_unet', 'nested_unet_512', 'cgnet']

bprc_path1 = 'test results/bprc.pth'
bprc_path2 = 'testing/bprc.pth'

legends1 = ['SGD', 'Adam', 'AdamW']
legends2 = ['UNet', 'UNet++', 'CGNet']


if __name__ == '__main__':
	# hyperparameters
	fig_size = (4.0, 2.5)  # size of other plots in paper

	save_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor'
	f_name = f_name2
	base_path = base_path2
	experiment = experiment2
	models = models2
	bprc_path = bprc_path2
	legends = legends2

	# -- create plot -- #
	plt.clf()
	plt.figure(figsize=fig_size)
	for m in range(len(models)):
		data_path = os.path.join(base_path, experiment, models[m], bprc_path)
		bprc = BinaryPrecisionRecallCurve(thresholds=1000)
		bprc.load_state_dict(torch.load(data_path, map_location='cpu', weights_only=True))
		bprc.plot(score=True, ax=plt.gca())
		bprc.reset()

	leg = plt.gca().legend()
	for n in range(len(legends)):
		num = leg.get_texts()[n].get_text().split('=')[1]
		leg.get_texts()[n].set_text('{}={}'.format(legends[n], num))

	plt.title('')
	plt.tight_layout()
	plt.savefig(os.path.join(save_path, '3_plotting_results/plots/{}'.format(f_name)))
