import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
	# hyperparameters
	# fig_size = (4.0, 2.5)  # size of other plots in paper
	fig_size = (4.0, 4.0)
	model_names = ['old_unet', 'nested_unet_512', 'cgnet']
	plot_colors = ['tab:blue', 'tab:orange', 'tab:green']
	save_path = './plots'

	plt.clf()
	# plt.rcParams.update({'font.size': 10})  # default font size
	plt.rcParams.update({'font.size': 8})
	fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=fig_size)

	y_ticks_right = np.linspace(0.0, 1.5, 3)
	y_ticks_left = np.linspace(0.5, 1.0, 3)
	x_range = range(1, 61)
	x_ticks = np.linspace(0, 60, 4)
	x_ticks[0] += 1

	for m in range(len(model_names)):

		data_path = '../2A_detection/experiments/{}/testing'.format(model_names[m])
		df1 = pd.read_csv(os.path.join(data_path, 'sc1_data.csv'))
		df2 = pd.read_csv(os.path.join(data_path, 'sc2_data.csv'))
		df3 = pd.read_csv(os.path.join(data_path, 'sc3_data.csv'))

		averaged_scores, averaged_foul_percentages = [], []
		for i in range(len(df1)):
			averaged_scores.append(
				np.mean([df1['avg_f1_score'][i], df2['avg_f1_score'][i], df3['avg_f1_score'][i]])
			)
			averaged_foul_percentages.append(
				np.mean([df1['percent_img_fouling'][i], df2['percent_img_fouling'][i], df3['percent_img_fouling'][i]])
			)

		ax[m].plot(x_range, averaged_scores, color=plot_colors[m])
		ax[m].grid(True)
		ax[m].set_xticks(x_ticks)
		ax[m].set_yticks(y_ticks_left)
		ax_twin = ax[m].twinx()
		ax_twin.plot(x_range, averaged_foul_percentages, color=plot_colors[m], linestyle='--')
		ax_twin.set_yticks(y_ticks_right)
		if m == 1:
			ax[m].set_ylabel('F1 scores', fontsize=10)
			ax_twin.set_ylabel('Fouling percentages', fontsize=10)
		if m == 2:
			ax[m].set_xlabel('Day', fontsize=10)

	plt.tight_layout()
	plt.savefig(os.path.join(save_path, 'TEST_plot.png'))
