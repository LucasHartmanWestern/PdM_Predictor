import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
	# hyperparameters
	# fig_size = (4.0, 2.5)  # size of other plots in paper
	fig_size = (4.0, 4.0)
	model_names = ['unet', 'nested_unet', 'cgnet']
	save_path = '../plots'

	# FOR LOOP HERE
	data_path = '../2A_detection/experiments/{}/test results'.format(model_names[0])

	# ----- ----- read csv files ----- ----- #
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

	# ----- ----- make plot ----- ----- #
	plt.clf()
	# plt.rcParams.update({'font.size': 10})  # default font size
	plt.rcParams.update({'font.size': 8})
	fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=fig_size)

	y_ticks_right = np.linspace(0.0, 1.5, 3)
	y_ticks_left = np.linspace(0.5, 1.0, 3)
	x_range = range(1, 61)
	x_ticks = np.linspace(0, 60, 4)
	x_ticks[0] += 1

	ax[0].plot(x_range, sc1_scores, color='tab:blue', label='scenario 1')
	ax[0].grid(True)
	ax[0].set_xticks(x_ticks)
	ax[0].set_yticks(y_ticks_left)
	ax0_2 = ax[0].twinx()
	ax0_2.plot(x_range, sc1_foul_percentages, color='blue', label='scenario 1', linestyle='--')
	ax0_2.set_yticks(y_ticks_right)

	ax[1].plot(x_range, sc2_scores, color='tab:orange', label='scenario 2')
	ax[1].grid(True)
	ax[1].set_xticks(x_ticks)
	ax[1].set_yticks(y_ticks_left)
	ax[1].set_ylabel('F1 scores for each scenario', fontsize=10)
	ax1_2 = ax[1].twinx()
	ax1_2.plot(x_range, sc2_foul_percentages, color='orange', label='scenario 2', linestyle='--')
	ax1_2.set_yticks(y_ticks_right)
	ax1_2.set_ylabel('Fouling percentages for each scenario', fontsize=10)

	ax[2].plot(x_range, sc3_scores, color='tab:green', label='scenario 3')
	ax[2].grid(True)
	ax[2].set_xticks(x_ticks)
	ax[2].set_yticks(y_ticks_left)
	ax[2].set_xlabel('Day', fontsize=10)
	ax2_2 = ax[2].twinx()
	ax2_2.plot(x_range, sc3_foul_percentages, color='green', label='scenario 3', linestyle='--')
	ax2_2.set_yticks(y_ticks_right)

	plt.tight_layout()
	plt.savefig(os.path.join(save_path, 'TEST_plot.png'))