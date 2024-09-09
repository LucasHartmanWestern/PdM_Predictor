import os
import shutil
from datetime import date, timedelta

import pandas as pd
from tqdm import tqdm

from synthetic_data_generator.utils.constants import hour_list


def exists_in_dir(f_str, dir_path):
	exists = False
	for f_name in os.listdir(dir_path):
		if f_str in f_name:
			exists = True
	return exists


def copy_paste_buffer(buf):
	for src, dst in buf:
		shutil.copyfile(src, dst)


if __name__ == '__main__':
	# hyperparameters
	start_date = date(2024, 6, 13)  # range: 06-13 to 08-28 (inclusively)
	end_date = date(2024, 8, 28)  # end date is not inclusive
	src_dir = '/Users/nick_1/Bell_5G_Data/all_1080_data/'
	dest_dir = '/Users/nick_1/Bell_5G_Data/4sc_ds/src_images'

	excluded_dates = [  # excluded due to too many missing hours
		'2024_06_20', '2024_06_21', '2024_06_22', '2024_06_23', '2024_06_24', '2024_06_25', '2024_06_28',
		'2024_07_07', '2024_07_10', '2024_07_20', '2024_07_21',
		'2024_08_16', '2024_08_17', '2024_08_18', '2024_08_19', '2024_08_26',
	]

	# ----- ----- ----- ----- ----- #

	# time and date lists
	date_range_list = pd.date_range(start_date, end_date - timedelta(days=1), freq='d').to_list()

	# --- obtain path for image and copy/ rename it --- #
	day_num = 1
	dates_not_found = []
	files_not_found = []

	for timestamp in tqdm(date_range_list, desc='Copying images'):

		date_str = str(timestamp).split(' ')[0].replace('-', '_').strip()
		if date_str in excluded_dates:
			continue

		files_found_for_date = 0
		file_copy_buffer = []

		for hour in hour_list:
			file_name = '{}_{}_snapshot_1.png'.format(date_str, hour)
			file_path = os.path.join(src_dir, file_name)
			dest_path = os.path.join(dest_dir, 'day_{}_{}.png'.format(day_num, hour))

			if exists_in_dir(file_name, src_dir):
				file_copy_buffer.append((file_path, dest_path))
				files_found_for_date += 1
			else:
				files_not_found.append(file_name)

		if files_found_for_date > 0:
			copy_paste_buffer(file_copy_buffer)
			day_num += 1
		else:
			dates_not_found.append(date_str)
	# ------------------------------------------------- #

	print('Could not find these specific files:')
	for file in files_not_found:
		print('\t{}'.format(file))

	print('0 files were found for these dates:')
	for date_str in dates_not_found:
		print('\t{}'.format(date_str))

