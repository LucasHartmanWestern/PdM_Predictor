import os
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from utils.constants import Partition, hour_list


def make_folders(root_dir, partition):
	os.makedirs(os.path.join(root_dir, str(partition.value), 'images'), exist_ok=True)
	os.makedirs(os.path.join(root_dir, str(partition.value), 'targets'), exist_ok=True)


def make_roi_ds(old_dir, new_dir, partition, num_scenarios):
	new_img_dir = os.path.join(new_dir, str(partition.value), 'images')
	new_tar_dir = os.path.join(new_dir, str(partition.value), 'targets')
	os.makedirs(new_img_dir, exist_ok=True)
	os.makedirs(new_tar_dir, exist_ok=True)

	roi_size = 120  # TODO: not sure if this is a good size or not
	y_indexes = 9  # 1080 / 120 = 9
	x_indexes = 16  # 1920 / 120 = 16

	for scenario_num in tqdm(range(1, num_scenarios + 1), desc='Converting each scenario...'):
		old_img_dir = os.path.join(old_dir, str(partition.value), 'scenario_{}'.format(scenario_num), 'images')
		old_tar_dir = os.path.join(old_dir, str(partition.value), 'scenario_{}'.format(scenario_num), 'targets')

		# convert images first
		for file in os.listdir(old_img_dir):
			img = cv2.imread(os.path.join(old_img_dir, file), cv2.IMREAD_COLOR)
			splits = file.split('.')[0].split('_')
			for y in range(y_indexes):
				y1, y2 = y*roi_size, y*roi_size+roi_size
				for x in range(x_indexes):
					x1, x2 = x*roi_size, x*roi_size+roi_size
					roi = img[y1:y2, x1:x2, :]
					roi_number = (y * 16) + (x + 1)
					new_file_name = 'SAMPLE_sc{}_day{}_{}_roi{}.png'.format(scenario_num, splits[2], splits[3], roi_number)
					cv2.imwrite(os.path.join(new_img_dir, new_file_name), roi.astype(np.uint8))

		# convert labels second
		for file in os.listdir(old_tar_dir):
			tar = cv2.imread(os.path.join(old_tar_dir, file), cv2.IMREAD_GRAYSCALE) # TODO: change this once labels are RGB
			splits = file.split('.')[0].split('_')
			for y in range(y_indexes):
				y1, y2 = y * roi_size, y * roi_size + roi_size
				for x in range(x_indexes):
					x1, x2 = x * roi_size, x * roi_size + roi_size
					roi = tar[y1:y2, x1:x2]
					roi_number = (y * 16) + (x + 1)
					new_file_name = 'TARGET_sc{}_day{}_roi{}.png'.format(scenario_num, splits[2], roi_number)
					cv2.imwrite(os.path.join(new_tar_dir, new_file_name), roi.astype(np.uint8))


def make_dataset_list(root_dir, partition, num_scenarios, total_days, rois_per_image):
	part_path = os.path.join(root_dir, str(partition.value))
	list_name = 'list.txt'
	list_file_path = os.path.join(part_path, list_name)
	open(list_file_path, 'w+').close()  # overwrite/ make new blank file
	with open(list_file_path, "a") as list_file:
		for sc in tqdm(range(1, num_scenarios + 1), desc='Creating roi dataset list'):
			for day in range(1, total_days + 1):
				for hour in hour_list:
					for roi_num in range(rois_per_image):  # number of roi's per image
						image_name = "SAMPLE_sc{}_day{}_{}_roi{}.png".format(sc, day, hour, roi_num)
						label_name = "TARGET_sc{}_day{}_roi{}.png".format(sc, day, roi_num)
						if os.path.exists(os.path.join(part_path, "images", image_name)):  # skip over missing hours
							img_path = os.path.join(str(partition.value), "images", image_name)
							tgt_path = os.path.join(str(partition.value), "targets", label_name)
							list_file.write(img_path + " " + tgt_path + " " + str(day) + "\n")


if __name__ == '__main__':
	# hyperparameters
	num_scs = 3
	num_data_days = 60
	num_rois_per_image = 144
	part = Partition.TEST
	dust_icon_path = './image_files/dust1.png'
	old_data_dir = '/Users/nick_1/Bell_5G_Data/SMS_ds'
	new_data_dir = '/Users/nick_1/Bell_5G_Data/ROI_ds_TEST'

	# ----- ----- ----- ----- #
	print('Creating roi {} dataset...'.format(part.value))
	start_time = datetime.now()

	make_folders(new_data_dir, part)  # make directories for dataset
	make_roi_ds(old_data_dir, new_data_dir, part, num_scs)  # convert images and labels into ROI's
	make_dataset_list(new_data_dir, part, num_scs, num_data_days, num_rois_per_image)  # make list file for dataset

	# ----- ----- ----- ----- #
	total_time = datetime.now() - start_time
	print('Script took {} to complete.'.format(total_time))