import csv
import os
import shutil
from datetime import datetime
from random import uniform

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.constants import Partition, hour_list, csv_headers_3spots
from utils.helper import get_scenario_paths, get_growth_bounds, gen_dust_variables, \
	interpret_sc3_sc4_csv, make_dust_cloud, get_img_slice_pts, denoise_to_binary, apply_synth_transparent_dust


def make_folders(root_dir, partition, num_scenarios):
	data_part_path = os.path.join(root_dir, str(partition.value))
	for scenario_num in range(1, num_scenarios + 1):
		sc_path = os.path.join(data_part_path, 'scenario_{}'.format(scenario_num))
		os.makedirs(os.path.join(sc_path, 'images'), exist_ok=True)
		os.makedirs(os.path.join(sc_path, 'targets'), exist_ok=True)


def make_dataset_timelines(root_dir, partition, num_days, num_scenarios):
	data_part_path = os.path.join(root_dir, str(partition.value))
	for scenario_num in tqdm(range(1, num_scenarios + 1), desc='Generating timelines for scenarios'):
		min_daily_growth, max_daily_growth = get_growth_bounds(num_days, 3.0)
		cleaning_flag = 0  # only used for visual inspection of csv file
		curr_growths = [0.0, 0.0, 0.0]
		dust_vars_list = [gen_dust_variables(), gen_dust_variables(), gen_dust_variables()]
		timeline = []
		for day in range(1, num_days + 1):
			if day > 1:
				for i in range(len(curr_growths)):
					curr_growths[i] += uniform(min_daily_growth, max_daily_growth)
					curr_growths[i] = round(curr_growths[i], 2)
					if curr_growths[i] > 100.0:  # enforce maximum total bound
						curr_growths[i] = 100.0
			timeline.append([
				day, cleaning_flag,
				curr_growths[0], dust_vars_list[0]['rows'], dust_vars_list[0]['cols'],
				dust_vars_list[0]['loc_x'], dust_vars_list[0]['loc_y'], dust_vars_list[0]['region'],
				curr_growths[1], dust_vars_list[1]['rows'], dust_vars_list[1]['cols'],
				dust_vars_list[1]['loc_x'], dust_vars_list[1]['loc_y'], dust_vars_list[1]['region'],
				curr_growths[2], dust_vars_list[2]['rows'], dust_vars_list[2]['cols'],
				dust_vars_list[2]['loc_x'], dust_vars_list[2]['loc_y'], dust_vars_list[2]['region']
			])
		save_path = os.path.join(data_part_path, "scenario_{}".format(scenario_num), 'timeline.csv')
		open(save_path, 'w+').close()  # overwrite/ make new blank file
		with open(save_path, 'a', encoding='UTF8', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(csv_headers_3spots)
			writer.writerows(timeline)


def make_dataset_images(root_dir, partition, dust_im_path, num_scenarios):
	for scenario_num in range(1, num_scenarios + 1):
		paths = get_scenario_paths(root_dir, partition, scenario_num)
		dust_img = cv2.imread(dust_im_path, cv2.IMREAD_COLOR)
		csv_dicts = interpret_sc3_sc4_csv(paths['csv'])

		for day in tqdm(range(1, csv_dicts[0]['total_days'] + 1), desc='Generating scenario {} images'.format(scenario_num)):
			# Part A: create list of dust cloud data for current day
			dust_clouds = []
			for deg_spot_data in csv_dicts:
				dust_cloud = make_dust_cloud(dust_img, deg_spot_data, day - 1)
				pt1, pt2 = get_img_slice_pts(dust_cloud, deg_spot_data, day - 1)
				dust_clouds.append((dust_cloud, pt1, pt2))

			# Part B: create synth label for current day
			label_img = np.zeros((1080, 1920, 3), dtype=np.uint8)  # blank image
			for dust_cloud, pt1, pt2 in dust_clouds:  # apply each dust cloud
				label_img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = dust_cloud  # insert dust cloud to label
			label_img = denoise_to_binary(label_img)  # denoise & convert to b/w
			label_save_path = os.path.join(paths['label'], 'LABEL_day_{}.png'.format(day))
			cv2.imwrite(label_save_path, label_img.astype(np.uint8))  # save synthetic label

			# Part C: generate synth image for each valid hour of current day
			for hour in hour_list:
				raw_img_name = 'day_{}_{}.png'.format(day, hour)
				raw_img_path = os.path.join(paths['src'], raw_img_name)
				synth_img_save_path = os.path.join(paths['img'], 'SYNTH_{}'.format(raw_img_name))
				if os.path.isfile(raw_img_path):  # skip over missing data
					image = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)  # read raw img file
					for dust_cloud, pt1, pt2 in dust_clouds:  # apply each dust cloud
						image_slice = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]  # cut out RoI for modification
						image_slice = apply_synth_transparent_dust(image_slice, dust_cloud)  # apply dust cloud to RoI
						image[pt1[1]:pt2[1], pt1[0]:pt2[0]] = image_slice  # re-insert edited RoI
					cv2.imwrite(synth_img_save_path, image.astype(np.uint8))  # save synthetic img


def make_dataset_list(root_dir, partition, num_scenarios):
	part_path = os.path.join(root_dir, str(partition.value))
	list_name = 'list.txt'
	list_file_path = os.path.join(part_path, list_name)
	open(list_file_path, 'w+').close()  # overwrite/ make new blank file
	list_file = open(list_file_path, "a")
	for scenario in tqdm(range(1, num_scenarios + 1), desc='Creating dataset list'):
		scenario_dir = os.path.join(part_path, "scenario_{}".format(scenario))
		total_days = pd.read_csv(os.path.join(scenario_dir, "timeline.csv"))['day'].values[-1]
		for day in range(1, total_days + 1):
			label_name = "LABEL_day_{}.png".format(day)
			for hour in hour_list:
				image_name = "SYNTH_day_{}_{}.png".format(day, hour)
				if os.path.exists(os.path.join(scenario_dir, "images", image_name)):  # skip over missing hours
					img_path = os.path.join(str(partition.value), "scenario_{}".format(scenario), "images", image_name)
					tgt_path = os.path.join(str(partition.value), "scenario_{}".format(scenario), "targets", label_name)
					list_file.write(img_path + " " + tgt_path + "\n")
	list_file.close()


def make_resized_ds(old_dir, new_dir, partition, num_scenarios):
	for scenario_num in tqdm(range(1, num_scenarios + 1), desc='Creating resized dataset'):
		old_img_dir = os.path.join(old_dir, str(partition.value), 'scenario_{}'.format(scenario_num), 'images')
		old_tar_dir = os.path.join(old_dir, str(partition.value), 'scenario_{}'.format(scenario_num), 'targets')

		new_img_dir = os.path.join(new_dir, str(partition.value), 'scenario_{}'.format(scenario_num), 'images')
		new_tar_dir = os.path.join(new_dir, str(partition.value), 'scenario_{}'.format(scenario_num), 'targets')

		os.makedirs(new_img_dir, exist_ok=True)
		os.makedirs(new_tar_dir, exist_ok=True)

		for file in os.listdir(old_img_dir):
			img = cv2.imread(os.path.join(old_img_dir, file), cv2.IMREAD_COLOR)
			img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
			cv2.imwrite(os.path.join(new_img_dir, file), img.astype(np.uint8))

		for file in os.listdir(old_tar_dir):
			tar = cv2.imread(os.path.join(old_tar_dir, file), cv2.IMREAD_GRAYSCALE)
			tar = cv2.resize(tar, (512, 512), interpolation=cv2.INTER_AREA)
			cv2.imwrite(os.path.join(new_tar_dir, file), tar.astype(np.uint8))

	old_list_path = os.path.join(old_dir, str(partition.value), 'list.txt')
	new_list_path = os.path.join(new_dir, str(partition.value), 'list.txt')
	shutil.copyfile(old_list_path, new_list_path)


if __name__ == '__main__':
	# hyperparameters
	num_scs = 2
	num_data_days = 15
	part = Partition.TEST
	dust_icon_path = './image_files/dust1.png'
	data_dir = '/Users/nick_1/Bell_5G_Data/CGS_ds'
	sm_data_dir = '/Users/nick_1/Bell_5G_Data/sm_CGS_ds'

	# ----- ----- ----- ----- #
	print('Creating synthetic {} dataset...'.format(part.value))
	start_time = datetime.now()

	make_folders(data_dir, part, num_scs)  # make directories for dataset
	make_dataset_timelines(data_dir, part, num_data_days, num_scs)  # make timeline files for each scenario
	make_dataset_images(data_dir, part, dust_icon_path, num_scs)  # make images and targets for each scenario
	make_dataset_list(data_dir, part, num_scs)  # make list file for dataset
	make_resized_ds(data_dir, sm_data_dir, part, num_scs)  # make resized dataset

	# ----- ----- ----- ----- #
	total_time = datetime.now() - start_time
	print('Script took {} to complete.'.format(total_time))


