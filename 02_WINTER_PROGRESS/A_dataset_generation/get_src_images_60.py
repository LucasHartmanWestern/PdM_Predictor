import os
import shutil
from datetime import date, timedelta

import pandas as pd
from tqdm import tqdm


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
    start_date = date(2024, 6, 13)  # start date inclusive
    end_date = date(2024, 8, 29)  # end date exclusive
    src_dir = '/Users/nick_1/Bell_5G_Data/all_1080_data/'
    dest_dir = '/Users/nick_1/PycharmProjects/UWO Masters/data_60/src_images'

    hour_list = [
        '8am', '9am', '10am', '11am', '12pm',
        '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm'
    ]

    excluded_dates = [
        "2024_06_20", "2024_06_21", "2024_06_22", "2024_06_23", "2024_06_24", "2024_06_25", "2024_06_28",
        "2024_07_07", "2024_07_10", "2024_07_20", "2024_07_21", "2024_07_25",
        "2024_08_16", "2024_08_17", "2024_08_18", "2024_08_19", "2024_08_26",
    ] # exclude enough dates to get 60 days of data (preferably exclude dates with the most missing times)
    # fun fact: 2024_07_25_7am has a bird in the image (therefore excluded)

    # ----- ----- ----- ----- ----- #

    # time and date lists
    date_range_list = pd.date_range(start_date, end_date - timedelta(days=1), freq='d').to_list()

    # --- obtain path for image and copy/ rename it --- #
    day_num = 1
    dates_found = []
    dates_found_missing_times = []

    for timestamp in tqdm(date_range_list, desc='Copying images'):

        date_str = str(timestamp).split(' ')[0].replace('-', '_').strip()
        if date_str in excluded_dates:
            continue

        files_found_for_date = 0
        file_copy_buffer = []
        times_not_found = []

        for hour in hour_list:
            dest_path = os.path.join(dest_dir, 'day_{}_{}.png'.format(day_num, hour))

            for snapshot_num in range(1, 4):
                file_name = '{}_{}_snapshot_{}.png'.format(date_str, hour, snapshot_num)
                file_path = os.path.join(src_dir, file_name)
                if exists_in_dir(file_name, src_dir):
                    file_copy_buffer.append((file_path, dest_path))
                    files_found_for_date += 1
                    break
                elif snapshot_num == 3:
                    times_not_found.append(file_name.split('_')[3])

        if len(times_not_found) > 0:
            dates_found.append(date_str)
            dates_found_missing_times.append(times_not_found)

        copy_paste_buffer(file_copy_buffer)
        day_num += 1
    
    # ------------------------------------------------- #

    print(f"\nCopied {day_num-1} days of images successfully.")
    print('\nSome files were missing for these dates in the dataset:\n')
    for date_str, times in zip(dates_found, dates_found_missing_times):
        print(f"{date_str}:\t{', '.join(times)}")
    
    
