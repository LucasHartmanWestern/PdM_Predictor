import os
from enum import Enum

# ----- Constants ----- #
max_vignette_strength = 25.0  # vignette strength will start at this value
min_vignette_strength = 5.00  # smallest possible vignette strength

data_path_mac = '/Users/nick_1/Bell_5G_Data'
data_path_win32 = 'C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data'
data_path_linux = '/mnt/storage_1/bell_5g_datasets'

project_dir = '/Users/nick_1/PycharmProjects/Western Summer Research/CoolingTowerSyntheticDataGenerator'
dust_icon_path = os.path.join(project_dir, 'synthetic_data_generator/image_files/dust1.png')
grate_mask_path = os.path.join(project_dir, 'synthetic_data_generator/image_files/metal_mask_v2.png')

hour_list = [
	'8am', '9am', '10am', '11am', '12pm',
	'1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm'
]
csv_headers_3spots = [
	'day', 'cleaning_occurred',

	'growth_percent_1', 'dust_rows_1', 'dust_cols_1',
	'dust_x_1', 'dust_y_1', 'region_1',

	'growth_percent_2', 'dust_rows_2', 'dust_cols_2',
	'dust_x_2', 'dust_y_2', 'region_2',

	'growth_percent_3', 'dust_rows_3', 'dust_cols_3',
	'dust_x_3', 'dust_y_3', 'region_3'
]


class Partition(Enum):
	TRAIN = 'train'
	TEST = 'test'
	VAL = 'validation'
