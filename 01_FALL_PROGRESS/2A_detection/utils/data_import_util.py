import os
import sys

from sklearn.model_selection import train_test_split


def data_to_xy(data, seperator=" "):
    x_set, y_set = [], []
    for line in data:
        x, y = line.split(seperator)
        x_set.append(x.strip())
        y_set.append(y.strip())
    return x_set, y_set


def get_data_from_list(list_path, split=None, seed=None):
    all_data = []
    for line in open(list_path, "r"):
        all_data.append(line)

    x1, y1, x2, y2 = None, None, None, None

    if split is not None and seed is not None:
        assert (0.01 <= split <= 0.99)
        data_p1, data_p2 = train_test_split(all_data, test_size=split, random_state=seed, shuffle=True)
        x1, y1 = data_to_xy(data_p1, seperator=" ")
        x2, y2 = data_to_xy(data_p2, seperator=" ")
    else:
        x1, y1 = data_to_xy(all_data, seperator=" ")

    return x1, y1, x2, y2


def get_xy_data(ds_parent_folder, partition, split=None, seed=None):
    assert partition in ['train', 'test', 'validation'], "ERROR: invalid partition '{}'".format(partition)

    if sys.platform == 'darwin':  # mac
        base_path = '/Users/nick_1/Bell_5G_Data'
    elif sys.platform == 'win32':  # windows
        base_path = 'C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data'
    else:  # ubuntu
        base_path = '/mnt/storage_1/bell_5g_datasets'

    list_path = os.path.join(base_path, ds_parent_folder, partition, 'list.txt')
    assert os.path.isfile(list_path), "ERROR: no dataset list exists at '{}'".format(list_path)

    x1, y1, x2, y2 = get_data_from_list(list_path, split=split, seed=seed)

    return x1, y1, x2, y2

