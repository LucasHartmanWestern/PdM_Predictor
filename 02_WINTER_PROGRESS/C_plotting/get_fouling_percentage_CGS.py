import os
import numpy as np
import cv2
import pandas as pd

if __name__ == "__main__":
    # Hyperparameters
    binary = True
    path_to_ds = '/Users/nick_1/PycharmProjects/UWO Masters/data_60/mar28_ds5' # CGS
    save_folder = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/03_ICML_Results'
    save_file_name = 'CGS_binary_fouling_percentage.csv' if binary else 'CGS_multiclass_fouling_percentage.csv'
    
    # loop through all images in ds/test/targets
    if binary:
        targets_path = os.path.join(path_to_ds, 'test', 'targets', 'binary')
    else:
        targets_path = os.path.join(path_to_ds, 'test', 'targets', 'multiclass')

    # create new csv file
    df = pd.DataFrame(columns=['day', 'background_percentage', 'fouling_percentage'])

    for day in range(1, 61):
        img = cv2.imread(os.path.join(targets_path, f'target_{day}.png'))
        # get percentage of pixels that are 0
        background_percentage = np.sum(img == 0) / img.size * 100
        # get percentage of pixels that are not 0
        fouling_percentage = np.sum(img != 0) / img.size * 100
        row_dict = {'day': day, 'background_percentage': np.round(background_percentage, 4), 'fouling_percentage': np.round(fouling_percentage, 4)}
        df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)

    df.to_csv(os.path.join(save_folder, save_file_name), index=False)