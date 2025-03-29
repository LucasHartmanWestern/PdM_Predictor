import os
import random
import time

import cv2
import numpy as np
from skimage import filters
from tqdm import tqdm


def imshow_and_wait(img):
    cv2.imshow('img', img)
    keys = cv2.waitKey(0) & 0xFF
    if keys == ord('q'):
        cv2.destroyAllWindows()
        quit()


if __name__ == '__main__':
    
    # hyperparameters
    input_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/01_FALL_PROGRESS/1B_synthetic_rain/image_files/input_sample.png'
    input_im = cv2.imread(input_path, cv2.IMREAD_COLOR)

    # ----- ----- ----- #

    grid_im = cv2.imread(input_path, cv2.IMREAD_COLOR)

    # vertical lines
    for i in range(60, grid_im.shape[1]-59, 360):
        cv2.line(grid_im, (i, 0), (i, grid_im.shape[0]), (0, 0, 255), 3)

    # horizontal lines
    for i in range(0, grid_im.shape[0], 360):
        cv2.line(grid_im, (60, i), (grid_im.shape[1]-60, i), (0, 0, 255), 3)

    imshow_and_wait(grid_im)

    # ----- ----- ----- #

    cropped_im = cv2.imread(input_path, cv2.IMREAD_COLOR)

    start_time = time.time()
    cropped_im = cropped_im[:, 60:cropped_im.shape[1]-60, :]
    print(f'Time taken to crop: {time.time() - start_time} seconds')

    imshow_and_wait(cropped_im)

    # ----- ----- ----- #

    for i in range(0, cropped_im.shape[0], 360):
        for j in range(0, cropped_im.shape[1], 360):
            roi = cropped_im[i:i+360, j:j+360, :]
            imshow_and_wait(roi)

    # ----- ----- ----- #

    cv2.destroyAllWindows()


