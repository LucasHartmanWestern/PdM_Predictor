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
    input_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/1B_synthetic_rain/image_files/input_sample.png'
    input_im = cv2.imread(input_path, cv2.IMREAD_COLOR)

    # ----- ----- ----- #
    print("================================================")
    # ----- ----- ----- #

    side_reduction = int((1920 - 1080) / 2)
    print(f'Side reduction: {side_reduction} pixels')
    print(f'Total pixels removed: {side_reduction * 1080 * 2} pixels')

    normal_im = cv2.imread(input_path, cv2.IMREAD_COLOR)
    cv2.line(normal_im, (side_reduction, 0), (side_reduction, normal_im.shape[0]), (0, 0, 255), 3)
    cv2.line(normal_im, (normal_im.shape[1]-side_reduction, 0), (normal_im.shape[1]-side_reduction, normal_im.shape[0]), (0, 0, 255), 3)
    imshow_and_wait(normal_im)

    cropped_im = cv2.imread(input_path, cv2.IMREAD_COLOR)
    cropped_im = cropped_im[:, side_reduction:cropped_im.shape[1]-side_reduction, :]
    imshow_and_wait(cropped_im)

    # ----- ----- ----- #
    print("================================================")
    # ----- ----- ----- #

    crop_width = int(1920 / 2)
    reduction = int((1080 - crop_width) / 2)
    print(f'Top/ Bottom reduction: {reduction} pixels')
    print(f'Total pixels removed: {reduction * 1920 * 2} pixels')

    normal_im = cv2.imread(input_path, cv2.IMREAD_COLOR)
    cv2.line(normal_im, (crop_width, reduction), (crop_width, normal_im.shape[0]-reduction), (0, 0, 255), 3)
    cv2.line(normal_im, (0, reduction), (normal_im.shape[1], reduction), (0, 0, 255), 3)
    cv2.line(normal_im, (0, normal_im.shape[0]-reduction), (normal_im.shape[1], normal_im.shape[0]-reduction), (0, 0, 255), 3)
    imshow_and_wait(normal_im)

    cropped_im = cv2.imread(input_path, cv2.IMREAD_COLOR)
    for i in range(2):
        example_im = cropped_im[reduction:cropped_im.shape[1]-reduction, i*crop_width:(i+1)*crop_width, :]
        print(example_im.shape)
        imshow_and_wait(example_im)

    # ----- ----- ----- #
    print("================================================")
    # ----- ----- ----- #

    cv2.destroyAllWindows()


