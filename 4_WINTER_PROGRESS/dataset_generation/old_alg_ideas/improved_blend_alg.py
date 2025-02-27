import os
import random
import time

import cv2
import numpy as np
from skimage import filters
from tqdm import tqdm


def create_folder_structure(dataset_folder_path):
    samples_path = os.path.join(dataset_folder_path, 'samples')
    targets_path = os.path.join(dataset_folder_path, 'targets')
    fouling_path = os.path.join(dataset_folder_path, 'fouling')
    os.makedirs(samples_path, exist_ok=True)
    os.makedirs(targets_path, exist_ok=True)
    os.makedirs(fouling_path, exist_ok=True)


def apply_fouling_spot(base_img, fouling_img):  # takes ~10 seconds
	gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]
	alphas = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1].astype(np.float32) * (1 / 255)
	for y in range(base_img.shape[0]):  # loop through pixels in y-axis
		for x in range(base_img.shape[1]):  # loop through pixels in x-axis
			for c in range(base_img.shape[2]):  # loop through color channels
				if gray[y, x] < thresh:  # if this pixel belongs to the background of raw img (not metal)
					# color pixel according to alpha values
					base_img[y, x, c] = ((1 - alphas[y, x]) * base_img[y, x, c]) + (alphas[y, x] * fouling_img[y, x, c])
				# else this pixel belongs to the foreground of raw img (the metal); color pixel same as base img
	return base_img


def apply_fouling_spot_FAST(base_img, fouling_img):  # takes ~5 seconds
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]
    alphas = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1].astype(np.float32) * (1 / 255)
    for y in range(base_img.shape[0]):  # loop through pixels in y-axis
        for x in range(base_img.shape[1]):  # loop through pixels in x-axis
            if gray[y, x] < thresh:  # if this pixel belongs to the background of raw img (not metal)
                base_img[y, x, :] = ((1 - alphas[y, x]) * base_img[y, x, :]) + (alphas[y, x] * fouling_img[y, x, :])
    return base_img


def apply_fouling_spot_FAST_V2(base_img, fouling_img):  # takes ~0.1 seconds
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]
    alphas = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1].astype(np.float32) * (1 / 255)
    for c in range(base_img.shape[2]):
        base_img[gray < thresh, c] = ((1 - alphas[gray < thresh]) * base_img[gray < thresh, c]) + (alphas[gray < thresh] * fouling_img[gray < thresh, c])
    return base_img


if __name__ == '__main__':
    # hyperparameters
    input_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/1B_synthetic_rain/image_files/input_sample.png'
    ds_path = '/Users/nick_1/PycharmProjects/UWO Masters/data/sds_test_1'

    # ----- ----- ----- #

    input_im = cv2.imread(input_path, cv2.IMREAD_COLOR)
    full_fouling_im = cv2.imread(os.path.join(ds_path, 'fouling', 'fouling_20.png'), cv2.IMREAD_COLOR)
    cv2.imshow('img', full_fouling_im)
    cv2.waitKey(0)
	
    start_time = time.time()
    result = apply_fouling_spot_FAST_V2(input_im, full_fouling_im)
    print('Dust application took {:.5f} seconds'.format(time.time() - start_time))
    cv2.imshow('img', result)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
