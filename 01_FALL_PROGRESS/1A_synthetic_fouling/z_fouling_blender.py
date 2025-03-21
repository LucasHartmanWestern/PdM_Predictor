import time
import random

import cv2
import numpy as np
from skimage import filters

# constant values
MAX_INTENSITY_FACTOR = 1.4
MIN_INTENSITY_FACTOR = 0.8

MAX_BLUR_STRENGTH = 0.8
MIN_BLUR_STRENGTH = 0.0


def make_fouling_spot(dust_img, rows, cols, rand_intensity=True, rand_blur=True):
	# make the tiled image using the dust sample
	h, w, c = dust_img.shape
	dust_cloud = np.zeros((h * rows, w * cols, c), dust_img.dtype)
	for row in range(rows):
		for col in range(cols):
			dust_cloud[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = dust_img[:, :, :]

	# shift intensity if enabled
	if rand_intensity:
		intensity_factor = random.uniform(MIN_INTENSITY_FACTOR, MAX_INTENSITY_FACTOR)
		# print('Intensity scale factor: {}'.format(intensity_factor))
		dust_cloud = np.uint8(intensity_factor * dust_cloud.astype(np.float32))

	# apply blurring if enabled
	if rand_blur:
		blur_factor = random.uniform(MIN_BLUR_STRENGTH, MAX_BLUR_STRENGTH)
		# print('Blur strength factor: {}'.format(blur_factor))
		dust_cloud = np.uint8(filters.gaussian(dust_cloud, sigma=blur_factor) * 255)

	# compute and apply black vignette
	h, w = dust_cloud.shape[:2]
	x_resultant_kernel = cv2.getGaussianKernel(w, w / 4.0)
	y_resultant_kernel = cv2.getGaussianKernel(h, h / 4.0)
	resultant_kernel = y_resultant_kernel * x_resultant_kernel.T
	mask = resultant_kernel / resultant_kernel.max()
	for ch in range(3):
		dust_cloud[:, :, ch] = dust_cloud[:, :, ch] * mask

	return dust_cloud


def apply_fouling_spot(base_img, fouling_img):
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


if __name__ == '__main__':
	# hyperparameters
	input_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/1B_synthetic_rain/image_files/input_sample.png'
	dust_path = './image_files/dust1.png'
	location = (600, 240)

	# ----- ----- ----- #

	input_im = cv2.imread(input_path, cv2.IMREAD_COLOR)
	fouling_im = cv2.imread(dust_path, cv2.IMREAD_COLOR)
	dc = make_fouling_spot(fouling_im, rows=8, cols=15, rand_intensity=True, rand_blur=True)
	cv2.imshow('img', dc)
	cv2.waitKey(0)

	dc_h, dc_w = dc.shape[:2]
	pt1_x, pt1_y = location
	pt2_x, pt2_y = pt1_x + dc_w, pt1_y + dc_h
	roi = input_im[pt1_y:pt2_y, pt1_x:pt2_x]

	start_time = time.time()
	roi = apply_fouling_spot(roi, dc)
	input_im[pt1_y:pt2_y, pt1_x:pt2_x] = roi
	print('Dust application took {:.5f} seconds'.format(time.time() - start_time))
	input_im = cv2.rectangle(input_im, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 0, 255), 3)
	cv2.imshow('img', input_im)
	cv2.waitKey(0)

	cv2.destroyAllWindows()


