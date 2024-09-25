import cv2
import numpy as np


def remove_background_dust(raw_img):
	gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]  # OTSU for raw img
	result = np.zeros(raw_img.shape, np.uint8)
	for y in range(result.shape[0]):  # loop through pixels in y-axis
		for x in range(result.shape[1]):  # loop through pixels in x-axis
			for c in range(result.shape[2]):  # loop through color channels
				if gray[y, x] < thresh:  # if this pixel belongs to the background of raw img (not metal)
					result[y, x, c] = 0  # color the pixel as default background color
				else:  # else this pixel belongs to the foreground of raw img (the metal)
					result[y, x, c] = raw_img[y, x, c]  # color pixel same as raw img
	return result


if __name__ == '__main__':
	# hyperparameters
	input_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/1B_synthetic_rain/image_files/input_sample.png'

	# ----- ------ ----- #

	raw_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
	cv2.imshow('img', raw_image)
	cv2.waitKey(0)

	edited_image = remove_background_dust(raw_image)
	cv2.imshow('img', edited_image)
	cv2.waitKey(0)

	cv2.destroyAllWindows()





