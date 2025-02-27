import cv2
import numpy as np


if __name__ == '__main__':
	# hyperparameters
	input_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/1B_synthetic_rain/image_files/input_sample.png'
	mask_path = './image_files/metal_seg_mask.png'

	# ----- ------ ----- #

	metal_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
	raw_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
	cv2.imshow('img', raw_image)
	cv2.waitKey(0)

	edited_image = raw_image.copy()
	edited_image[metal_mask == 0] = (0, 0, 0)
	cv2.imshow('img', edited_image)
	cv2.waitKey(0)

	cv2.destroyAllWindows()





