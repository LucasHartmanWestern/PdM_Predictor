import os
from random import randint, uniform

import cv2
import numpy as np
from tqdm import tqdm


# Constants
MAX_SIZE, MIN_SIZE = 250, 25
MAX_ALPHA, MIN_ALPHA = 0.7, 0.1
MAX_COLOR, MIN_COLOR = 255, 51


def apply_raindrop_to_img_2(img):
	size = randint(MIN_SIZE, MAX_SIZE)
	location = (randint(0, 1920), randint(0, 1080))
	alpha = round(uniform(MIN_ALPHA, MAX_ALPHA), 4)
	blur_ksize = 2 * (randint(size - MIN_SIZE, size + MIN_SIZE) // 2) + 1  # will always be an odd number
	color = randint(MIN_COLOR, MAX_COLOR)

	mask = np.zeros((1080, 1920), np.uint8)
	mask = cv2.circle(mask, location, size, (color, color, color), cv2.FILLED)
	mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
	mask_alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

	colored = np.ones(img.shape, dtype=np.uint8) * 255
	blended = np.copy(img)
	blended = cv2.convertScaleAbs(blended * (1 - mask_alpha) + colored * mask_alpha)
	result = cv2.addWeighted(blended, alpha, img, 1 - alpha, 0)
	return result


def apply_raindrop_to_img(img):
	size = randint(MIN_SIZE, MAX_SIZE)
	location = (randint(0, 1920), randint(0, 1080))
	alpha = round(uniform(MIN_ALPHA, MAX_ALPHA), 4)

	result = np.copy(img)
	cv2.circle(result, location, size, (255, 255, 255), cv2.FILLED)
	result = cv2.addWeighted(result, alpha, img, 1 - alpha, 0)
	return result


if __name__ == '__main__':
	# hyperparameters
	num_samples = 3
	min_raindrops, max_raindrops = 10, 25
	src_img_path = './image_files/input_sample.png'
	dest_dir_path = './output_samples'

	# ----- ----- ----- ----- #

	src_img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)

	for i in tqdm(range(1, num_samples + 1), desc='Generating samples'):
		num_droplets = randint(min_raindrops, max_raindrops)
		curr_img = np.copy(src_img)

		for j in range(num_droplets):
			curr_img = apply_raindrop_to_img_2(curr_img)

		cv2.imwrite(os.path.join(dest_dir_path, 'sample_{}.png'.format(i)), curr_img)

