import os
import random
import time

import cv2
import numpy as np
from skimage import filters
from tqdm import tqdm


# global constants
DIRECTIONS = ['N', 'S', 'W', 'E']

IMG_DIMS_RGB = (1080, 1920, 3)
IMG_DIMS = (1080, 1920)
TILE_DIMS = (36, 20)
MATRIX_DIMS = (IMG_DIMS[0] // TILE_DIMS[0], IMG_DIMS[1] // TILE_DIMS[1])

BORDER_DEPTH = 3

LABEL_DEPTH_EASY = 3
LABEL_DEPTH_MEDIUM = 2
LABEL_DEPTH_HARD = 1
LABEL_DEPTH_UNCHANGED = 0


def apply_boundry_effect_basic(img, tile_matrix, tile_h, tile_w, num_rows, num_cols):
    img = img.astype(np.float64)
    for row in range(num_rows):
        for col in range(num_cols):
            tile_N = tile_matrix[row-1, col] if row-1 >= 0 else False
            tile_W = tile_matrix[row, col-1] if col-1 >= 0 else False
            tile_S = tile_matrix[row+1, col] if row+1 < num_rows else False
            tile_E = tile_matrix[row, col+1] if col+1 < num_cols else False

            tile_NW = tile_matrix[row-1, col-1] if row-1 >= 0 and col-1 >= 0 else False
            tile_SW = tile_matrix[row+1, col-1] if row+1 < num_rows and col-1 >= 0 else False
            tile_SE = tile_matrix[row+1, col+1] if row+1 < num_rows and col+1 < num_cols else False
            tile_NE = tile_matrix[row-1, col+1] if row-1 >= 0 and col+1 < num_cols else False

            if not tile_N or not tile_W or not tile_S or not tile_E:
                img[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w, :] *= 0.3
            elif not tile_NW or not tile_SW or not tile_SE or not tile_NE:
                img[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w, :] *= 0.6

    return img.astype(np.uint8)


def apply_boundry_effect_adv(img, tile_matrix, tile_h, tile_w, num_rows, num_cols, boundry_depth=4, intensity_factor=0.2):
    img = img.astype(np.float64)

    for row in range(num_rows):
        for col in range(num_cols):

            for depth in range(1, boundry_depth+1):
                tile_N = tile_matrix[row-depth, col] if row-depth >= 0 else False
                tile_W = tile_matrix[row, col-depth] if col-depth >= 0 else False
                tile_S = tile_matrix[row+depth, col] if row+depth < num_rows else False
                tile_E = tile_matrix[row, col+depth] if col+depth < num_cols else False

                tile_NW = tile_matrix[row-depth, col-depth] if row-depth >= 0 and col-depth >= 0 else False
                tile_SW = tile_matrix[row+depth, col-depth] if row+depth < num_rows and col-depth >= 0 else False
                tile_SE = tile_matrix[row+depth, col+depth] if row+depth < num_rows and col+depth < num_cols else False
                tile_NE = tile_matrix[row-depth, col+depth] if row-depth >= 0 and col+depth < num_cols else False

                if not tile_N or not tile_W or not tile_S or not tile_E:
                    img[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w, :] *= depth * intensity_factor
                elif not tile_NW or not tile_SW or not tile_SE or not tile_NE:
                    img[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w, :] *= depth * intensity_factor
                else:
                    pass

    return img.astype(np.uint8)


def growth_alg_v1(ds_path, input_im, fouling_im, metal_mask, max_seed_points=10, num_time_steps=50, max_new_tiles=10):
    fouling_im_h, fouling_im_w = fouling_im.shape[:2]
    num_rows = input_im.shape[0] // fouling_im.shape[0]
    num_cols = input_im.shape[1] // fouling_im.shape[1]

    tile_matrix = np.zeros((num_rows, num_cols), dtype=bool)
    full_fouling_im = np.zeros(input_im.shape, dtype=np.uint8)

    # num_seed_points = random.randint(1, max_seed_points)
    num_seed_points = max_seed_points  # debugging, change back to random num when testing is done
    seed_points = [(random.randint(0, num_cols-1), random.randint(0, num_rows-1)) for p in range(num_seed_points)]
    num_new_tiles = num_seed_points
    # print('seed points: {}'.format(seed_points))
	
    # loop through each time step
    for time_step in tqdm(range(num_time_steps), desc='Generating fouling...'):

        # region-growing stemming from each seed point
        for (col, row) in seed_points:
            if time_step > 0:
                num_new_tiles = random.randint(1, max_new_tiles)

                for t in range(num_new_tiles):
                    c, r = col, row
                    border_reached = False

                    while tile_matrix[r, c]:
                        direction = random.choice(DIRECTIONS)
                        if direction == 'N':
                            r -= 1
                        elif direction == 'S':
                            r += 1
                        elif direction == 'W':
                            c -= 1
                        elif direction == 'E':
                            c += 1

                        if r < 0 or r >= num_rows or c < 0 or c >= num_cols:
                            border_reached = True
                            break

                    if not border_reached:
                        full_fouling_im[r * fouling_im_h:(r + 1) * fouling_im_h, c * fouling_im_w:(c + 1) * fouling_im_w, :] = fouling_im
                        tile_matrix[r, c] = True

            else:
                full_fouling_im[row * fouling_im_h:(row + 1) * fouling_im_h, col * fouling_im_w:(col + 1) * fouling_im_w, :] = fouling_im
                tile_matrix[row, col] = True
                      
        # apply metal mask to full_fouling_im
        for ch in range(full_fouling_im.shape[2]):
            full_fouling_im[:, :, ch] = cv2.bitwise_and(full_fouling_im[:, :, ch], metal_mask)

        # apply border effect
        full_fouling_im_w_boundry = full_fouling_im.copy()
        full_fouling_im_w_boundry = apply_boundry_effect_adv(full_fouling_im_w_boundry, tile_matrix, fouling_im_h, fouling_im_w, num_rows, num_cols)
        # print('Time step {}'.format(time_step+1))
        # cv2.imshow('img', full_fouling_im_w_boundry)
        # cv2.waitKey(0)

        # save images
        cv2.imwrite(os.path.join(ds_path, 'fouling', 'fouling_{}.png'.format(time_step+1)), full_fouling_im_w_boundry)

    print('Fouling generated.')
    return full_fouling_im


def apply_fouling_spot(base_img, fouling_img):
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]
    alphas = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1].astype(np.float32) * (1 / 255)
    for c in range(base_img.shape[2]):
        base_img[gray < thresh, c] = ((1 - alphas[gray < thresh]) * base_img[gray < thresh, c]) + (alphas[gray < thresh] * fouling_img[gray < thresh, c])
    return base_img


def gen_samples(ds_path, img, metal_mask, save=True):
    samples_path = os.path.join(ds_path, 'samples')
    fouling_path = os.path.join(ds_path, 'fouling')

    # loop through each fouling image
    for file in tqdm(os.listdir(fouling_path), desc='Generating samples...'):
        if file.endswith('.png'):
            clean_img = img.copy() 
            fouling = cv2.imread(os.path.join(fouling_path, file), cv2.IMREAD_COLOR)

            # apply metal mask to fouling image before applying fouling spot
            for ch in range(fouling.shape[2]):
                fouling[:, :, ch] = cv2.bitwise_and(fouling[:, :, ch], metal_mask)

            # apply fouling spot to base image
            sample = apply_fouling_spot(clean_img, fouling)

            # save completed sample image
            if save:
                cv2.imwrite(os.path.join(samples_path, file.replace('fouling_', 'sample_')), sample)
    print('Samples generated.')


if __name__ == '__main__':
    
    # hyperparameters
    input_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/1B_synthetic_rain/image_files/input_sample.png'
    dust_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/1A_synthetic_fouling/image_files/dust1.png'
    metal_path = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/1A_synthetic_fouling/image_files/metal_seg_mask.png'
    ds_path = '/Users/nick_1/PycharmProjects/UWO Masters/data/sds_test_2'

    input_im = cv2.imread(input_path, cv2.IMREAD_COLOR)
    fouling_tile_im = cv2.imread(dust_path, cv2.IMREAD_COLOR)
    metal_mask = cv2.imread(metal_path, cv2.IMREAD_GRAYSCALE)
    metal_mask = cv2.bitwise_not(metal_mask)  # invert metal mask

    # ----- ----- ----- #

    # create_folder_structure(ds_path)
    # gen_tile_matrices(ds_path)
    # gen_fouling_imgs(ds_path, fouling_tile_im, metal_mask)
    # gen_targets(ds_path, metal_mask)
    # gen_samples(ds_path, input_im, metal_mask)

    # cv2.imshow('img', result)
    # cv2.waitKey(0)

    cv2.destroyAllWindows()


