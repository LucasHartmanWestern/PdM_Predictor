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


def gen_target(fouling_img):
    fouling_img = cv2.cvtColor(fouling_img, cv2.COLOR_BGR2GRAY)
    fouling_img = cv2.fastNlMeansDenoising(fouling_img, None, 20, 7, 21)
    fouling_img = cv2.threshold(fouling_img, 0, 255, cv2.THRESH_OTSU)[1]
    return fouling_img


def get_avg_threshold(ds_path):
    thresholds = []
    fouling_path = os.path.join(ds_path, 'fouling')
    for file_num in range(1, 51):
        im = cv2.imread(os.path.join(fouling_path, 'fouling_{}.png'.format(file_num)), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.fastNlMeansDenoising(im, None, 20, 7, 21)
        thresholds.append(cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)[0])
    return int(np.mean(thresholds))


def gen_targets(ds_path):
    # start_time = time.time()
    # thresh = get_avg_threshold(ds_path)  # TODO: ask santiago if realism is more important than accuracy
    # print('Avg thresh function took {:.5f} seconds'.format(time.time() - start_time))
    # print('Average threshold: {}'.format(thresh))
    fouling_path = os.path.join(ds_path, 'fouling')
    for file_num in range(1, 51):
        im = cv2.imread(os.path.join(fouling_path, 'fouling_{}.png'.format(file_num)), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.fastNlMeansDenoising(im, None, 20, 7, 21)
        # t, im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)  # TODO: find a way to get a consistently good threshold value!!!
        im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY)[1]  # basic blurry version
        # im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)[1]  # more accurate version with threshold
        # print('Showing target #{}:\tthresh={:.5f}'.format(file_num, t))
        print('Showing target #{}'.format(file_num))
        cv2.imshow('img', im)
        cv2.waitKey(0)


def gen_targets_OLD(ds_path, metal_mask, binary_label=False, label_border_depth=LABEL_DEPTH_UNCHANGED):
    target_path = os.path.join(ds_path, 'targets')
    fouling_path = os.path.join(ds_path, 'fouling')
    tile_matrices_path = os.path.join(ds_path, 'tile_matrices')

    # loop through each tile matrix and each fouling image
    for file in tqdm(os.listdir(tile_matrices_path), desc='Generating targets...'):
        if file.endswith('.png'):
            tile_matrix = cv2.imread(os.path.join(tile_matrices_path, file), cv2.IMREAD_GRAYSCALE)
            target = cv2.imread(os.path.join(fouling_path, file.replace('tile_matrix_', 'fouling_')), cv2.IMREAD_COLOR)


            # imshow_and_wait(target)

            # apply label border depth to target image
            if label_border_depth != LABEL_DEPTH_UNCHANGED:
                for row in range(MATRIX_DIMS[0]):
                    for col in range(MATRIX_DIMS[1]):
                        if tile_matrix[row, col] > 0 and tile_matrix[row, col] <= label_border_depth:
                            target[row*TILE_DIMS[0]:(row+1)*TILE_DIMS[0], col*TILE_DIMS[1]:(col+1)*TILE_DIMS[1], :] = 0


            # imshow_and_wait(target)

            # binarize target image
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            target = cv2.fastNlMeansDenoising(target, None, 30, 7, 21)
            target = cv2.threshold(target, 0, 255, cv2.THRESH_BINARY)[1]

            # apply metal mask to binary target image
            if binary_label:
                target = cv2.bitwise_and(target, metal_mask)

            # create BGR target image
            else:
                target_gray = target.copy()
                target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
                # background class = blue
                target[:, :, 0][target_gray == 0] = 255
                target[:, :, 1][target_gray == 0] = 0
                target[:, :, 2][target_gray == 0] = 0
                # fouling class = green
                target[:, :, 0][target_gray > 0] = 0
                target[:, :, 1][target_gray > 0] = 255
                target[:, :, 2][target_gray > 0] = 0  
                # metal class = red
                target[:, :, 0][metal_mask == 0] = 0
                target[:, :, 1][metal_mask == 0] = 0
                target[:, :, 2][metal_mask == 0] = 255

            # save target image
            # cv2.imwrite(os.path.join(target_path, file.replace('tile_matrix_', 'target_')), target)
    print('Targets generated.')


if __name__ == '__main__':
    # hyperparameters
    ds_path = '/Users/nick_1/PycharmProjects/UWO Masters/data/sds_test_1'

    # ----- ----- ----- #

    gen_targets(ds_path)

    # full_fouling_im = cv2.imread(os.path.join(ds_path, 'fouling', 'fouling_20.png'), cv2.IMREAD_COLOR)
    # cv2.imshow('img', full_fouling_im)
    # cv2.waitKey(0)
	
    # start_time = time.time()
    # result = gen_target(full_fouling_im)
    # print('Function took {:.5f} seconds'.format(time.time() - start_time))
    # cv2.imshow('img', result)
    # cv2.waitKey(0)

    cv2.destroyAllWindows()
