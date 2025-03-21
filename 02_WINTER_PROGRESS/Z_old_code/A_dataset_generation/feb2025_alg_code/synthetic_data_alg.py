import os
import random
import time
import shutil
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

# ---------- Global Constants ---------- #

DIRECTIONS = ['N', 'S', 'W', 'E']

IMG_DIMS_RGB = (1080, 1920, 3)
IMG_DIMS = (1080, 1920)
TILE_DIMS = (36, 20)
MATRIX_DIMS = (IMG_DIMS[0] // TILE_DIMS[0], IMG_DIMS[1] // TILE_DIMS[1])
ROI_DIMS = (360, 360)

HOURS = ['8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm']

# ---------- Helper Methods ---------- #

def imshow_and_wait(img):
    cv2.imshow('img', img)
    keys = cv2.waitKey(0) & 0xFF
    if keys == ord('q'):
        cv2.destroyAllWindows()
        quit()


def create_folder_structure(ds_path):
    fouling_path = os.path.join(ds_path, 'components', 'fouling')
    tile_path = os.path.join(ds_path, 'components', 'tile_matrices')
    os.makedirs(fouling_path, exist_ok=True)
    os.makedirs(tile_path, exist_ok=True)

    full_samples_path = os.path.join(ds_path, 'full_size', 'samples')
    full_targets_path = os.path.join(ds_path, 'full_size', 'targets')
    full_binary_targets_path = os.path.join(ds_path, 'full_size', 'binary_targets')
    os.makedirs(full_samples_path, exist_ok=True)
    os.makedirs(full_targets_path, exist_ok=True)
    os.makedirs(full_binary_targets_path, exist_ok=True)

    rois_samples_path = os.path.join(ds_path, 'rois', 'samples')
    rois_targets_path = os.path.join(ds_path, 'rois', 'targets')
    rois_binary_targets_path = os.path.join(ds_path, 'rois', 'binary_targets')
    os.makedirs(rois_samples_path, exist_ok=True)
    os.makedirs(rois_targets_path, exist_ok=True)
    os.makedirs(rois_binary_targets_path, exist_ok=True)


def apply_fouling_spot(base_img, fouling_img):
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]
    alphas = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1].astype(np.float32) * (1 / 255)
    for c in range(base_img.shape[2]):
        base_img[gray < thresh, c] = ((1 - alphas[gray < thresh]) * base_img[gray < thresh, c]) + (alphas[gray < thresh] * fouling_img[gray < thresh, c])
    return base_img


def visualize_target(target_img):
    target_copy = target_img.copy()
    target_copy = target_copy.astype(np.float32)
    target_copy *= (255.0/(np.unique(target_copy).shape[0]-1))
    target_copy = target_copy.astype(np.uint8)
    imshow_and_wait(target_copy)


def visualize_tile_matrix(target_img):
    target_copy = target_img.copy()
    target_copy = target_copy.astype(np.float32)
    target_copy *= (255/(np.unique(target_copy).shape[0]-1))
    target_copy = target_copy.astype(np.uint8)
    target_copy = cv2.resize(target_copy, (1920, 1080), interpolation=cv2.INTER_AREA)
    imshow_and_wait(target_copy)

# ---------- Algorithm Methods ---------- #

def gen_tile_matrices(ds_path, num_time_steps, num_seed_points, num_intensities, max_new_tiles):
    tile_matrix_bool = np.zeros(MATRIX_DIMS, dtype=bool)
    seed_points = [(random.randint(0, MATRIX_DIMS[0]-1), random.randint(0, MATRIX_DIMS[1]-1)) for p in range(num_seed_points)]
    num_new_tiles = num_seed_points

    # loop through each time step
    for time_step in tqdm(range(num_time_steps), desc='[Part 1/8] Generating Tile Matrices'):

        # region-growing stemming from each seed point
        for (row, col) in seed_points:
            if time_step > 0:
                num_new_tiles = random.randint(1, max_new_tiles)

                for t in range(num_new_tiles):
                    r, c = row, col
                    border_reached = False

                    # search for free tile
                    while tile_matrix_bool[r, c]:
                        direction = random.choice(DIRECTIONS)
                        if direction == 'N':
                            r -= 1
                        elif direction == 'S':
                            r += 1
                        elif direction == 'W':
                            c -= 1
                        elif direction == 'E':
                            c += 1

                        # stop searching for free tile if image border is reached
                        if r < 0 or r >= MATRIX_DIMS[0] or c < 0 or c >= MATRIX_DIMS[1]:
                            border_reached = True
                            break

                    # if free tile is found, add it to the tile matrix
                    if not border_reached:
                        tile_matrix_bool[r, c] = True
                    else:
                        pass
            else:
                tile_matrix_bool[row, col] = True

        # apply border effect to copy of tile matrix
        tile_matrix_img = tile_matrix_bool.copy().astype(np.uint8)
        tile_matrix_img *= num_intensities
        for row in range(MATRIX_DIMS[0]):
            for col in range(MATRIX_DIMS[1]):
                if tile_matrix_bool[row, col]:
                    for depth in range(1, num_intensities):
                        tile_N = tile_matrix_bool[row-depth, col] if row-depth >= 0 else True
                        tile_W = tile_matrix_bool[row, col-depth] if col-depth >= 0 else True
                        tile_S = tile_matrix_bool[row+depth, col] if row+depth < MATRIX_DIMS[0] else True
                        tile_E = tile_matrix_bool[row, col+depth] if col+depth < MATRIX_DIMS[1] else True

                        if (not tile_N) or (not tile_W) or (not tile_S) or (not tile_E):
                            tile_matrix_img[row, col] = depth
                            break

        # save tile matrix with border effect
        cv2.imwrite(os.path.join(ds_path, 'components', 'tile_matrices', 'tile_matrix_{}.png'.format(time_step+1)), tile_matrix_img.astype(np.uint8))


def gen_fouling_imgs(ds_path, tile_im_path, num_intensities, show=False):
    tile_im = cv2.imread(tile_im_path, cv2.IMREAD_COLOR)
    fouling_im_blank = np.zeros(IMG_DIMS_RGB, dtype=np.uint8)
    fouling_path = os.path.join(ds_path, 'components', 'fouling')
    tile_matrices_path = os.path.join(ds_path, 'components', 'tile_matrices')

    # determine intensity multiplier
    intensity_multiplier = 1.0 / num_intensities

    # loop through each tile matrix
    for file in tqdm(os.listdir(tile_matrices_path), desc='[Part 2/8] Generating Fouling Images'):
        if file.endswith('.png'):
            fouling_im = fouling_im_blank.copy().astype(np.float32)
            tile_matrix = cv2.imread(os.path.join(tile_matrices_path, file), cv2.IMREAD_GRAYSCALE)
            
            # paste each tile into the fouling image
            for row in range(MATRIX_DIMS[0]):
                for col in range(MATRIX_DIMS[1]):
                    if tile_matrix[row, col] > 0:
                        fouling_im[row*TILE_DIMS[0]:(row+1)*TILE_DIMS[0], col*TILE_DIMS[1]:(col+1)*TILE_DIMS[1], :] = tile_im
                        if tile_matrix[row, col] < num_intensities:
                            fouling_im[row*TILE_DIMS[0]:(row+1)*TILE_DIMS[0], col*TILE_DIMS[1]:(col+1)*TILE_DIMS[1], :] *= tile_matrix[row, col] * intensity_multiplier

            fouling_im = fouling_im.astype(np.uint8)

            if show:
                imshow_and_wait(fouling_im)

            # save image
            cv2.imwrite(os.path.join(fouling_path, file.replace('tile_matrix_', 'fouling_')), fouling_im)


def gen_targets(ds_path, metal_mask, include_metal_class=True, show=False):
    target_path = os.path.join(ds_path, 'full_size', 'targets')
    binary_target_path = os.path.join(ds_path, 'full_size', 'binary_targets')
    tile_matrices_path = os.path.join(ds_path, 'components', 'tile_matrices')

    # loop through each tile matrix and each fouling image
    for file in tqdm(os.listdir(tile_matrices_path), desc='[Part 3/8] Generating Full-Size Targets'):
        if file.endswith('.png'):
            tile_matrix = cv2.imread(os.path.join(tile_matrices_path, file), cv2.IMREAD_GRAYSCALE)
            target = np.zeros((IMG_DIMS), dtype=np.uint8)
            binary_target = np.zeros((IMG_DIMS), dtype=np.uint8)

            # apply class values to target image
            for row in range(MATRIX_DIMS[0]):
                for col in range(MATRIX_DIMS[1]):
                    if tile_matrix[row, col] > 0:
                        tile_value = tile_matrix[row, col] + 1 if include_metal_class else tile_matrix[row, col]
                        target[row*TILE_DIMS[0]:(row+1)*TILE_DIMS[0], col*TILE_DIMS[1]:(col+1)*TILE_DIMS[1]] = tile_value
                        binary_target[row*TILE_DIMS[0]:(row+1)*TILE_DIMS[0], col*TILE_DIMS[1]:(col+1)*TILE_DIMS[1]] = 1

            # apply metal mask to target image
            target[metal_mask == 0] = 1 if include_metal_class else 0
            binary_target[metal_mask == 0] = 0

            if show:
                visualize_target(target)

            # save target images
            cv2.imwrite(os.path.join(target_path, file.replace('tile_matrix_', 'target_')), target)
            cv2.imwrite(os.path.join(binary_target_path, file.replace('tile_matrix_', 'binary_target_')), binary_target)


def gen_samples(ds_path, src_imgs_path, metal_mask, num_time_steps, show=False):
    samples_path = os.path.join(ds_path, 'full_size', 'samples')
    fouling_path = os.path.join(ds_path, 'components', 'fouling')

    # loop through each time step
    for day in tqdm(range(1, num_time_steps+1), desc='[Part 4/8] Generating Full-Size Samples'):
        fouling_img = cv2.imread(os.path.join(fouling_path, f"fouling_{day}.png"), cv2.IMREAD_COLOR)

        # loop through each hour
        for hour in HOURS:
            src_img = cv2.imread(os.path.join(src_imgs_path, f"day_{day}_{hour}.png"), cv2.IMREAD_COLOR)

            # apply metal mask to fouling image before applying fouling to src image
            for ch in range(fouling_img.shape[2]):
                fouling_img[:, :, ch] = cv2.bitwise_and(fouling_img[:, :, ch], metal_mask)

            # apply fouling spot to base image
            sample = apply_fouling_spot(src_img, fouling_img)

            if show:
                imshow_and_wait(sample)

            # save completed sample image
            cv2.imwrite(os.path.join(samples_path, f"sample_{day}_{hour}.png"), sample)


def gen_rois(ds_path):
    rois_samples_path = os.path.join(ds_path, 'rois', 'samples')
    rois_targets_path = os.path.join(ds_path, 'rois', 'targets')
    rois_binary_targets_path = os.path.join(ds_path, 'rois', 'binary_targets')
    samples_path = os.path.join(ds_path, 'full_size', 'samples')
    targets_path = os.path.join(ds_path, 'full_size', 'targets')
    binary_targets_path = os.path.join(ds_path, 'full_size', 'binary_targets')

    # generate ROIs for samples
    for file in tqdm(os.listdir(samples_path), desc='[Part 5/8] Generating ROIs from Samples'):
        if file.endswith('.png'):
            sample = cv2.imread(os.path.join(samples_path, file), cv2.IMREAD_COLOR)
            sample = sample[:, 60:sample.shape[1]-60, :]
            roi_num = 1
            for i in range(0, sample.shape[0], 360):
                for j in range(0, sample.shape[1], 360):
                    roi = sample[i:i+360, j:j+360, :]
                    if roi.shape != (360, 360, 3):
                        print('ERROR: roi was wrong shape: {}'.format(roi.shape))
                        quit()
                    else:
                        cv2.imwrite(os.path.join(rois_samples_path, file.replace('.png', f'_roi_{roi_num}.png')), roi)
                        roi_num += 1

    # generate ROIs for targets
    for file in tqdm(os.listdir(targets_path), desc='[Part 6/8] Generating ROIs from Targets'):
        if file.endswith('.png'):
            target = cv2.imread(os.path.join(targets_path, file), cv2.IMREAD_GRAYSCALE)
            target = target[:, 60:target.shape[1]-60]
            roi_num = 1
            for i in range(0, target.shape[0], 360):
                for j in range(0, target.shape[1], 360):
                    roi = target[i:i+360, j:j+360]
                    if roi.shape != (360, 360):
                        print('ERROR: roi was wrong shape: {}'.format(roi.shape))
                        quit()
                    else:
                        cv2.imwrite(os.path.join(rois_targets_path, file.replace('.png', f'_roi_{roi_num}.png')), roi)
                        roi_num += 1

    # generate ROIs for binary targets
    for file in tqdm(os.listdir(binary_targets_path), desc='[Part 7/8] Generating ROIs from Binary Targets'):
        if file.endswith('.png'):
            target = cv2.imread(os.path.join(binary_targets_path, file), cv2.IMREAD_GRAYSCALE)
            target = target[:, 60:target.shape[1]-60]
            roi_num = 1
            for i in range(0, target.shape[0], 360):
                for j in range(0, target.shape[1], 360):
                    roi = target[i:i+360, j:j+360]
                    if roi.shape != (360, 360):
                        print('ERROR: roi was wrong shape: {}'.format(roi.shape))
                        quit()
                    else:
                        cv2.imwrite(os.path.join(rois_binary_targets_path, file.replace('.png', f'_roi_{roi_num}.png')), roi)
                        roi_num += 1


def gen_ds_lists(ds_path, num_time_steps):
    full_list_path = os.path.join(ds_path, 'full_size', 'list.txt')
    full_binary_list_path = os.path.join(ds_path, 'full_size', 'binary_list.txt')
    roi_list_path = os.path.join(ds_path, 'rois', 'list.txt')
    roi_binary_list_path = os.path.join(ds_path, 'rois', 'binary_list.txt')

    open(full_list_path, 'w+').close()
    open(full_binary_list_path, 'w+').close()
    open(roi_list_path, 'w+').close()
    open(roi_binary_list_path, 'w+').close()

    with open(full_list_path, 'a') as f1, open(full_binary_list_path, 'a') as f2, open(roi_list_path, 'a') as f3, open(roi_binary_list_path, 'a') as f4:
        for day in tqdm(range(1, num_time_steps+1), desc='[Part 8/8] Generating Dataset Lists'):

            for hour in HOURS:
                full_sample_path = f"samples/sample_{day}_{hour}.png"
                full_target_path = f"targets/target_{day}.png"
                full_binary_target_path = f"binary_targets/binary_target_{day}.png"
                f1.write(f"{full_sample_path},{full_target_path}\n")
                f2.write(f"{full_sample_path},{full_binary_target_path}\n")

                for roi_num in range(1, 16):
                    roi_sample_path = f"samples/sample_{day}_{hour}_roi_{roi_num}.png"
                    roi_target_path = f"targets/target_{day}_roi_{roi_num}.png"
                    roi_binary_target_path = f"binary_targets/binary_target_{day}_roi_{roi_num}.png"
                    f3.write(f"{roi_sample_path},{roi_target_path}\n")
                    f4.write(f"{roi_sample_path},{roi_binary_target_path}\n")

# ---------- Dataset Generation Method ---------- #

def gen_new_dataset(
        ds_path, 
        src_imgs_path,
        num_time_steps, 
        custom_seed=None,
        max_seed_points=10, 
        min_seed_points=None, 
        max_new_tiles_per_time_step=10, 
        fouling_intensity_levels=5, # originally was 4
        include_metal_class=True,
        tile_im_path='dust1.png',
        metal_mask_path='metal_seg_mask_inv.png',
):
    # create folder structure
    create_folder_structure(ds_path)

    # apply seed for reproducibility
    seed = int(custom_seed) if custom_seed is not None else int.from_bytes(os.urandom(4), byteorder="big")
    print(f"Generating New Dataset Using Seed: {seed}")
    random.seed(seed)

    # determine number of tile seed points
    num_tile_seed_points = max_seed_points if min_seed_points is None else random.randint(min_seed_points, max_seed_points)

    # save hyperparameters file and apply seed
    file_path = os.path.join(ds_path, 'hyperparams.txt')
    open(file_path, 'w+').close()
    with open(file_path, 'w') as f:
        f.write(f"reproducibility seed: {seed}\n")
        f.write(f"num time steps: {num_time_steps}\n")
        f.write(f"num tile seed points: {num_tile_seed_points}\n")
        f.write(f"max new tiles per time step: {max_new_tiles_per_time_step}\n")
        f.write(f"num fouling intensity levels: {fouling_intensity_levels}\n")
        f.write(f"include metal class in targets: {include_metal_class}\n")

    # generate building blocks
    gen_tile_matrices(ds_path, num_time_steps, num_tile_seed_points, fouling_intensity_levels, max_new_tiles_per_time_step)
    gen_fouling_imgs(ds_path, tile_im_path, fouling_intensity_levels)

    # generate full-size targets and samples
    metal_mask = cv2.imread(metal_mask_path, cv2.IMREAD_GRAYSCALE)
    gen_targets(ds_path, metal_mask, include_metal_class)
    gen_samples(ds_path, src_imgs_path, metal_mask, num_time_steps)

    # generate ROIs dataset
    gen_rois(ds_path)
    gen_ds_lists(ds_path, num_time_steps)

    print('New Dataset Generated.')


# ---------- Main Method ---------- #

if __name__ == '__main__':
    # TODO: look for related work on tiling algorithms to help for academic justification for our use of it.
    # TODO: make sure what we are doing is multi-class segmentation.
    
    # hyperparameters
    num_days = 43
    src_folder_path = '/Users/nick_1/PycharmProjects/UWO Masters/data/src_images'
    dataset_path = '/Users/nick_1/PycharmProjects/UWO Masters/data/feb28_ds1'
    

    # ----- ----- ----- #

    gen_new_dataset(dataset_path, src_folder_path, num_days)

    # ----- ----- ----- #

    # dataset_path = '/Users/nick_1/PycharmProjects/UWO Masters/data'
    # for i in range(5):
    #     ds_num_path = os.path.join(dataset_path, f'feb28_ds{i+1}')
    #     gen_new_dataset(ds_num_path, src_folder_path, num_days)

    # ----- ----- ----- #

    cv2.destroyAllWindows()


