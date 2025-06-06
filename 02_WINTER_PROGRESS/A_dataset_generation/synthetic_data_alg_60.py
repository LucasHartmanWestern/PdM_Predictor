import os
import random
import time
import shutil
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

# ---------- Global Constants ---------- #

NUM_DAYS = 60
CROP_AMOUNT = 60

TILE_DIMS = (36, 20)
IMG_DIMS = (1080, 1920)
IMG_DIMS_RGB = (1080, 1920, 3)
CROPPED_IMG_DIMS = (1080, 1800)
CROPPED_IMG_DIMS_RGB = (1080, 1800, 3)
MATRIX_DIMS = (IMG_DIMS[0] // TILE_DIMS[0], IMG_DIMS[1] // TILE_DIMS[1])

PARTITIONS = ['train', 'val', 'test']
DIRECTIONS = ['N', 'S', 'W', 'E']
HOURS = ['8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm']

# ---------- Helper Methods ---------- #

def imshow_and_wait(img):
    cv2.imshow('img', img)
    keys = cv2.waitKey(0) & 0xFF
    if keys == ord('q'):
        cv2.destroyAllWindows()
        quit()


def get_seeds(save_path, custom_dataset_seed=None):
    '''
    Generates a dataset seed and 3 partition seeds for training, validation, and testing.

    Also writes the dataset seed to a file located at <save_path>/dataset_seed.txt
    '''
    # generate dataset seed
    if custom_dataset_seed is None:
        dataset_seed = int.from_bytes(os.urandom(4), byteorder="big")
    else:
        dataset_seed = custom_dataset_seed

    # write dataset seed to file
    file_path = os.path.join(save_path, 'dataset_seed.txt')
    open(file_path, 'w+').close()
    with open(file_path, 'w') as f:
        f.write(f"{dataset_seed}\n")

    # generate partition seeds
    random.seed(dataset_seed)
    partition_seeds = [random.randint(100000000, 999999999) for _ in range(3)]

    return dataset_seed, partition_seeds


def create_folder_structure(save_path):
    fouling_path = os.path.join(save_path, 'components', 'fouling')
    tile_path = os.path.join(save_path, 'components', 'tile_matrices')
    os.makedirs(fouling_path, exist_ok=True)
    os.makedirs(tile_path, exist_ok=True)

    samples_path = os.path.join(save_path, 'samples')
    binary_targets_path = os.path.join(save_path, 'targets', 'binary')
    multiclass_targets_path = os.path.join(save_path, 'targets', 'multiclass')
    os.makedirs(samples_path, exist_ok=True)
    os.makedirs(binary_targets_path, exist_ok=True)
    os.makedirs(multiclass_targets_path, exist_ok=True)


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

def gen_tile_matrices(save_path, num_seed_points, num_intensities, max_new_tiles):
    tile_matrix_bool = np.zeros(MATRIX_DIMS, dtype=bool)
    seed_points = [(random.randint(0, MATRIX_DIMS[0]-1), random.randint(0, MATRIX_DIMS[1]-1)) for p in range(num_seed_points)]
    num_new_tiles = num_seed_points

    # loop through each time step
    for time_step in tqdm(range(NUM_DAYS), desc='[Part 1/5] Generating Tile Matrices'):

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
        cv2.imwrite(os.path.join(save_path, 'components', 'tile_matrices', 'tile_matrix_{}.png'.format(time_step+1)), tile_matrix_img.astype(np.uint8))


def gen_fouling_imgs(save_path, tile_im_path, num_intensities, show=False):
    tile_im = cv2.imread(tile_im_path, cv2.IMREAD_COLOR)
    fouling_im_blank = np.zeros(IMG_DIMS_RGB, dtype=np.uint8)
    fouling_path = os.path.join(save_path, 'components', 'fouling')
    tile_matrices_path = os.path.join(save_path, 'components', 'tile_matrices')

    # determine intensity multiplier
    intensity_multiplier = 1.0 / num_intensities

    # loop through each tile matrix
    for file in tqdm(os.listdir(tile_matrices_path), desc='[Part 2/5] Generating Fouling Images'):
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


def gen_targets(save_path, metal_mask, crop_width, include_metal_class=True, show=False):
    multiclass_target_path = os.path.join(save_path, 'targets', 'multiclass')
    binary_target_path = os.path.join(save_path, 'targets', 'binary')
    tile_matrices_path = os.path.join(save_path, 'components', 'tile_matrices')

    # loop through each tile matrix and each fouling image
    for file in tqdm(os.listdir(tile_matrices_path), desc='[Part 3/5] Generating Targets'):
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

            # crop target images if specified
            if crop_width:
                target = target[:, CROP_AMOUNT:target.shape[1]-CROP_AMOUNT]
                binary_target = binary_target[:, CROP_AMOUNT:binary_target.shape[1]-CROP_AMOUNT]
                assert target.shape == CROPPED_IMG_DIMS, f"Target image shape is {target.shape}, expected {CROPPED_IMG_DIMS}"
                assert binary_target.shape == CROPPED_IMG_DIMS, f"Binary Target image shape is {binary_target.shape}, expected {CROPPED_IMG_DIMS}"

            # save target images
            cv2.imwrite(os.path.join(multiclass_target_path, file.replace('tile_matrix_', 'target_')), target)
            cv2.imwrite(os.path.join(binary_target_path, file.replace('tile_matrix_', 'target_')), binary_target)


def gen_samples(save_path, src_imgs_path, metal_mask, crop_width, show=False):
    samples_path = os.path.join(save_path, 'samples')
    fouling_path = os.path.join(save_path, 'components', 'fouling')

    # loop through each time step
    for day in tqdm(range(1, NUM_DAYS+1), desc='[Part 4/5] Generating Samples'):
        fouling_img = cv2.imread(os.path.join(fouling_path, f"fouling_{day}.png"), cv2.IMREAD_COLOR)

        # loop through each hour
        for hour in HOURS:
            src_img_file = f"day_{day}_{hour}.png"

            # esnure file exists
            if os.path.exists(os.path.join(src_imgs_path, src_img_file)):
                src_img = cv2.imread(os.path.join(src_imgs_path, src_img_file), cv2.IMREAD_COLOR)

                # apply metal mask to fouling image before applying fouling to src image
                for ch in range(fouling_img.shape[2]):
                    fouling_img[:, :, ch] = cv2.bitwise_and(fouling_img[:, :, ch], metal_mask)

                # apply fouling spot to base image
                sample = apply_fouling_spot(src_img, fouling_img)

                if show:
                    imshow_and_wait(sample)

                # crop sample images if specified
                if crop_width:
                    sample = sample[:, CROP_AMOUNT:sample.shape[1]-CROP_AMOUNT, :]
                    assert sample.shape == CROPPED_IMG_DIMS_RGB, f"Sample image shape is {sample.shape}, expected {CROPPED_IMG_DIMS_RGB}"

                # save completed sample image
                cv2.imwrite(os.path.join(samples_path, f"sample_{day}_{hour}.png"), sample)


def gen_ds_lists(save_path):
    binary_list_path = os.path.join(save_path, 'binary_list.txt')
    multiclass_list_path = os.path.join(save_path, 'multiclass_list.txt')
    open(binary_list_path, 'w+').close()
    open(multiclass_list_path, 'w+').close()

    with open(binary_list_path, 'a') as f1, open(multiclass_list_path, 'a') as f2:
        for day in tqdm(range(1, NUM_DAYS+1), desc='[Part 5/5] Generating Dataset Lists'):
            for hour in HOURS:
                sample_path = f"samples/sample_{day}_{hour}.png"
                binary_target_path = f"targets/binary/target_{day}.png"
                multiclass_target_path = f"targets/multiclass/target_{day}.png"
                if os.path.exists(os.path.join(save_path, sample_path)):
                    f1.write(f"{sample_path},{binary_target_path}\n")
                    f2.write(f"{sample_path},{multiclass_target_path}\n")

# ---------- Dataset Generation Method ---------- #

def gen_new_dataset(
        ds_path, 
        src_imgs_path,
        custom_seed=None,
        max_seed_points=10, 
        min_seed_points=1, 
        max_new_tiles_per_time_step=10, 
        fouling_intensity_levels=5,
        include_metal_class=True,
        crop_image_widths=True,
        tile_im_path='dust1.png',
        metal_mask_path='metal_seg_mask_inv.png',
):
    os.makedirs(ds_path, exist_ok=True)
    
    # create main seed
    ds_seed, partition_seeds = get_seeds(ds_path, custom_seed)

    # iterate over partitions
    for i in range(len(PARTITIONS)):
        print(f"\n--- [{i+1}/3] Generating {PARTITIONS[i]} partition ---\n")

        # create save path and seed algorithm for current partition
        save_path = os.path.join(ds_path, PARTITIONS[i])
        create_folder_structure(save_path)
        random.seed(partition_seeds[i])

        # determine number of tile seed points
        num_tile_seed_points = max_seed_points if min_seed_points is None else random.randint(min_seed_points, max_seed_points)

        # save hyperparameters file and apply seed
        file_path = os.path.join(save_path, 'hyperparameters.txt')
        open(file_path, 'w+').close()
        with open(file_path, 'w') as f:
            f.write(f"\n--- {PARTITIONS[i]} partition ---\n")
            f.write(f"\nReproducibility:\n")
            f.write(f"\tDataset seed: {ds_seed}\n")
            f.write(f"\tPartition seed: {partition_seeds[i]}\n")
            f.write(f"\nAlgorithm Parameters:\n")
            f.write(f"\tnum time steps: {NUM_DAYS}\n")
            f.write(f"\tnum tile seed points: {num_tile_seed_points}\n")
            f.write(f"\tmax new tiles per time step: {max_new_tiles_per_time_step}\n")
            f.write(f"\tnum fouling intensity levels: {fouling_intensity_levels}\n")
            f.write(f"\tinclude metal class in targets: {include_metal_class}\n")

        # generate building blocks
        gen_tile_matrices(save_path, num_tile_seed_points, fouling_intensity_levels, max_new_tiles_per_time_step)
        gen_fouling_imgs(save_path, tile_im_path, fouling_intensity_levels)

        # generate full-size targets and samples
        metal_mask = cv2.imread(metal_mask_path, cv2.IMREAD_GRAYSCALE)
        gen_targets(save_path, metal_mask, crop_image_widths, include_metal_class)
        gen_samples(save_path, src_imgs_path, metal_mask, crop_image_widths)
        gen_ds_lists(save_path)

    print('\n--- New Dataset Generated ---\n')

# ---------- Main Method ---------- #

if __name__ == '__main__':
    
    # hyperparameters
    src_folder_path = '/Users/nick_1/PycharmProjects/UWO Masters/data_60/src_images'
    dataset_path = '/Users/nick_1/PycharmProjects/UWO Masters/data_60/mar28_ds3'
    
    # ----- ----- ----- #

    gen_new_dataset(dataset_path, src_folder_path)

    # ----- ----- ----- #

    cv2.destroyAllWindows()


