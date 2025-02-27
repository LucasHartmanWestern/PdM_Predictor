import os
import random
import time

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

BORDER_DEPTH = 3

HOURS = ['8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm']

# ---------- Methods ---------- #

def imshow_and_wait(img):
    cv2.imshow('img', img)
    keys = cv2.waitKey(0) & 0xFF
    if keys == ord('q'):
        cv2.destroyAllWindows()
        quit()


def visualize_target(target_img, include_metal_class=True):
    target_copy = target_img.copy()
    target_copy = target_copy.astype(np.float32)
    num_classes = BORDER_DEPTH + 2 if include_metal_class else BORDER_DEPTH + 1
    target_copy *= (255/num_classes)
    target_copy = target_copy.astype(np.uint8)
    imshow_and_wait(target_copy)


def view_targets(data_path, num_time_steps):
    for day in range(1, num_time_steps+1):
        img_path = f"targets/target_{day}.png"
        print(f"Showing file: {img_path}")
        img = cv2.imread(os.path.join(data_path, img_path), cv2.IMREAD_GRAYSCALE)
        visualize_target(img)
    print('Done.')


# ---------- Main Method ---------- #

if __name__ == '__main__':
    
    # hyperparameters
    folder_path = '/Users/nick_1/PycharmProjects/UWO Masters/data/ds1'
    num_days = 43

    # ----- ----- ----- #

    view_targets(folder_path, num_days)

    # ----- ----- ----- #

    cv2.destroyAllWindows()


