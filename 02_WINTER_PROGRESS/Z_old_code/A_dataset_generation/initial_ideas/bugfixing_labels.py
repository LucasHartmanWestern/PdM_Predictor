
import os
import cv2
import numpy as np


def imshow_and_wait(img):
    cv2.imshow('img', img)
    keys = cv2.waitKey(0) & 0xFF
    if keys == ord('q'):
        cv2.destroyAllWindows()
        quit()


def visualize_target(target_img):
    target_copy = target_img.copy()
    target_copy = target_copy.astype(np.float32)
    target_copy *= (255/(np.unique(target_copy).shape[0]-1))
    target_copy = target_copy.astype(np.uint8)
    imshow_and_wait(target_copy)


def visualize_tile_matrix(target_img):
    target_copy = target_img.copy()
    print(f"\n{np.unique(target_copy)}")
    target_copy = target_copy.astype(np.float32)
    target_copy *= (255/(np.unique(target_copy).shape[0]-1))
    target_copy = target_copy.astype(np.uint8)
    target_copy = cv2.resize(target_copy, (1920, 1080), interpolation=cv2.INTER_AREA)
    imshow_and_wait(target_copy)


if __name__ == '__main__':

    # img = cv2.imread("./TESTING_TILE.png", cv2.IMREAD_GRAYSCALE)
    # visualize_tile_matrix(img)

    tm_base_path = "/Users/nick_1/PycharmProjects/UWO Masters/data/testing_tiles/components/tile_matrices"
    bad_file_count = 0
    for file in os.listdir(tm_base_path):
        if file.endswith(".png"):
            path = os.path.join(tm_base_path, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if len(np.unique(img)) > 7:
                bad_file_count += 1
                visualize_tile_matrix(img)
    print(f"bad tile matrices count: {bad_file_count}")

    
    # tar_base_path = "/Users/nick_1/PycharmProjects/UWO Masters/data/feb24_ds1/full_size/targets"
    # for file in os.listdir(tar_base_path):
    #     if file.endswith(".png"):
    #         path = os.path.join(tar_base_path, file)
    #         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #         print(f"--- file: {file} ---")
    #         print(f"shape: {img.shape}, type: {img.dtype}")
    #         print(f"max value: {img.max()}, min value: {img.min()}")
    #         print(np.unique(img))
    #         visualize_target(img)