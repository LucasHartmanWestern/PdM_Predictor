import time

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------- Global Constants ---------- #

# ROI_GRID_DIMS = (3, 5)
# ROI_DIMS = (360, 360)
# IMG_DIMS_CROPPED = (1080, 1800)
# IMG_DIMS_FULL = (1080, 1920)

# ---------- Image Processing Methods ---------- #

def imshow_and_wait(img):
    cv2.imshow('img', img)
    keys = cv2.waitKey(0) & 0xFF
    if keys == ord('q'):
        cv2.destroyAllWindows()
        quit()


def visualize_seg_mask(img, num_classes=None):
    img_copy = img.copy()
    img_copy = img_copy.astype(np.float32)
    unique_vals = np.unique(img_copy).shape[0]-1 if num_classes is None else num_classes-1
    img_copy *= (255.0/unique_vals)
    img_copy = img_copy.astype(np.uint8)
    imshow_and_wait(img_copy)


def preprocess_target(target: torch.Tensor, use_rois: bool, resize_shape=None, verbose=False):
    start_time = time.time()
    if use_rois:
        # create a tensor of 15x360x360 ROI's if specified
        target = torch.squeeze(target, 0)
        target = target.detach().cpu().numpy()
        assert target.shape == (1080, 1920), "ERROR: target was wrong shape: {}".format(target.shape)
        rois = []
        target = target[:, 60:target.shape[1]-60]
        for i in range(0, target.shape[0], 360):
            for j in range(0, target.shape[1], 360):
                roi = target[i:i+360, j:j+360]
                assert roi.shape == (360, 360), "ERROR: roi was wrong shape: {}".format(roi.shape)
                rois.append(torch.from_numpy(roi).type(torch.LongTensor))
        assert len(rois) == 15, "ERROR: expected 15 ROIs, got {}".format(len(rois))
        result = torch.stack(rois)

    else:
        # unchanged target
        result = target
        if resize_shape is not None:
            # resize target if specified
            result = cv2.resize(result, resize_shape, interpolation=cv2.INTER_AREA)

    result = result.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if verbose:
        print(f"DEBUG: Time taken to preprocess target: {time.time() - start_time} seconds")
    return result


def preprocess_sample(sample: torch.Tensor, use_rois: bool, resize_shape=None, verbose=False):
    start_time = time.time()
    if use_rois:
        # create a tensor of 15x360x360 ROI's if specified
        sample = torch.squeeze(sample, 0)
        sample = sample.detach().cpu().numpy()
        assert sample.shape == (3, 1080, 1920), "ERROR: sample was wrong shape: {}".format(sample.shape)
        rois = []
        sample = sample[:, :, 60:sample.shape[2]-60]
        for i in range(0, sample.shape[1], 360):
            for j in range(0, sample.shape[2], 360):
                roi = sample[:, i:i+360, j:j+360]
                assert roi.shape == (3, 360, 360), "ERROR: roi was wrong shape: {}".format(roi.shape)
                rois.append(torch.from_numpy(roi).type(torch.FloatTensor))
        assert len(rois) == 15, "ERROR: expected 15 ROIs, got {}".format(len(rois))
        result = torch.stack(rois)
    else: 
        # unchanged sample
        result = sample
        if resize_shape is not None:
            # resize sample if specified
            result = cv2.resize(result, resize_shape, interpolation=cv2.INTER_AREA)

    result = result.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if verbose:
        print(f"DEBUG: Time taken to preprocess sample: {time.time() - start_time} seconds")
    return result


def postprocess_seg_mask(image_batch: torch.Tensor, num_classes: int, use_rois: bool, verbose=False):
    start_time = time.time()
    if use_rois:
        image_batch = image_batch.detach().cpu().numpy()

        if len(image_batch.shape) == 4:
            # output image format
            assert image_batch.shape == (15, num_classes, 360, 360), "ERROR: rois were wrong shape: {}".format(image_batch.shape)

            temp = np.zeros((num_classes, 1080, 1800))
            roi_idx = 0
            for i in range(0, temp.shape[1], 360):
                for j in range(0, temp.shape[2], 360):
                    temp[:, i:i+360, j:j+360] = image_batch[roi_idx, :, :, :]
                    roi_idx += 1

            result = np.zeros((1080, 1800))
            for ch in range(temp.shape[0]):
                result[temp[ch, :, :] > 0] = ch

        else:
            # target image format
            assert image_batch.shape == (15, 360, 360), "ERROR: rois were wrong shape: {}".format(image_batch.shape)

            result = np.zeros((1080, 1800))
            roi_idx = 0
            for i in range(0, result.shape[0], 360):
                for j in range(0, result.shape[1], 360):
                    result[i:i+360, j:j+360] = image_batch[roi_idx, :, :]
                    roi_idx += 1

    else:
        image_batch = torch.squeeze(image_batch, 0).detach().cpu().numpy()
        result = np.zeros((1080, 1800))
        for ch in range(result.shape[0]):
            result[image_batch[ch, :, :] > 0] = ch

    assert result.shape == (1080, 1800), "ERROR: result was wrong shape: {}".format(result.shape)
    if verbose:
        print(f"DEBUG: seg mask unique values: {np.unique(result)}")
        print(f"DEBUG: Time taken to postprocess seg mask: {time.time() - start_time} seconds")
    visualize_seg_mask(result, num_classes)

