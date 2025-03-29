import time

import cv2
import numpy as np
import torch

# ---------- Global Constants ---------- #

ROI_DIMS = (360, 360)
ROI_DIMS_RGB = (3, 360, 360)
ROI_BATCH_DIMS = (15, 360, 360)

IMG_DIMS_CROPPED = (1080, 1800)
IMG_DIMS_CROPPED_RGB = (3, 1080, 1800)

IMG_DIMS_FULL = (1080, 1920)
IMG_DIMS_FULL_RGB = (3, 1080, 1920)

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
        target = torch.squeeze(target, 0).detach().cpu().numpy()
        assert target.shape == IMG_DIMS_CROPPED, "ERROR: target was wrong shape: {}".format(target.shape)
        rois = []
        for i in range(0, target.shape[0], 360):
            for j in range(0, target.shape[1], 360):
                roi = target[i:i+360, j:j+360]
                assert roi.shape == ROI_DIMS, "ERROR: roi was wrong shape: {}".format(roi.shape)
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
        sample = torch.squeeze(sample, 0).detach().cpu().numpy()
        assert sample.shape == IMG_DIMS_CROPPED_RGB, "ERROR: sample was wrong shape: {}".format(sample.shape)
        rois = []
        for i in range(0, sample.shape[1], 360):
            for j in range(0, sample.shape[2], 360):
                roi = sample[:, i:i+360, j:j+360]
                assert roi.shape == ROI_DIMS_RGB, "ERROR: roi was wrong shape: {}".format(roi.shape)
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


def postprocess_seg_mask(output_image: torch.Tensor, num_classes: int, use_rois: bool, verbose=False, show=False):
    start_time = time.time()
    if use_rois:
        # re-stitching ROIs into full image
        output_image = output_image.detach().cpu().numpy()

        if len(output_image.shape) == 4:
            # output image format
            assert output_image.shape == (15, num_classes, 360, 360), "ERROR: output rois are wrong shape: {}".format(output_image.shape)
            temp = np.zeros((num_classes, 1080, 1800))
            roi_idx = 0
            for i in range(0, temp.shape[1], 360):
                for j in range(0, temp.shape[2], 360):
                    temp[:, i:i+360, j:j+360] = output_image[roi_idx, :, :, :]
                    roi_idx += 1

            result = np.zeros(IMG_DIMS_CROPPED)
            for ch in range(temp.shape[0]):
                result[temp[ch, :, :] > 0] = ch

        else:
            # target image format
            assert output_image.shape == (15, 360, 360), "ERROR: target rois are wrong shape: {}".format(output_image.shape)
            result = np.zeros(IMG_DIMS_CROPPED)
            roi_idx = 0
            for i in range(0, result.shape[0], 360):
                for j in range(0, result.shape[1], 360):
                    result[i:i+360, j:j+360] = output_image[roi_idx, :, :]
                    roi_idx += 1

    else:
        # post-processing full image
        output_image = torch.squeeze(output_image, 0).detach().cpu().numpy()

        if len(output_image.shape) == 3:
            # output image format
            assert output_image.shape == (num_classes, 1080, 1800), "ERROR: output image is wrong shape: {}".format(output_image.shape)
            result = np.zeros(IMG_DIMS_CROPPED)
            for ch in range(output_image.shape[0]):
                result[output_image[ch, :, :] > 0] = ch

        else:
            # target image format
            assert output_image.shape == IMG_DIMS_CROPPED, "ERROR: target image is wrong shape: {}".format(output_image.shape)
            result = output_image

    # final checks and return
    assert result.shape == IMG_DIMS_CROPPED, "ERROR: result was wrong shape: {}".format(result.shape)
    if verbose:
        print(f"DEBUG: seg mask unique values: {np.unique(result)}")
        print(f"DEBUG: Time taken to postprocess seg mask: {time.time() - start_time} seconds")

    if show:
        visualize_seg_mask(result, num_classes)

    result = np.expand_dims(result, axis=0)
    result = torch.from_numpy(result).type(torch.FloatTensor)
    return result.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

