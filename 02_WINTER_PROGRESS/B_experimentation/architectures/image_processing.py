import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------- Global Constants ---------- #

ROI_GRID_DIMS = (3, 5)
ROI_DIMS = (360, 360)
IMG_DIMS_CROPPED = (1080, 1800)
IMG_DIMS_FULL = (1080, 1920)

# ---------- Image Processing Methods ---------- #

def preprocess_target(target: np.ndarray, use_roi_cropping=False):
    if use_roi_cropping:
        target = target[:, 60:target.shape[1]-60]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.from_numpy(target).type(torch.LongTensor).to(device=device)


def sample_to_rois(sample: np.ndarray):
    assert sample.shape == (3, 1080, 1920), "ERROR: sample was wrong shape: {}".format(sample.shape)
    rois = []
    sample = sample[:, :, 60:sample.shape[2]-60]
    for i in range(0, sample.shape[1], 360):
        for j in range(0, sample.shape[2], 360):
            roi = sample[:, i:i+360, j:j+360]
            assert roi.shape == (3, 360, 360), "ERROR: roi was wrong shape: {}".format(roi.shape)
            rois.append(torch.from_numpy(roi))
    assert len(rois) == 15, "ERROR: expected 15 ROIs, got {}".format(len(rois))
    return torch.stack(rois)


def rois_to_sample(rois: torch.Tensor):
    rois = rois.detach().cpu().numpy()
    assert rois.shape == (15, 3, 360, 360), "ERROR: rois were wrong shape: {}".format(rois.shape)

    rois = np.split(rois, axis=0)
    result = np.zeros((3, 1080, 1800))

    roi_idx = 0
    for i in range(0, result.shape[1], 360):
        for j in range(0, result.shape[2], 360):
            result[:, i:i+360, j:j+360] = rois[roi_idx]
            roi_idx += 1

    return torch.from_numpy(result)

# ---------- Sample Processing PyTorch Modules ---------- #

class Preprocess_Sample(nn.Module):
    # !!! Always assume batch size of 1 !!!
    def __init__(self, resize_shape=None, use_rois=False):
        if use_rois:
            assert resize_shape is None, "ERROR: cannot resize image if use_rois is True"
            
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resize_shape = resize_shape
        self.use_rois = use_rois

    def forward(self, x):
        # resize sample if specified
        if self.resize_shape is not None:
            x = cv2.resize(x, self.resize_shape, interpolation=cv2.INTER_AREA)

        # transpose sample from (H, W, C) to (C, H, W) and normalize values from [0, 255] to [0, 1]
        x = np.transpose(x, axes=(2, 0, 1))
        x = x.astype(np.float32)
        x *= (1 / 255.0)

        # create a tensor of 15x360x360 ROI's if specified
        if self.use_rois:
            x = sample_to_rois(x)
        
        return x.to(device=self.device)


class Postprocess_Sample(nn.Module):
    def __init__(self, use_rois=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_rois = use_rois

    def forward(self, x):
        if self.use_rois:
            x = rois_to_sample(x)

        return x.to(device=self.device)
