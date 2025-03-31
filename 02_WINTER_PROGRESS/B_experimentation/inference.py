import argparse
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import *
from image_processing import postprocess_seg_mask, visualize_seg_mask
from custom_ds_60 import Custom_DS_60

# ---------- Inference Method ---------- #

def run_inference(ds_name):
    test_ds = Custom_DS_60(ds_name, 'test', True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model_count = 1
    for model_name in ["unet", "cgnet"]:
        for binary_targets in [True, False]:
            for use_rois in [True, False]:

                # get model results folder
                model_results_folder = f"{model_name}_{'rois' if use_rois else 'full'}_{'binary' if binary_targets else 'multiclass'}"
                weights_file = os.path.join(".", "RESULTS", model_results_folder, "best_weights.pth")
                save_path = os.path.join(".", "RESULTS", model_results_folder, "outputs")
                classes = 2 if binary_targets else 7
                os.makedirs(save_path, exist_ok=True)

                # load model with weights
                model = get_model(model_name, num_classes=classes, use_rois=use_rois)
                model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu'), weights_only=True))
                
                # run inference for this model
                model.eval()
                with torch.no_grad():
                    for day_num, hour_num, sample, _ in tqdm(test_loader, desc=f"[{model_count}/8] {save_path.split('/')[-2]}"):
                        save_file_path = os.path.join(save_path, f"output_{day_num.item()}_{hour_num[0]}.png")
                        output = model(sample)
                        output = postprocess_seg_mask(output, classes, model.use_rois)
                        cv2.imwrite(save_file_path, visualize_seg_mask(output, classes))
                        del day_num, hour_num, sample, output

                model_count += 1

    print("\ninference script finished.\n")

# ---------- Main Method ---------- #

if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, required=True, help="dataset folder name (str)")
    args = parser.parse_args()

    # run inference
    run_inference(args.dataset.lower())