from datetime import datetime
import pandas as pd
import os


# Global Constants
MODELS = ['cgnet', 'unet']
INPUT_TYPES = ['rois', 'full']
TARGET_TYPES = ['binary', 'multiclass']


if __name__ == "__main__":
    # Hyperparameters
    path_to_results = '/Users/nick_1/PycharmProjects/UWO Masters/ICML_Jun7_Backup'
    save_folder = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/03_ICML_Results'
    
    # BINARY
    target_type = 'binary'
    headers = ['CNN Model', 'Input Type', 'Target Type', 'Trial', 'Day', 'Hour', 'F1 Score', 'Jaccard Index', 'TP', 'FP', 'FN', 'TN']
    df = pd.DataFrame(columns=headers)
    for trial in range(1, 4):
        for model in MODELS:
            for input_type in INPUT_TYPES:
                csv_path = os.path.join(path_to_results, f'trial_{trial}', f'{model}_{input_type}_{target_type}', 'CGS_testing_metrics.csv')
                if not os.path.exists(csv_path):
                    print(f"File not found: {csv_path}")
                    continue
                current_df = pd.read_csv(csv_path)
                current_df['CNN Model'] = model
                current_df['Input Type'] = input_type
                current_df['Target Type'] = target_type
                current_df['Trial'] = trial
                current_df = current_df[headers]
                df = pd.concat([df, current_df], ignore_index=True)
    df.to_csv(os.path.join(save_folder, 'CGS_binary_results.csv'), index=False)



    # MULTICLASS
    target_type = 'multiclass'
    headers = ['CNN Model', 
               'Input Type', 
               'Target Type', 
               'Trial',
               'Day',
               'Hour',
               'Class 0 F1 Score', 
               'Class 1 F1 Score', 
               'Class 2 F1 Score', 
               'Class 3 F1 Score', 
               'Class 4 F1 Score', 
               'Class 5 F1 Score', 
               'Class 6 F1 Score',
               'Class 0 Jaccard Index', 
               'Class 1 Jaccard Index', 
               'Class 2 Jaccard Index', 
               'Class 3 Jaccard Index', 
               'Class 4 Jaccard Index', 
               'Class 5 Jaccard Index', 
               'Class 6 Jaccard Index']
    df = pd.DataFrame(columns=headers)
    for trial in range(1, 4):
        for model in MODELS:
            for input_type in INPUT_TYPES:
                csv_path = os.path.join(path_to_results, f'trial_{trial}', f'{model}_{input_type}_{target_type}', 'CGS_testing_metrics.csv')
                if not os.path.exists(csv_path):
                    print(f"File not found: {csv_path}")
                    continue
                current_df = pd.read_csv(csv_path)
                current_df['CNN Model'] = model
                current_df['Input Type'] = input_type
                current_df['Target Type'] = target_type
                current_df['Trial'] = trial
                current_df = current_df[headers]
                df = pd.concat([df, current_df], ignore_index=True)
    df.to_csv(os.path.join(save_folder, 'CGS_multiclass_results.csv'), index=False)

