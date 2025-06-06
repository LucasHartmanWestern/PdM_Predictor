from datetime import datetime
import pandas as pd
import os


# Global Constants
MODELS = ['cgnet', 'unet']
INPUT_TYPES = ['rois', 'full']
TARGET_TYPES = ['binary', 'multiclass']


if __name__ == "__main__":
    # Hyperparameters
    path_to_results = '/Users/nick_1/PycharmProjects/UWO Masters/MAR28_RESULTS'
    
    # BINARY TRIAL 1
    target_type = 'binary'
    headers = ['CNN Model', 'Input Type', 'Target Type', 'Day', 'Hour', 'F1 Score', 'Jaccard Index']
    df = pd.DataFrame(columns=headers)
    for model in MODELS:
        for input_type in INPUT_TYPES:
            current_df = pd.read_csv(os.path.join(path_to_results, f'{model}_{input_type}_{target_type}', 'testing_metrics.csv'))
            current_df = current_df.drop(columns=['Class 0 F1 Score', 'Class 0 Jaccard Index'])
            current_df = current_df.rename(columns={'Class 1 F1 Score': 'F1 Score', 'Class 1 Jaccard Index': 'Jaccard Index'})
            current_df['CNN Model'] = model
            current_df['Input Type'] = input_type
            current_df['Target Type'] = target_type
            current_df = current_df[headers]
            df = pd.concat([df, current_df], ignore_index=True)
    df.to_csv('binary_trial_1_results.csv', index=False)



    # MULTICLASS TRIAL 1
    target_type = 'multiclass'
    headers = ['CNN Model', 
               'Input Type', 
               'Target Type', 
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
    for model in MODELS:
        for input_type in INPUT_TYPES:
            current_df = pd.read_csv(os.path.join(path_to_results, f'{model}_{input_type}_{target_type}', 'testing_metrics.csv'))
            current_df['CNN Model'] = model
            current_df['Input Type'] = input_type
            current_df['Target Type'] = target_type
            current_df = current_df[headers]
            df = pd.concat([df, current_df], ignore_index=True)
    df.to_csv('multiclass_trial_1_results.csv', index=False)

