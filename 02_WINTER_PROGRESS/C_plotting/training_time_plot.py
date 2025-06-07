from datetime import datetime
import pandas as pd
import os


# Global Constants
MODELS = ['cgnet', 'unet']
ROI_MODES = ['rois', 'full']
TARGET_MODES = ['binary', 'multiclass']


def calculate_training_time_trial1(log_file_path):
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        start_time_str = None
        end_time_str = None

        for line in lines:
            if 'lowest loss:' in line:
                end_epoch = int(line.split(' ')[-1])
                break

        for line in lines:
            if 'starting training...' in line:
                # Extract timestamp from the line like '2025-03-31 03:15:33.590674 starting training...'
                start_time_str = line.split(' ')[0] + ' ' + line.split(' ')[1]
            elif f'epoch {end_epoch}/25 metrics:' in line:
                # Extract timestamp from the line like '2025-03-31 03:41:16.651587 training complete.'
                end_time_str = line.split(' ')[0] + ' ' + line.split(' ')[1]
                break # Assuming 'training complete.' signifies the end and appears once

        if start_time_str and end_time_str:
            # Parse the timestamps
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
            end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S.%f')

            # Calculate the difference
            diff = end_time - start_time
            return str(diff)
        else:
            return "Could not find start and/or end training timestamps in the log file."

    except FileNotFoundError:
        return f"Error: The file '{log_file_path}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"
    

def calculate_training_time_other(log_file_path):
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'model training time:' in line:
                training_time_str = line.split(' ')[-1].strip()
                break

        return training_time_str

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return f"Error: The file '{log_file_path}' was not found."
    # except Exception as e:
    #     print(f"An error occurred: {e}")        
    #     return f"An error occurred: {e}"       



def create_training_time_csv(path_to_results, save_folder):
    df = pd.DataFrame(columns=['model', 'roi_mode', 'target_mode', 'trial', 'training_time'])

    # get the training time for each model, roi mode, and target mode
    for trial in range(1, 4):
        for model in MODELS:
            for roi_mode in ROI_MODES:
                for target_mode in TARGET_MODES:
                    log_file_path = os.path.join(path_to_results, f'trial_{trial}', f'{model}_{roi_mode}_{target_mode}', 'training.log')
                    if not os.path.exists(log_file_path):
                        training_time = 'N/A'
                    else:
                        if trial == 1:
                            training_time = calculate_training_time_trial1(log_file_path)
                        else:
                            training_time = calculate_training_time_other(log_file_path)

                    row_dict = {'model': model, 'roi_mode': roi_mode, 'target_mode': target_mode, 'trial': trial, 'training_time': training_time}
                    df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)

    # save the dataframe to a CSV file
    df.to_csv(os.path.join(save_folder, 'training_time_plot.csv'), index=False)


if __name__ == "__main__":
    # Hyperparameters
    path_to_results = '/Users/nick_1/PycharmProjects/UWO Masters/ICML_Jun7_Backup'
    save_folder = '/Users/nick_1/PycharmProjects/UWO Masters/PdM_Predictor/02_WINTER_PROGRESS/C_plotting/generated_csvs'

    # create csv file with training time
    create_training_time_csv(path_to_results, save_folder)
    print('done')

