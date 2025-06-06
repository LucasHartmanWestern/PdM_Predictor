from datetime import datetime
import pandas as pd
import os


# Global Constants
MODELS = ['cgnet', 'unet']
ROI_MODES = ['rois', 'full']
TARGET_MODES = ['binary', 'multiclass']


def calculate_training_time(log_file_path):
    """
    Calculates the total training time of an ML model from a log file.

    Args:
        log_file_path (str): The path to the training log file.

    Returns:
        str: A string representing the total training time, or an error message.
    """
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
            return end_time - start_time
        else:
            return "Could not find start and/or end training timestamps in the log file."

    except FileNotFoundError:
        return f"Error: The file '{log_file_path}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"


def create_training_time_csv(path_to_results):
    """
    Creates a CSV file with the training time for each model, roi mode, and target mode.
    """
    df = pd.DataFrame(columns=['model', 'roi_mode', 'target_mode', 'training_time'])

    # get the training time for each model, roi mode, and target mode
    for model in MODELS:
        for roi_mode in ROI_MODES:
            for target_mode in TARGET_MODES:
                log_file_path = os.path.join(path_to_results, f'{model}_{roi_mode}_{target_mode}', 'training.log')
                training_time = calculate_training_time(log_file_path)
                print(f'{model}_{roi_mode}_{target_mode} training time: {training_time}')
                row_dict = {'model': model, 'roi_mode': roi_mode, 'target_mode': target_mode, 'training_time': str(training_time)}
                df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)

    # save the dataframe to a CSV file
    df.to_csv('training_time_plot.csv', index=False)


def plot_training_time(path_to_results):
    """
    Plots the training time for each model, roi mode, and target mode.
    """
    df = pd.read_csv('training_time_plot.csv')
    print(df)


if __name__ == "__main__":
    # Hyperparameters
    path_to_results = '/Users/nick_1/PycharmProjects/UWO Masters/MAR28_RESULTS'

    # create csv file with training time
    create_training_time_csv(path_to_results)

    # plot the training time

