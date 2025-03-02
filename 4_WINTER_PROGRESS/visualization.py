import matplotlib.pyplot as plt
import os


def create_metric_plots(metrics_dict, save_path):
    for key, val in metrics_dict.items():
        if key != "Epoch":
            plt.clf()
            plt.plot(val["Train"])
            plt.plot(val["Val"])
            plt.title(f"Training {key}")
            plt.ylabel(key)
            plt.xlabel("Epoch")
            plt.legend(['Train', 'Val'])
            plt.savefig(os.path.join(save_path, f"{key.lower().split(' ')[0]}_plot.png"))


# def create_metric_plots(csv_filepath):
#     for key, val in metrics_dict.items():
#         if key != "Epoch":
#             plt.clf()
#             plt.plot(val["Train"])
#             plt.plot(val["Val"])
#             plt.title(f"Training {key}")
#             plt.ylabel(key)
#             plt.xlabel("Epoch")
#             plt.legend(['Train', 'Val'])
#             plt.savefig(os.path.join(save_path, f"{key.lower().split(' ')[0]}_plot.png"))
