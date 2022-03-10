import numpy as np


def load_data(folderpath):
    with open(folderpath + '/dataset_emg_angles.npz', 'rb') as file:
        data = np.load(file, allow_pickle=True)
        emg_data = data["emg_data"]
        angle_data = data["angle_data"]

    return emg_data, angle_data
