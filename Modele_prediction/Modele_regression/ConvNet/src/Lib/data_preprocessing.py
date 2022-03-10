import math

import numpy as np

def normalize(emg_data):
    mean = np.sum(emg_data) / emg_data.shape[0]
    variance = []
    for i in range(emg_data.shape[0]):
        variance.append(np.power(emg_data[i] - mean, 2))
    variance = np.sum(np.array(variance)) / emg_data.shape[0]

    x = []
    for i in range(emg_data.shape[0]):
        x.append((emg_data[i] - mean) / np.sqrt(variance))

    return np.array(x)