import numpy as np
import tensorflow as tf

from sklearn.preprocessing import normalize


def create_angles_window_average(angle_data, average_window, n_timesteps):
    windows = []
    for i in range(angle_data.shape[0] - n_timesteps + 1):
        windows.append(
            np.sum(angle_data[n_timesteps + i - 1:i + average_window + n_timesteps - 1], axis=0) / average_window)
    return np.array(windows)


def create_time_serie(emg_data, angle_data, n_timesteps, average_window):
    for i in range(emg_data.shape[0]):
        windows = create_sliding_windows(emg_data[i], n_timesteps)
        angles = angle_data[i][n_timesteps - 1:]
        # angles = create_angles_window_average(angle_data[i], average_window, n_timesteps)
        # angles = np.delete(angles, [(i+5)%N_ANGLES for i in range(N_ANGLES_TO_DELETE)], axis=1)
        # print(angle)
        # print(angle_data[0:5])

        # print("EMG :", emg_data[i].shape)
        # print("Windows :", windows.shape)
        # print("Angles :", angles.shape)
        # print("\n")
        if 0 < i:
            # windows = np.asarray([windows])
            # print(i, windows.shape, time_series.shape)
            emg_series = np.vstack((emg_series, windows))
            angle_series = np.vstack((angle_series, angles))
        else:
            emg_series = windows
            angle_series = angles

    print("EMG shape :", emg_series.shape, "Angles shape :", angle_series.shape)
    return emg_series, angle_series


# Create sliding windows of n_timesteps (data augmentation technique)
def create_sliding_windows(data, n_timesteps):
    windows = []
    for i in range(data.shape[0] - n_timesteps + 1):
        windows.append(data[i: i + n_timesteps])
    return np.array(windows)


TOTAL_SEQUENCES = 572
N_PHASES = 3
N_ANGLES = 18
N_ANGLES_TO_DELETE = 0
N_ELECTRODES = 7
