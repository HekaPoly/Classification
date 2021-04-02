import numpy as np
from convnet import ModelConv
from scipy.io import loadmat
import os
from os import path
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Label one category
def label_category(category, windows_length):
    indexLabel = categories.index(category)
    Y = np.full((windows_length, 1), indexLabel)
    return Y

# #Create 3d array for lstmae training
# def create_time_series(filepath, n_timesteps):
#     time_series = []
#     Y = []
#     first_y = True 
#     first_windows = True
#     for category in categories:
#         print(category)
#         data = np.load(filepath + "/" + category + ".npy")
#         data = normalize(data, axis=1)
#         windows = create_sliding_windows(data, n_timesteps)
#         y_category = label_category(category, len(windows))
#         if first_y:                                        
#             Y = np.array(y_category)
#             first_y = False
#         else:
#             Y = np.vstack((Y,y_category))

#         if first_windows:                                   
#             time_series = windows
#             first_windows = False
#         else:
#             time_series = np.vstack((time_series, windows))

#     return time_series, Y

def create_angles(angle_data, n_timesteps):
    pass

def create_time_serie(emg_data, angle_data, n_timesteps):
    for i in range(emg_data.shape[0]):
        windows = create_sliding_windows(emg_data[i], n_timesteps)
        angles = angle_data[i][n_timesteps-1:]
        # print("EMG :", emg_data[i].shape)
        # print("Windows :", windows.shape)
        # print("Angles :", angles.shape)
        # print("\n")
        if 0 < i:
            #windows = np.asarray([windows])
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
N_ELECTRODES = 7

if __name__=="__main__":
    if os.name == 'nt': # Windows
        filepath = "..\\..\\Acquisition\\Data\\Dataset_avec_angles_tester"
    else: # Linux/Mac
        filepath = "../../Acquisition/Data/Dataset_avec_angles_tester"

    with open(filepath + '/dataset_emg_angles.npz', 'rb') as file:
        data = np.load(file, allow_pickle=True)
        emg_data = data["emg_data"]
        angle_data = data["angle_data"]

    print("EMG dataset size:", emg_data.shape)
    print("Angle size:", angle_data.shape)

    # Data normalization
    angle_data = angle_data
    # x_train = emg_data
    x_train, x_test, y_train, y_test = train_test_split(
        emg_data, angle_data, test_size=0.10, random_state=40
    )
    
    # print(x_train.shape, x_test.shape)
    # print(y_train.shape, y_test.shape)
    # print(x_train[0].shape)
    # print(x_train[151].shape)
    # x_train = create_sliding_windows(x_train[0], 30)
    # x_test = create_sliding_windows(x_test, 30)
    # y_train = create_sliding_windows(y_train[0], 30)
    # y_test = create_sliding_windows(y_test[0], 30)
    
    # print(sx_train.shape)
    
    n_timesteps = 30
    if not path.exists('dataset_processed.npy'):
        x_train, y_train = create_time_serie(x_train, y_train, n_timesteps)
        x_test, y_test = create_time_serie(x_test, y_test, n_timesteps)

        with open('dataset_processed.npy', 'wb') as f:
            np.save(f, x_train)
            np.save(f, y_train)
            np.save(f, x_test)
            np.save(f, y_test)
            
    else:
        with open('dataset_processed.npy', 'rb') as f:
            x_train = np.load(f, allow_pickle=True)
            y_train = np.load(f, allow_pickle=True)
            x_test = np.load(f, allow_pickle=True)
            y_test = np.load(f, allow_pickle=True)

    batch = 30
    batch_size = 300

    model = ModelConv(N_ANGLES, N_ELECTRODES, n_timesteps)
    print("x train shape :", x_train.shape, "y train shape :", y_train.shape)
    print("x test shape :", x_test.shape, "y test shape :", y_test.shape)
    model.train(x_train, y_train, x_test, y_test, batch, batch_size)


    # l-sl + 1

    # [1, 2, 3, 4]
    # [[1,2],[2,3],[3,4]]
    # [[1,2,3], [2,3,4]]

