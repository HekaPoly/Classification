#Create time series to feed into LSTM autoencoder

import numpy as np
import os
from sklearn.preprocessing import normalize
from pandas import read_csv

categories = ["HandOpen", "HandRest","ObjectGrip","PichGrip","WristPron","WristSupi","WristExten","WristFlex"]

#Create 3d array
def create_time_series(filepath, n_timesteps):

    time_series = []
    Y = []
    first_y = True 
    first_windows = True
    for category in categories:
        print(category)
        data = np.load(filepath + "/" + category + ".npy")
        data = normalize(data, axis=1)
        windows = split_into_windows(data, n_timesteps)
        y_category = create_Y_for_category(category, len(windows))
        if first_y:                                           #Not a good way to do it but hey it works
            Y = np.array(y_category)
            first_y = False
        else:
            Y = np.vstack((Y,y_category))

        if first_windows:                                     #udem
            time_series = windows
            first_windows = False
        else:
            time_series = np.vstack((time_series, windows))

    return time_series, Y



# Create windows of n_timesteps BY ADDING ENOUGH PADDING AND THEN SPLITTING THE ARRAY
def split_into_windows(X, n_timesteps):

    windows = []

    # Fill array until it can be split into windows of 10 timesteps (elements)
    while len(X) % n_timesteps != 0:
        X = np.row_stack((X, X[ len(X) -1 ]))

    windows = np.array_split(X, (len(X) / n_timesteps), axis=0)
    return np.array(windows)


#Create y data for one category(movement)
def create_Y_for_category(category, windows_length):

    index_label = categories.index(category)
    Y = np.full((windows_length, 1), index_label)
    return Y

# Create sliding windows of n_timesteps (this method is not used)
def create_sliding_windows(data, n_timesteps):

    windows = []
    for i in range(data.shape[0] - n_timesteps):
        windows.append(data[i: i + n_timesteps])

    return np.array(windows)