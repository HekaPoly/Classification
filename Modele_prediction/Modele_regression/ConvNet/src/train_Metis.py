import numpy as np
from convnet import ModelConv
from scipy.io import loadmat
import os
from os import path
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle
import matplotlib.pyplot as plt

"""
N_PHASES = 3
N_ANGLES = 18
N_ELECTRODES = 7

def display_angle_sequence(angle_data, sequence):
    for j in range(N_PHASES):
        for angle in range(N_ANGLES):
            for i in range(len(angle_data[N_PHASES * sequence + j])):
                angle_values[angle].append(angle_data[N_PHASES * sequence + j][i][angle])
    
    for angle in angle_values:
        plt.plot(angle)
    plt.show()

# electrod data array(list) [data][wrapper][sequence?][point in seq][electrode_data]
def generate_np_dataset(emg_data, angle_data):
    emg_array = []
    angle_array = []
    
    for sequence in range(int(len(emg_data)/N_PHASES)):
        for i in range(N_PHASES):
            emg_phase = emg_data[N_PHASES * sequence + i]
            angle_phase = angle_data[N_PHASES * sequence + i]
            
            if emg_phase.shape[1] == N_ELECTRODES and angle_phase.shape[1] == N_ANGLES:
                if i > 0:
                    emg_seq = np.concatenate((emg_phase, emg_seq), axis=0)
                    angle_seq = np.concatenate((angle_phase, angle_seq), axis=0)
                else:
                    emg_seq = emg_phase
                    angle_seq = angle_phase
        
        emg_array.append(emg_seq)
        angle_array.append(angle_seq)

    return np.asarray(emg_array), np.asarray(angle_array)
"""
# Create sliding windows of n_timesteps (data augmentation technique)

def create_sliding_windows(x, n_timesteps):
    windows = []
    number_of_pts = x.shape
    for i in range(number_of_pts - n_timesteps + 1):
        windows.append(x[i: i + n_timesteps])
    return np.array(windows)

def create_time_serie(emg_data, angle_data, n_timesteps, average_window):
    nb_of_pts = max(emg_data.shape[0],emg_data.shape[1]) 
    for i in range(nb_of_pts):
        windows = create_sliding_windows(emg_data[i], n_timesteps)
        angles = create_sliding_windows(angle_data[i],n_timesteps) #angle_data[i:i + n_timesteps]
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

if __name__=="__main__":
    if os.name == 'nt': # Windows
        filepath = "..\\..\\Acquisition\\Data\\7_electrodes_Philippe\\regroupement_des donnes_par_categorie\\test_data"
    else: # Linux/Mac
        filepath = "../../Acquisition/Data/7_electrodes_Philippe/regroupement_des donnes_par_categorie/test_data"

    data_files = [] #loaded files that end in .npy
    for dir_file in os.scandir(filepath):
        if(dir_file.name.endswith(".npy")):
            data_files.append(np.load(dir_file))
    print(data_files)
    

    n_timesteps = 30
    average_window = 10

    meta_x_train = []
    meta_y_train = []
    meta_x_test = []
    meta_y_test = []
    meta_arrays = [meta_x_train,meta_y_train,meta_x_test,meta_y_test]
    for data_file in data_files:
        #x_train = emg_data
        #data_files.transpose
        emg_data = data_file[0:-1,0:6]
        angle_data = data_file[0:-1,6]
        print(emg_data.shape)
        print(angle_data.shape)
        x_train, x_test, y_train, y_test = train_test_split(
        emg_data, angle_data, test_size=0.10, random_state=40)
        x_train_windows, y_train_windows = create_time_serie(x_train, y_train, n_timesteps, average_window)
        x_test_windows, y_test_windows = create_time_serie(x_test, y_test, n_timesteps, average_window)
        
        x_train_windows, y_train_windows = sk_shuffle(x_train_windows, y_train_windows) #shuffling the train and validation sets
        x_test_windows, y_test_windows = sk_shuffle(x_test_windows, y_test_windows)

        meta_windows = [x_train_windows,y_train_windows,x_test_windows,y_test_windows]
        for i in range(len(meta_windows)): 
            meta_arrays[i].append(meta_windows[i])
    """
    if not path.exists('dataset_processed.npy'):
        x_train, y_train = create_time_serie(x_train, y_train, n_timesteps, average_window)
        x_test, y_test = create_time_serie(x_test, y_test, n_timesteps, average_window)

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
    """
    epochs = 11
    batch_size = 500

    """
    angle_data = data['EMG_KIN_v4']['Kinematic_data'][0]
    emg_data = data['EMG_KIN_v4']['EMG_data'][0]
    
    emg_data, angle_data = generate_np_dataset(emg_data, angle_data)
    print(emg_data.shape)
    print(angle_data.shape)
    with open(filepath + '/dataset_emg_angles.npz', 'wb') as file:
        np.savez_compressed(file, emg_data=emg_data, angle_data=angle_data)

    # Loading file exemple
    with open(filepath + '/dataset_emg_angles.npz', 'rb') as file:
        data = np.load(file, allow_pickle=True)
        print(data.files)
        print(data['emg_data'].shape)
        print(data['angle_data'].shape)
    """