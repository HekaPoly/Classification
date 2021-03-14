import numpy as np
from convnet import ModelConv
from scipy.io import loadmat
import os
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

N_PHASES = 3

def display_angle_sequence(angle_data, sequence):
    for j in range(N_PHASES):
        for angle in range(18):
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
        emg_sequence = []
        angle_sequence = []
        for i in range(N_PHASES):
            emg_sequence += emg_data[N_PHASES * sequence + i].tolist()
            angle_sequence += angle_data[N_PHASES * sequence + i].tolist()
        emg_array.append(emg_sequence)
        angle_array.append(angle_sequence)

    return np.array(emg_array), np.array(angle_array)

def saveas_npy(dataset : dict):
    pass

TOTAL_SEQUENCES = 512

if __name__=="__main__":
    if os.name == 'nt': # Windows
        filepath = "..\\..\\Acquisition\\Data\\Dataset_avec_angles_tester"
    else: # Linux/Mac
        filepath = "../../Acquisition/Data/Dataset_avec_angles_tester"

    data = loadmat(filepath + "/KIN_MUS_UJI.mat")
    #graph electrodes
    electrodes_values = []
    # for electrode in range(len(data['EMG_KIN_v4']['EMG_data'][0][0][0])):
    #     electrodes_values.append([])
    #     for i in range(len(data['EMG_KIN_v4']['EMG_data'][0][0])):
    #         electrodes_values[-1].append(data['EMG_KIN_v4']['EMG_data'][0][0][i][electrode]) 
    
    # for electrode in electrodes_values:
    #     plt.plot(electrode)
    # plt.show()

    angle_values = [[] for i in range(18)]
    angle_data = data['EMG_KIN_v4']['Kinematic_data'][0]
    emg_data = data['EMG_KIN_v4']['EMG_data'][0]
    # phase_data = data['EMG_KIN_v4']['Phase'][0]

    #display_angle_sequence(angle_data, 5)
    
    emg_data, angle_data = generate_np_dataset(emg_data, angle_data)
    print(emg_data.shape)
    print(angle_data.shape)
    print(angle_data[0])
    # X_data, Y_data = extract_features(filepath)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_data, Y_data, stratify = Y_data,test_size=0.20, random_state=42
    # ) [[asdf]]