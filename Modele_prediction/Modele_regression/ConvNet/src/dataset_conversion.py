import numpy as np
from convnet import ModelConv
from scipy.io import loadmat
import os
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

if __name__=="__main__":
    if os.name == 'nt': # Windows
        filepath = "..\\..\\Acquisition\\Data\\Dataset_avec_angles_tester"
    else: # Linux/Mac
        filepath = "../../Acquisition/Data/Dataset_avec_angles_tester"

    data = loadmat(filepath + "/KIN_MUS_UJI.mat")
    
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