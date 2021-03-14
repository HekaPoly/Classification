from convnet import ModelConv
from scipy.io import loadmat
import os
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Extract features
def extract_features(filepath):
    n_timesteps = 30
    X, Y = create_time_series(filepath, n_timesteps)
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)
    Y_categorical = to_categorical(Y)

    return X, Y_categorical

#Create 3d array
def create_time_series(filepath, n_timesteps):
    time_series = []
    Y = []
    first_y = True 
    first_windows = True

    data = loadmat(filepath + "/KIN_MUS_UJI.mat")
    data = normalize(data, axis=1)
    windows = create_sliding_windows(data, n_timesteps)
    if first_windows:                                   
        time_series = windows
        first_windows = False
    else:
        time_series = np.vstack((time_series, windows))
    
    return time_series, Y

    # Create sliding windows of n_timesteps (data augmentation technique)
def create_sliding_windows(data, n_timesteps):
    windows = []
    for i in range(data.shape[0] - n_timesteps):
        windows.append(data[i: i + n_timesteps])

    return np.array(windows)

TOTAL_SEQUENCES = 512

if __name__=="__main__":
    if os.name == 'nt': # Windows
        filepath = "..\\..\\Acquisition\\Data\\Dataset_avec_angles_tester"
    else: # Linux/Mac
        filepath = "../../Acquisition/Data/Dataset_avec_angles_tester"

    data = loadmat(filepath + "/KIN_MUS_UJI.mat")
    #graph electrodes
    electrodes_values = []
    # electrod data array(list) [data][wrapper][sequence?][point in seq][electrode_data]
    # for electrode in range(len(data['EMG_KIN_v4']['EMG_data'][0][0][0])):
    #     electrodes_values.append([])
    #     for i in range(len(data['EMG_KIN_v4']['EMG_data'][0][0])):
    #         electrodes_values[-1].append(data['EMG_KIN_v4']['EMG_data'][0][0][i][electrode]) 
    
    # for electrode in electrodes_values:
    #     plt.plot(electrode)
    # plt.show()

    angle_values = [[] for i in range(18)]
    angle_data = data['EMG_KIN_v4']['Kinematic_data'][0]
    phase_data = data['EMG_KIN_v4']['Phase'][0]
    q = 3
    
    #for j in range(2): # le nombre de phase
    
    sequence = 5
    for j in range(3):
        for angle in range(18):
            for i in range(len(angle_data[3*sequence + j])):
                print(i, j, angle)
                angle_values[angle].append(angle_data[3*sequence + j][i][angle])
    
    for angle in angle_values:
        plt.plot(angle)
    plt.show()
    # print(len(electrodes_values))
    # print(data['EMG_KIN_v4']['Kinematic_data'][0][0]) # maybe angles array of list
    # print(data['EMG_KIN_v4']['Subject'][0][0])
    # for seq in data['EMG_KIN_v4']['EMG_data']:
    #     print(seq)

    # X_data, Y_data = extract_features(filepath)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_data, Y_data, stratify = Y_data,test_size=0.20, random_state=42
    # ) [[asdf]]

