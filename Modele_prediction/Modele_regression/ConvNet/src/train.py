from convnet import ModelConv
import os
from os import path
from sklearn.model_selection import train_test_split

from Lib.data_preprocessing import *
from Lib.sliding_window import *

TOTAL_SEQUENCES = 572
N_PHASES = 3
N_ANGLES = 18
N_ANGLES_TO_DELETE = 0
N_ELECTRODES = 7

if __name__=="__main__":
    if os.name == 'nt': # Windows
        filepath = "..\\..\\..\\..\\Acquisition\\Data\\Dataset_avec_angles_tester"
    else: # Linux/Mac
        filepath = "../../../../Acquisition/Data/Dataset_avec_angles_tester"

    with open(filepath + '/dataset_emg_angles.npz', 'rb') as file:
        data = np.load(file, allow_pickle=True)
        emg_data = data["emg_data"]
        angle_data = data["angle_data"]

    print("EMG dataset size:", emg_data.shape)
    print("Angle size:", angle_data.shape)

    # Data normalization
    angle_data = angle_data
    x_train, x_test, y_train, y_test = train_test_split(
        emg_data, angle_data, test_size=0.10, random_state=40
    )
    
    n_timesteps = 70
    if not path.exists('dataset_processed.npy'):
        x_train, y_train = create_time_serie(x_train, y_train, n_timesteps)
        x_test, y_test = create_time_serie(x_test, y_test, n_timesteps)

        test = normalize(x_train[0])

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

    epochs = 11
    batch_size = 500

    model = ModelConv(N_ANGLES-N_ANGLES_TO_DELETE, N_ELECTRODES, n_timesteps)
    print("x train shape :", x_train.shape, "y train shape :", y_train.shape)
    print("x test shape :", x_test.shape, "y test shape :", y_test.shape)
    model.train(x_train, y_train, x_test, y_test, epochs, batch_size)
    model.evaluate(x_test, y_test)

    model.save("conv_angle_v1_sw.h5")

    print(model.predict(x_test).shape)
    print(y_test.shape)

    y_hat = model.predict(x_test)
    taille = len(y_hat)

    angles_hat = []
    angles_test = []
    angles_MSE = []

    for j in range(18):
        for i in range(10000):
            angles_hat.append(y_hat[i][j])
            angles_test.append(y_test[i][j])
            angles_MSE.append((y_hat[i][j] - y_test[i][j])**2)
        #print(np.shape(angles_hat))
        #print(np.shape(angles_test))
        fig, (ax1,ax2) = plt.subplots(2,1)
        ax1.plot(angles_hat, 'r')
        ax1.plot(angles_test, 'b')
        ax2.plot(angles_MSE, 'g')
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Squared error")
        ax2.set_title("MSE = " + str(np.mean(angles_MSE)))
        plt.show()
        angles_hat = []
        angles_test = []
        angles_MSE = []