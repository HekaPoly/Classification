import slidingWindow
from sklearn.model_selection import train_test_split
from model import Model
import numpy as np
from os import path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filepath = "..\..\..\Acquisition\Data\Dataset_avec_angles_tester"
    saved_model_path = "model"

    with open(filepath + '/dataset_emg_angles.npz', 'rb') as file:
        data = np.load(file, allow_pickle=True)
        emg_data = data["emg_data"]
        angle_data = data["angle_data"]

    print("EMG dataset size:", emg_data.shape)
    print("Angle size:", angle_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(
            emg_data, angle_data, test_size=0.10, random_state=40
        )

    n_timesteps = 30
    average_window = 10

    if not path.exists('dataset_processed.npy'):
        x_train, y_train = slidingWindow.create_time_serie(x_train, y_train, n_timesteps, average_window)
        x_test, y_test = slidingWindow.create_time_serie(x_test, y_test, n_timesteps, average_window)

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

    # parameter for the model
    n_layers = 2
    d_model = x_train.shape[2]
    n_heads = 7
    units = 1024
    dropout = 0.1
    time_steps = x_train.shape[1]
    epochs = 10
    batch_size = 500
    n_angle = 18

    transformer = Model(time_steps, n_layers, units, d_model, n_heads, dropout, n_angle)
    transformer.load(saved_model_path)

    y_hat = transformer.predict(x_test)

    angles_hat = []
    angles_test = []
    angles_MSE = []

    for j in range(18):
        for i in range(len(y_test)):
            angles_hat.append(y_hat[i][j])
            angles_test.append(y_test[i][j])
        plt.plot(angles_hat, color='r', label="Angle predit")
        plt.plot(angles_test, color='g', label="Angle original")
        plt.title("Prediction d'angle")
        plt.ylabel("Angle en degrés")
        plt.xlabel("Tenseur")
        plt.legend()
        plt.show()
        angles_hat.clear()
        angles_test.clear()