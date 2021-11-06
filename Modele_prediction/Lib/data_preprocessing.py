import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import data_loader


def get_frequencie(emg_data):
    result = np.fft.fft(emg_data, axis=0)
    print(result)
    plt.plot(result)
    plt.show()
    return result

def apply_butter_filter(emg_data, freq):
    butter_filter = signal.butter(2, freq, analog=True, output='sos')
    #result = np.fft.ifft(butter_filter * fft_freq[:,1])
    return signal.sosfilt(butter_filter, emg_data)


#folderpath = r"..\..\Acquisition\Data\Dataset_avec_angles_tester"
#emg_data, angle_data = data_loader.load_data(folderpath)
#get_frequencie(emg_data)

data = np.load(r"C:\Users\leona\Documents\Metis\Classification\Acquisition\Data\7_electrodes_Philippe\regroupement_des donnes_par_categorie\test_data_with_freestyle\shoulder_extension_3s_2000Hz.npy")
print(data.shape)
result = apply_butter_filter(data, 10000)
get_frequencie(result)
plt.plot(result)
plt.show()
