import numpy as np
import scipy.signal as signal
from scipy.fft import *
import matplotlib.pyplot as plt

import data_loader


def get_frequencie(emg_data):
    result = []
    for i in range(emg_data.shape[0]):
        print(emg_data[i])
        result.append(fft(emg_data[i]))

    print(result)
    fft_result = np.abs(np.array(result))
    plt.plot(np.swapaxes(fft_result,0,1)[0])
    plt.show()

def low_pass_filter(data):
    result = signal.butter(2, )

    return result


#folderpath = r"..\..\Acquisition\Data\Dataset_avec_angles_tester"
#emg_data, angle_data = data_loader.load_data(folderpath)
#get_frequencie(emg_data)

data = np.load(r"C:\Users\leona\Documents\Metis\Classification\Acquisition\Data\7_electrodes_Philippe\regroupement_des donnes_par_categorie\test_data_with_freestyle\shoulder_extension_3s_2000Hz.npy")
print(data.shape)
get_frequencie(data)
plt.plot(data)
plt.show()
