import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def get_frequencie(emg_data):
    result = np.fft.fft(emg_data, axis=0)
    print(result)
    plt.plot(result)
    plt.show()
    return result

def apply_butter_filter(emg_data, freq):
    butter_filter = signal.butter(2, freq, analog=True, output='sos')
    return signal.sosfilt(butter_filter, emg_data)

