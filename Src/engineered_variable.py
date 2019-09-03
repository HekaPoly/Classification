import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import glob
import os
import re
import classes
from sklearn.preprocessing import normalize

kWindowSize = 0.1
kSamplingRate = 2000
kAcquisitionTime = 45
kNumFeatures = 7
kNelectrodes = 7

outputs = [
    "shoulder_abduction_",
    "shoulder_abduction_",
    "shoulder_abduction_",
    "rest_post_abduction_",
    "rest_post_abduction_",
    "shoulder_adduction_",
    "shoulder_adduction_",
    "shoulder_adduction_",
    "rest_low_arm_",
    "rest_low_arm_",
    "rest_low_arm_",
    "shoulder_flexion_",
    "shoulder_flexion_",
    "rest_post_shoulder_flexion_",
    "rest_post_shoulder_flexion_",
    "rest_post_shoulder_flexion_",
    "shoulder_extension_",
    "shoulder_extension_",
    "shoulder_extension_",
    "rest_low_arm_",
    "rest_low_arm_",
    "elbow_flexion_",
    "elbow_flexion_",
    "elbow_flexion_",
    "rest_post_elbow_flexion_",
    "rest_post_elbow_flexion_",
    "elbow_extension_",
    "elbow_extension_",
    "elbow_extension_",
    "rest_low_arm_",
    "rest_low_arm_",
    "elbow_flexion_",
    "elbow_flexion_",
    "rest_post_elbow_flexion_",
    "rest_post_elbow_flexion_",
    "rest_post_elbow_flexion_",
    "elbow_extension_",
    "elbow_extension_",
    "elbow_flexion_",
    "elbow_flexion_",
    "elbow_flexion_",
    "rest_post_elbow_flexion_",
    "rest_post_elbow_flexion_",
    "rest_post_elbow_flexion_",
    "elbow_extension_",
    "elbow_extension_",
]

def remove_dc_component(data):
    return signal.detrend(data)


# std
def compute_variance(data):
    return np.var(data, dtype=np.float64)


def ssc_peaks(data):
    peaks = signal.find_peaks(data)
    return peaks[0].shape[0]


def zero_crossings(data):
    return (np.where(np.diff(np.sign(data)))[0]).shape[0]


def mean_absolute_value(data):
    rectified = np.absolute(data)
    return np.mean(rectified)


def rms(data):
    return np.sqrt(np.mean(data ** 2))


def max(data):
    return np.max(data)


def extract_class_name(name):
    digit_index = re.search("\d", name)
    return name[0:digit_index.start()]


def extract_features(window):
    variance = compute_variance(window)
    return [
        rms(window),
        max(window),
        np.sqrt(variance),
        mean_absolute_value(window),
        zero_crossings(window),
        ssc_peaks(window),
        variance
    ]


if __name__ == "__main__":

    slice_size = int(kWindowSize * kSamplingRate)
    n_slices = int(kAcquisitionTime * kSamplingRate / slice_size)
    # files = glob.glob('..\\test_data\\*.npy')
    features = np.zeros([n_slices, kNelectrodes * kNumFeatures * 3])
    y = np.zeros([n_slices, classes.get_num_classes()])
    # for file_number, file in enumerate(files):
    # filename = os.path.basename(file)
    file = "C:\\Users\\hazbo\\Documents\\git\\Metis-Exoskeleton\\Acquisition\\ringBufferDMA\\" \
           "hostReception\\freestyle_45s_2000Hz_2.npy"
    # current_class_name = extract_class_name(filename)
    file_data = np.load(file)
    file_data = normalize(file_data, axis=1)
    for i in range(n_slices):
        for j in range(kNelectrodes):
            window = file_data[j, i * slice_size : i * slice_size + slice_size]
            window = remove_dc_component(window)
            features[i, kNumFeatures * j:kNumFeatures * j + kNumFeatures] = extract_features(window)
        current_class_name = outputs[int(i*slice_size/kSamplingRate)]
        y[i, classes.from_string(current_class_name)] = 1

    features[0, kNelectrodes * kNumFeatures:2*kNelectrodes * kNumFeatures] = features[0, :kNelectrodes * kNumFeatures]
    features[0, 2*kNelectrodes * kNumFeatures:] = features[0, :kNelectrodes * kNumFeatures]
    features[1, kNelectrodes * kNumFeatures:2*kNelectrodes * kNumFeatures] = features[0, :kNelectrodes * kNumFeatures]
    features[1, 2*kNelectrodes * kNumFeatures:] = features[0, :kNelectrodes * kNumFeatures]
    for i in range(2, features.shape[0]):
        features[i, kNelectrodes * kNumFeatures:2*kNelectrodes * kNumFeatures] = features[i-1, :kNelectrodes * kNumFeatures]
        features[i, 2*kNelectrodes * kNumFeatures:] = features[i-2, :kNelectrodes * kNumFeatures]
        # features[i, kNelectrodes * kNumFeatures:] = features[i, :kNelectrodes * kNumFeatures] - features[i-1, :kNelectrodes * kNumFeatures]

    np.save("features_demo2", features)
    np.save("y_demo", y)
