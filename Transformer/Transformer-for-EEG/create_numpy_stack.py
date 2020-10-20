import numpy as np
data1 = np.load('elbow_extension_3s_2000Hz_1.npy')
data2 = np.load('rest_post_shoulder_flexion_3s_2000Hz_4.npy')
data3 = np.load('shoulder_abduction_3s_2000Hz_1.npy')

out = np.stack((data1,data2,data3))
np.save('stack.npy', out)
print('a')