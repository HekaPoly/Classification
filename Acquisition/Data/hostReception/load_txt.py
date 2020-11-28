import numpy as np
import matplotlib.pyplot as plt


data = np.load("freestyle_30s_2000Hz_3.npy")
plt.subplot(241)
plt.plot(data[0])
plt.subplot(242)
plt.plot(data[1])
plt.subplot(243)
plt.plot(data[2])
plt.subplot(244)
plt.plot(data[3])
plt.subplot(245)
plt.plot(data[4])
plt.subplot(246)
plt.plot(data[5])
plt.subplot(247)
plt.plot(data[6])
plt.show()
