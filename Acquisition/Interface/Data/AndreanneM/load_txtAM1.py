import numpy as np
import matplotlib.pyplot as plt

data=np.load("Flexioncoude_3s_4000Hz_3.npy")
print(data)


plt.plot(data[0])
plt.title("Electrode 1")
plt.show()





