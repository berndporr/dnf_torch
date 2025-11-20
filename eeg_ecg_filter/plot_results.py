import numpy as np
import matplotlib.pyplot as plt
import sys

fs = 1000

data = np.loadtxt("eeg_filtered.dat")

fnn = data[:,0]
eeg = data[:,1]
ecg = data[:,2]

fig=plt.figure()
ax1 = plt.subplot(311)
ax2 = plt.subplot(312, sharex = ax1)
ax3 = plt.subplot(313, sharex = ax1)
ax1.plot(eeg)
ax2.plot(ecg)
ax3.plot(fnn)
plt.xlabel("time")
plt.ylabel("volt")
plt.show()    

