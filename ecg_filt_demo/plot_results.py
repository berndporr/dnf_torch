import numpy as np
import matplotlib.pyplot as plt
import sys

fs = 1000

data = np.loadtxt("ecg_filtered.dat")

filtered = data[:,0]
orig = data[:,1]
remover = data[:,2]

t = np.linspace(0,1/fs*len(orig),len(orig))

fig=plt.figure()
ax1 = plt.subplot(311)
ax2 = plt.subplot(312, sharex = ax1)
ax3 = plt.subplot(313, sharex = ax1)
ax1.plot(t,orig)
ax1.set_title("Original signal")
ax1.set_ylabel("Orig/V")
ax2.plot(t,remover)
ax2.set_title("Remover")
ax2.set_ylabel("Rem/V")
ax3.plot(t,filtered)
ax3.set_title("Filtered")
ax3.set_ylabel("Filt/V")
plt.xlabel("time/sec")
plt.show()
