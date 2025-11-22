import numpy as np
import matplotlib.pyplot as plt
import sys

fs = 1000

data = np.loadtxt("ecg_filtered.dat")

d0 = data[:,0]
d1 = data[:,1]
d2 = data[:,2]

fig=plt.figure()
ax1 = plt.subplot(311)
ax2 = plt.subplot(312, sharex = ax1)
ax3 = plt.subplot(313, sharex = ax1)
ax1.plot(d0)
ax2.plot(d1)
ax3.plot(d2)
plt.xlabel("time")
plt.ylabel("volt")
plt.show()    
