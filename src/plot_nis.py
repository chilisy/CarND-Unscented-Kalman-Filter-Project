import numpy as np
import matplotlib.pyplot as plt

nis_data = np.genfromtxt('../XcodeProj/Debug/NIS.csv', delimiter=',')

nis_radar = nis_data[:, 0]
nis_laser = nis_data[:, 1]

ax = plt.subplot(211)
ax.set_title("NIS Radar")
plt.plot(nis_radar)
plt.plot([0, 244], [7.815, 7.815])

ax = plt.subplot(212)
ax.set_title("NIS Laser")
plt.plot(nis_laser)
plt.plot([0, 244], [5.991, 5.991])
plt.show()