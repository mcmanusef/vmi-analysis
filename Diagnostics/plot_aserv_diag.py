import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Qt5Agg')

filename = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20231207\o2_s_2250.cv4"
with h5py.File(filename, "r") as file:
    file_dict={}
    for k in file.keys():
        file_dict[k]=file[k][()]



# Plot the data
plt.figure(figsize=(10, 8))

# Plot pulses data
ax=plt.subplot(111)
plt.plot(np.linspace(0,1,num=len(file_dict['pulses'])),file_dict["pulses"],c='k')

plt.plot(np.linspace(0,1,num=len(file_dict['itof'])),file_dict['itof'],c='g')
# plt.title("ITOF")
plt.plot(np.linspace(0,1,num=len(file_dict['etof'])),file_dict['etof'],c='r')
# plt.title("ETOF")
plt.plot(np.linspace(0,1,num=len(file_dict['toa'])),file_dict['toa'],c='b')
# plt.title("TOA")


plt.tight_layout()
plt.show()

# plt.plot(np.diff(file_dict['toa']/25))
#%%
