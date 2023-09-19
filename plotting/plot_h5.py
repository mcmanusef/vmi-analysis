import h5py
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('image', cmap='jet')
matplotlib.use('Qt5Agg')
path=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20230803\optimized_lens_18W_p.h5"
plt.figure()
num=1000000
with h5py.File(path,mode='r') as f:
    print(len(f['x']))
    print(len(f['y']))
    x=f['x'][0:num]
    y=f['y'][0:num]
plt.figure()
plt.hist2d(x,y,bins=256,range=((0,256),(0,256)),norm="log")