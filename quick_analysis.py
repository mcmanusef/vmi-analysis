from vmi_analysis import coincidence_v4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
fname=r"D:\Data\11_25\c2h4_p_5W.cv4"

data = coincidence_v4.load_file_coin(fname)
# data = coincidence_v4.load_file(r"D:\Data\c2h4_p_5W.cv4")
#%%
x,y,t,etof,itof=data
etof+=0.26*np.random.random_sample(len(etof))
#%%
n=1024//2
gamma=0.3
cmap='viridis'

fig = plt.figure(figsize=(8, 10))
plt.suptitle(fname)
plt.subplot(311)
plt.gca().set_axisbelow(True)
plt.title('Time of Flight Spectra (Log Scale, Density)')
plt.semilogy()
plt.xlabel('Time (ns)')
plt.ylabel('Count')
plt.grid()

plt.hist(t, bins=3000, range=(0, 20000), color='b', alpha=0.7, label='ToA')
plt.hist(etof, bins=3000, range=(0, 20000), color='r', alpha=0.7, label='e-ToF')
plt.hist(itof, bins=3000, range=(0, 20000), color='g', alpha=0.7, label='i-ToF')
plt.legend()

mask1 = (t > 500) & (t<700)
mask2 = (etof > 495) & (etof<497)
# mask3 = (itof > 10500) & (itof<13000)
mask = mask1 & mask2

xc=x-133.2
yc=y-131.7
theta = 0.43
x_rot = xc*np.cos(theta) - yc*np.sin(theta)
y_rot = xc*np.sin(theta) + yc*np.cos(theta)

plt.subplot(323)
plt.xlabel('Polarization Axis (pixels)')
plt.ylabel('Propagation Axis (pixels)')
plt.title('VMI Slice (2 ns e-ToF Window)')

hist,xe,ye= np.histogram2d(x_rot[mask],y_rot[mask], bins=n, range=((-128, 128), (-128, 128)))
plt.imshow(hist.T, extent=[xe[0],xe[-1],ye[0],ye[-1]], cmap=cmap, origin='lower', norm=matplotlib.colors.PowerNorm(gamma))
plt.colorbar()

plt.subplot(324)
plt.xlabel('Polarization Axis (pixels)')
plt.ylabel('Propagation Axis (pixels)')
plt.title('VMI Slice (2 ns e-ToF Window, Log Scale)')

plt.imshow(hist.T, extent=[xe[0],xe[-1],ye[0],ye[-1]], cmap=cmap, origin='lower',norm=matplotlib.colors.LogNorm())
plt.colorbar()

plt.subplot(313)
plt.title('Radial Distribution')
plt.xlabel('Radius (pixels)')
plt.ylabel('Counts')
plt.grid()

r=np.sqrt(x_rot[mask]**2+y_rot[mask]**2)
hist,re= np.histogram(r, bins=n, range=(0, 128))
plt.plot(re[:-1], hist)

plt.tight_layout()
plt.savefig(fname.replace('.cv4','.png'))