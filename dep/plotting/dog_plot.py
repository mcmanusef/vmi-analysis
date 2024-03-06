import h5py
import matplotlib.widgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

matplotlib.use("QT5Agg")

file = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240208\xe_01_p.cv4"
# file=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clust_v3\xe001_p.cv3"

num = -1
cx, cy = 126.5, 141.5


def P_xy(x):
    return np.sqrt(0.000503545) * (x) * np.sqrt(2 * 0.03675)


with h5py.File(file, mode='r') as f:
    x = f['x'][0:num]
    y = f['y'][0:num]
    t = f['t'][0:num] % 1e6

n = 256
px = P_xy(x - cx)
py = P_xy(y - cy)
plt.figure()
plt.hist2d(x, y, range=((0, 256), (0, 256)), bins=n, cmap='jet', density=True)
plt.scatter(cx, cy)

plt.figure()
plt.hist(t, bins=1000, range=(0, 20000))
idx = np.argwhere(np.logical_and(t < 749000, t > 749100)).flatten()
idx2= np.argwhere((x-205)**2+(y-197)**2>1.5).flatten()
idx3= np.argwhere((x-197)**2+(y-205)**2>1).flatten()
idx=np.intersect1d(np.intersect1d(idx,idx2),idx3)
# idx=np.argwhere(t<1e6).flatten()
plt.hist(t[idx], bins=1000, range=(749000,749100))
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Create the histogram
r = 256

#
# hist, xedges, yedges, im = ax.hist2d(px[idx], py[idx], range=((-0.6, 0.6), (-0.6, 0.6)), bins=n, cmap='jet',
#                                      density=True)
hist, xedges, yedges, im = ax.hist2d(x, y, range=((0, 256), (0, 256)), bins=n, cmap='jet', density=True)
xx, yy = np.mgrid[-r:r:2 * r / n, -r:r:2 * r / n]
rr = np.sqrt(xx ** 2 + yy ** 2)
# hist=hist/np.max(hist)
# Create the slider axes
axes = [plt.axes((0.25, 0.15 - 0.015 * i, 0.65, 0.015), facecolor='lightgoldenrodyellow') for i in range(8)]
# Create the slider
sliders = {
    'gamma': matplotlib.widgets.Slider(axes[0], 'Gamma', 0, 2.0, valinit=1.0),
    'max': matplotlib.widgets.RangeSlider(axes[1], 'Value Range', 0, 2, valinit=(0, 1)),
    'blur': matplotlib.widgets.Slider(axes[2], 'Blur', 0, 10, valinit=0),
    'dog_strength': matplotlib.widgets.Slider(axes[3], 'DoG Strength', 0, 1, valinit=0),
    'dog_kernel': matplotlib.widgets.RangeSlider(axes[4], 'DoG Kernel', 0, 20, valinit=(0, 1)),
    'dog_diff': matplotlib.widgets.Slider(axes[5], 'DoG Difference', -1, 1, valinit=0),
    'dog_range': matplotlib.widgets.RangeSlider(axes[6], 'DoG Range', 0, 1, valinit=(0, 1)),
    'radius': matplotlib.widgets.RangeSlider(axes[7], 'Radius Range', 0, r, valinit=(0, r))
}


# Update function for the slider
def update(val):
    c = hist
    c0 = scipy.ndimage.gaussian_filter(c, sliders['blur'].val)
    cd = c  # dog_clamp(c)
    c1 = scipy.ndimage.gaussian_filter(cd, sliders['dog_kernel'].val[0])
    c2 = scipy.ndimage.gaussian_filter(cd, sliders['dog_kernel'].val[1])
    cdog = (c1 - c2 * np.abs(sliders['dog_diff'].val)) * np.sign(sliders['dog_diff'].val + 0.0001)
    cdog = dog_clamp(cdog)
    c = c0 * (1 - sliders['dog_strength'].val) + cdog * sliders['dog_strength'].val
    c = np.maximum(np.minimum(c, sliders['max'].val[1] * np.max(c)) - sliders['max'].val[0] * np.max(c), 0)
    c = np.power(c, sliders['gamma'].val)
    c = c / np.max(c) * np.max(hist) / (sliders['max'].val[1] - sliders['max'].val[0])

    im.set_array(np.where(np.logical_and(rr < sliders['radius'].val[1], rr > sliders['radius'].val[0]), c, 0).T)
    fig.canvas.draw_idle()


def dog_clamp(cdog):
    return np.maximum(
        np.minimum(cdog, sliders['dog_range'].val[1] * np.max(cdog)) - sliders['dog_range'].val[0] * np.max(cdog), 0)


# Connect the slider to the update function
[sliders[i].on_changed(update) for i in sliders]
# Initial plot display
update(1.0)
plt.show()