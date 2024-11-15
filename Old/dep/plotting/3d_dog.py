import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets
import scipy.ndimage
import h5py

from plotting.plotting_utils import itof_filter
import coincidence_v4

matplotlib.use('Qt5Agg')

def do_nc_with_dog(file, angle, center, dead_pixels, rx, ry, rt, rtoa, n):
    # Load data
    data = coincidence_v4.load_file_nc(file)
    x, y, toa, etof = data
    t = etof + 0.26 * np.random.random_sample(len(etof))

    # Apply dead pixel filter and rotate coordinates
    x, y, t, toa = dp_filter(dead_pixels, x, y, t, toa)
    x, y = rotate_coords(angle, center, x, y)

    # Filter coordinates
    t, x, y, toa = filter_coords((t, x, y, toa), (rt, rx, ry, rtoa))

    # Create the figure and the sliders
    fig, ax = plt.subplots(1,1, num="No Coincidence with DoG")
    plt.subplots_adjust(bottom=0.25)
    slider_axes = [plt.axes([0.25, 0.1 - 0.015 * i, 0.65, 0.015], facecolor='lightgoldenrodyellow') for i in range(7)]
    sliders = {
        'gamma': matplotlib.widgets.Slider(slider_axes[0], 'Gamma', 0, 2.0, valinit=1.0),
        'max': matplotlib.widgets.RangeSlider(slider_axes[1], 'Value Range', 0, 2, valinit=(0, 1)),
        'blur': matplotlib.widgets.Slider(slider_axes[2], 'Blur', 0, 10, valinit=0),
        'dog_strength': matplotlib.widgets.Slider(slider_axes[3], 'DoG Strength', 0, 1, valinit=0),
        'dog_kernel': matplotlib.widgets.RangeSlider(slider_axes[4], 'DoG Kernel', 0, 20, valinit=(0, 1)),
        'dog_diff': matplotlib.widgets.Slider(slider_axes[5], 'DoG Difference', -1, 1, valinit=0),
        'dog_range': matplotlib.widgets.RangeSlider(slider_axes[6], 'DoG Range', 0, 1, valinit=(0, 1))
    }
    hist, _, _, _ = ax.hist2d(y, t, bins=n, range=[rx, ry])

    # Update function for the slider
    def update(val):
        # Apply difference of Gaussians (DoG) processing
        c0 = scipy.ndimage.gaussian_filter(hist, sliders['blur'].val)
        c1 = scipy.ndimage.gaussian_filter(hist, sliders['dog_kernel'].val[0])
        c2 = scipy.ndimage.gaussian_filter(hist, sliders['dog_kernel'].val[1])
        cdog = (c1 - c2 * np.abs(sliders['dog_diff'].val)) * np.sign(sliders['dog_diff'].val + 0.0001)
        cdog = dog_clamp(cdog)
        c = c0 * (1 - sliders['dog_strength'].val) + cdog * sliders['dog_strength'].val
        c = np.maximum(np.minimum(c, sliders['max'].val[1] * np.max(c)) - sliders['max'].val[0] * np.max(c), 0)
        c = np.power(c, sliders['gamma'].val)
        c = c / np.max(c) * np.max(hist) / (sliders['max'].val[1] - sliders['max'].val[0])

        # Update the plot
        ax.imshow(c, origin='lower', extent=[ry[0], ry[1], rt[0], rt[1]], aspect='auto')
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    for slider in sliders.values():
        slider.on_changed(update)

    # Initial plot display
    update(None)
    plt.show()

# Helper functions
def dp_filter(dead_pixels, x, y, *args):
    dp_dists = [np.sqrt((x - x0) ** 2 + (y - y0) ** 2) for x0, y0 in dead_pixels]
    dp_index = np.argwhere(np.min(dp_dists, axis=0) > 2).flatten()
    return (x[dp_index], y[dp_index], *[arg[dp_index] for arg in args])

def rotate_coords(angle, center, x, y):
    x, y = coincidence_v4.rotate_data(x - center[0], y - center[1], angle)
    return x, y

def filter_coords(coords, ranges):
    index=functools.reduce(np.intersect1d,
                           (np.argwhere([r[0]<c<r[1] for c in c_list]) for c_list,r in zip(coords,ranges)))
    return tuple(c[index] for c in coords)

def dog_clamp(cdog):
    return np.maximum(np.minimum(cdog, 1 * np.max(cdog)) - 0 * np.max(cdog), 0)  # DoG Range is (0, 1)

if __name__ == '__main__':
    file = "J:\\ctgroup\\DATA\\UCONN\\VMI\\VMI\\20240122\\o2_06_e.cv4"
    angle = -np.arctan2(69-198,150-101)
    center = (122,131)
    dead_pixels = [(191, 197), (196, 194), (0, 0)]
    rx = (-5,5)
    ry = (-center[1], 256-center[1])
    rt = (749040, 749070)
    rtoa = (748800-25, 748850-25)
    n = 256

    do_nc_with_dog(file, angle, center, dead_pixels, rx, ry, rt, rtoa, n)

