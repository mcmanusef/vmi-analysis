import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import coincidence_v4

def trans_jet(opaque_point=0.15):
    # Create a colormap that looks like jet
    cmap = plt.cm.jet

    # Create a new colormap that is transparent at low values
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:int(cmap.N * opaque_point), -1] = np.linspace(0, 1, int(cmap.N * opaque_point))
    return ListedColormap(cmap_colors)

def itof_filter(rtof, itof, *args):
    itof_index = [i for i, it in enumerate(itof) if rtof[0] < it < rtof[1]]
    return tuple(arg[itof_index] for arg in args)


def rotate_coords(angle, center, x, y):
    x, y = coincidence_v4.rotate_data(x - center[0], y - center[1], angle)
    return x, y


def dp_filter(dead_pixels, x, y, *args):
    dp_dists = [np.sqrt((x - x0) ** 2 + (y - y0) ** 2) for x0, y0 in dead_pixels]
    dp_index = np.argwhere(functools.reduce(np.minimum, dp_dists) > 2).flatten()
    x = x[dp_index]
    y = y[dp_index]
    out=tuple(arg[dp_index] for arg in args)
    print(len(x))
    return x, y, *out


def filter_coords(coords,ranges):
    index=functools.reduce(np.intersect1d,
                           (np.argwhere([r[0]<c<r[1] for c in c_list]) for c_list,r in zip(coords,ranges)))
    print(len(index))
    return tuple(c[index] for c in coords)


def diff_filter(max_diff,t1,t2, *args):
    index=np.argwhere(np.abs(t1-t2)<max_diff).flatten()
    return t1[index], t2[index], *(arg[index] for arg in args)
