import functools
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import mayavi.mlab
import numpy as np
from mayavi import mlab
import cv3_analysis
from minor_utils.tof_calibration import get_calibration
from indev import coincidence_v4

matplotlib.rc('image', cmap='jet')
matplotlib.use('Qt5Agg')
plt.close("all")
def main(do_coincidence=False, do_nc=True, do_3d=False, do_clusters=False):
    file=r"C:\Users\mcman\Code\VMI\indev\test.cv3"
    file=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20230913\air_6.cv3"
    n = 512
    rx=ry=(0, 256)
    # rx=(118,122)
    # ry=(128,134)
    rt=(380,420)
    rtoa=(240,270)
    rtof=(9000,11000)
    rtof_plot=(0,20000)
    rt=rtoa=rtof=(0,1e6)
    angle=0#-np.arctan2(-100,56)
    center=(0,0)
    print(angle)

    dead_pixels = [
        (191, 197),
        (196, 194),
        (0, 0)
    ]

    data=coincidence_v4.load_file(file)

    if do_coincidence:
        x,y,toa,etof,itof=data
        t=etof+0.26*np.random.random_sample(len(etof))

        x,y,t, itof,toa= dp_filter(dead_pixels, x, y, t, itof, toa)
        x,y,t,toa = itof_filter(rtof, itof, x, y, t,toa)
        x, y = rotate_coords(angle, center, x, y)
        t, toa, x, y = diff_filter(200, t, toa, x, y)
        t,x,y,toa = filter_coords((t,x,y,toa), (rt,rx,ry,rtoa))

        fig, ax= plt.subplots(2,2, num="Coincidence")
        plt.subplots_adjust(bottom=0.2)
        h1,*_,i1=ax[0,0].hist2d(x, y, bins=n, range=[rx, ry])
        h2,*_,i2=ax[0,1].hist2d(t, y, bins=n, range=[rt, ry])
        h3,*_,i3=ax[1,0].hist2d(x, t, bins=n, range=[rx, rt])
        ax[1,1].hist(itof,bins=1000, range=rtof_plot)
        plt.sca(ax[1,1])
        plt.axvline(rtof[0],c='r')
        plt.axvline(rtof[1],c='r')

        ax_gamma_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider_gamma = matplotlib.widgets.Slider(ax_gamma_slider, 'Gamma', 0.1, 2.0, valinit=1.0)
        def update(val):
            gamma = slider_gamma.val
            for hist,im in [(h1,i1),(h2,i2),(h3,i3)]:
                c = np.power(hist, gamma)  # Apply gamma correction to the histogram
                c = c/np.max(c)*np.max(hist)
                im.set_array(c.T)
            fig.canvas.draw_idle()
        slider_gamma.on_changed(update)
        update(1.0)

    if do_3d:
        hist3d=np.histogramdd((x,y,t), range=(rx,ry,rt), bins=64)[0]
        mayavi.mlab.figure()
        mayavi.mlab.contour3d(hist3d, contours=10, transparent=True)

    if do_nc:
        x,y,toa,etof=coincidence_v4.load_file_nc(file)
        t=etof+0.26*np.random.random_sample(len(etof))

        x,y,t,toa= dp_filter(dead_pixels, x, y, t, toa)
        x, y = rotate_coords(angle, center, x,y)
        t, toa, x, y = diff_filter(200, t, toa, x, y)
        t,x,y,toa = filter_coords((t,x,y,toa), (rt,rx,ry,rtoa))

        fig, ax= plt.subplots(2,2, num="No Coincidence")
        ax[0,0].hist2d(x, y, bins=n, range=[rx, ry], norm="log")
        ax[0,1].hist2d(t, y, bins=n, range=[rt, ry], norm="log")
        ax[1,0].hist2d(x, t, bins=n, range=[rx, rt], norm="log")
        ax[1,1].hist(t,bins=1000, range=rt)

    if do_clusters:
        x,y,toa,etof=coincidence_v4.load_file_nc(file)
        t=toa+0.26*np.random.random_sample(len(toa))

        x,y,t= dp_filter(dead_pixels, x, y, t)
        x, y = rotate_coords(angle, center,x,y)
        t,x,y = filter_coords((t,x,y), (rtoa,rx,ry))

        fig, ax= plt.subplots(2,2, num="Clusters")
        ax[0,0].hist2d(x, y, bins=n, range=[rx, ry], norm="log", vmax=400)
        ax[0,1].hist2d(t, y, bins=n, range=[rtoa, ry], norm="log")
        ax[1,0].hist2d(x, t, bins=n, range=[rx, rtoa], norm="log")
        ax[1,1].hist(t,bins=1000, range=rtoa)


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

#%%
if __name__ == '__main__':
    main()
