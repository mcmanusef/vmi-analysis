import os
import h5py
import matplotlib
import matplotlib.pyplot as plt
# import mayavi.mlab
import numpy as np
import scipy.io
# from mayavi import mlab
from Old import coincidence_v4
from plotting.plotting_utils import itof_filter, rotate_coords, dp_filter, filter_coords

matplotlib.rc('image', cmap='jet')
matplotlib.use('Qt5Agg')
plt.close("all")
def main(do_coincidence=True, do_nc=True, do_3d=False, do_clusters=True, do_raw=False, calibrate=False, save_or_load=False):
    # file=r"C:\Users\mcman\Code\VMI\indev\test.cv3"
    #
    # file=r"C:\Users\mcman\Code\VMI\Data\air_s_70.cv3"

    # file=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240122\o2_02_p.cv4"
    # file=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240208\xe_03_a.cv4"
    # file=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20231031\after_torr.cv4"
    file=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240805\s_3.cv4"
    n = 256
    t0 = 748000
    rt=(749040, 749070)
    # rt=(749000, 749040)
    rtoa=(748800-100, 748850-25)
    # rt=rtoa=(0,1e6)
    rtof=(10000 + t0, 12000 + t0)
    rtof_plot=(0 + t0, 30000 + t0)
    rtof=(766000, 768000)
    rtof=(770000, 773000)
    angle=-np.arctan2(69-198,150-101)
    # angle=0
    center=(122,131)
    # center=(0,0)
    rx=(-center[0],256-center[0])
    ry=(-center[1],256-center[1])

    # rx=(-5,5)
    # ry=(-5,5)

    # file=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\xe005_e.cv4"
    # n = 256
    # rt=(490,520)
    # rtoa=(0,1000)
    # rtof=(15000,17000)
    # rtof_plot=(0,30000)
    # angle=np.arctan2(91-168,183-66)
    # center=(120,132)
    # rx=(-120,256-120)
    # ry=(-132,256-132)


    # ry=(-2,2)
    print(angle)

    dead_pixels = [
        (191, 197),
        (196, 194),
        (0, 0)
    ]

    if do_coincidence:
        if save_or_load:
            keys=['x','y','toa','etof','itof']
            if os.path.exists(file+".mat"):
                data=scipy.io.loadmat(file+".mat")
                x,y,toa,etof,itof=(data[k].flatten() for k in keys)
            else:
                data= coincidence_v4.load_file_coin(file)
                x,y,toa,etof,itof=data
                scipy.io.savemat(file+".mat", {k:d for k,d in zip(keys,data)})
        else:
            data= coincidence_v4.load_file_coin(file)
            x,y,toa,etof,itof=data

        t=etof+0.26*np.random.random_sample(len(etof))

        x,y,t, itof,toa= dp_filter(dead_pixels, x, y, t, itof, toa)
        x,y,t,toa = itof_filter(rtof, itof, x, y, t, toa)
        x, y = rotate_coords(angle, center, x, y)
        # t, toa, x, y = diff_filter(200, t, toa, x, y)
        t,x,y,toa = filter_coords((t, x, y, toa), (rt, rx, ry, rtoa))

        if calibrate:
            x=P_xy(x)
            y=P_xy(y)
            t=P_z(t-749048)
            rx=ry=rt=(-1,1)

        tofs=[['e-tof'], ['i-tof']]
        grid=[['xy','yt'],['xt', tofs]]
        fig, axd= plt.subplot_mosaic(grid, num="Coincidence")
        plt.subplots_adjust(bottom=0.2)
        h1,*_,i1=axd['xy'].hist2d(x, y, bins=n, range=[rx, ry])
        h2,*_,i2=axd['yt'].hist2d(t, y, bins=n, range=[rt, ry])
        h3,*_,i3=axd['xt'].hist2d(x, t, bins=n, range=[rx, rt])
        axd['e-tof'].hist(t,bins=1000, range=rt)
        axd['i-tof'].hist(itof,bins=1000, range=rtof_plot)
        plt.sca(axd['i-tof'])
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
        update(1)
    #
    # if do_3d:
    #     hist3d=np.histogramdd((x,y,t), range=(rx,ry,rt), bins=128)[0]
    #     mayavi.mlab.figure()
    #     mayavi.mlab.contour3d(hist3d, contours=30, transparent=True)

    if do_nc:
        if save_or_load:
            keys=['x','y','toa','etof']
            if os.path.exists(file+"_nc.mat"):
                data=scipy.io.loadmat(file+"_nc.mat")
                x,y,toa,etof=(data[k].flatten() for k in keys)
            else:
                data= coincidence_v4.load_file_nc(file)
                x,y,toa,etof=data
                scipy.io.savemat(file+"_nc.mat", {k:d for k,d in zip(keys,data)})
        else:
            data= coincidence_v4.load_file_nc(file)
            x,y,toa,etof=data
        t=etof+0.26*np.random.random_sample(len(etof))

        x,y,t,toa= dp_filter(dead_pixels, x, y, t, toa)
        x, y = rotate_coords(angle, center, x, y)
        # t, toa, x, y = diff_filter(200, t, toa, x, y)
        t,x,y,toa = filter_coords((t, x, y, toa), (rt, rx, ry, rtoa))

        fig, ax= plt.subplots(2,2, num="No Coincidence")
        ax[0,0].hist2d(x, y, bins=n, range=[rx, ry])
        ax[0,1].hist2d(t, y, bins=n, range=[rt, ry])
        ax[1,0].hist2d(x, t, bins=n, range=[rx, rt])
        ax[1,1].hist(t,bins=1000, range=rt)

    if do_clusters:
        with h5py.File(file) as f:
            x,y,toa=f['x'][()],f['y'][()],f['t'][()]
            print(len(x))
        # x,y,toa,etof=coincidence_v4.load_file_nc(file)
        t=toa+0.26*np.random.random_sample(len(toa))

        x,y,t= dp_filter(dead_pixels, x, y, t)
        x, y = rotate_coords(angle, center, x, y)
        t,x,y = filter_coords((t, x, y), (rtoa, rx, ry))

        fig, ax= plt.subplots(2,2, num="Clusters")
        ax[0,0].hist2d(x, y, bins=n, range=[rx, ry])
        ax[0,1].hist2d(t, y, bins=n, range=[rtoa, ry])
        ax[1,0].hist2d(x, t, bins=n, range=[rx, rtoa])
        ax[1,1].hist(t,bins=1000, range=rtoa)

    if do_raw:
        with h5py.File(file) as f:
            x,y,toa=f['x'][()],f['y'][()],f['t'][()]
            print(len(x))
        # x,y,toa,etof=coincidence_v4.load_file_nc(file)
        t=toa+1.6*np.random.random_sample(len(toa))
        x, y = rotate_coords(angle, center, x, y)

        fig, ax= plt.subplots(2,2, num="Raw")
        ax[0,0].hist2d(x, y, bins=n, range=[rx, ry])
        ax[0,1].hist2d(t, y, bins=n, range=[rtoa, ry])
        ax[1,0].hist2d(x, t, bins=n, range=[rx, rtoa])
        ax[1,1].hist(t,bins=1000, range=rtoa)


def P_xy(x):
    return np.sqrt(0.000503545) * x * np.sqrt(2 * 0.03675)

def P_z(t):
    pos = np.poly1d([4.51E+04, 1.63E+06, -5.49E+04, 0, 0])
    neg = np.poly1d([0, 0, 8.65E+04, 0, 0])
    Ez = pos(t/1000) * (t > 0) + neg(t/1000) * (t < 0)
    return np.sqrt(np.abs(Ez)) * (0 + (Ez > 0) - (Ez < 0)) * np.sqrt(2 * 0.03675)

#%%
if __name__ == '__main__':
    main()
