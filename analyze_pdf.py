# -*- coding: utf-8 -*-


# %% Initializing
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.backends.backend_pdf
import matplotlib as mpl
from datetime import datetime
import scipy.interpolate as inter


from numba import njit
from numba import errors
from numba.typed import List
mpl.rc('image', cmap='viridis')
mpl.rcParams['font.size'] = 12
plt.close('all')


parser = argparse.ArgumentParser(
    prog='Analyze Data', description="Analyze a clustered dataset and save the data to a pdf")
parser.add_argument('--out', dest='output',
                    default="output.pdf", help="Output Filename")
parser.add_argument('--t', dest='t', type=float,
                    default=0, help="time in us at which the laser is in the chamber")
parser.add_argument('--t0', dest='t0', type=float,
                    default=28.5, help="time of arrival of the zero energy electrons (in ns after t)")
parser.add_argument('--x0', dest='x0', type=int,
                    default=125, help="x coordinate of the zero energy electrons")
parser.add_argument('--y0', dest='y0', type=int,
                    default=125, help="y coordinate of the zero energy electrons")
parser.add_argument('--etof', dest='etof', nargs=2, type=int,
                    default=[20, 40], help="range of electron ToF to include (in ns)")
parser.add_argument('--itof', dest='itof', nargs=2, type=int,
                    default=[0, 40], help="range of ion ToF to include (in us)")
parser.add_argument('--pol', dest='pol', type=float, nargs=2,
                    default=[0, 0], help="angle of major axis (in degrees ccw from s-polarization) and ratio between major and minor axes")
parser.add_argument('filename')

args = parser.parse_args("xe005_e_cluster.h5 --t 500 --pol -26 0.7".split())


@njit
def __e_coincidence(etof_corr, pulse_corr, x, y, t_etof):
    xint = List()
    yint = List()
    tint = List()
    for [j, i] in enumerate(etof_corr):
        idxr = np.searchsorted(pulse_corr, i, side='right')
        idxl = np.searchsorted(pulse_corr, i, side='left')

        for k in range(idxl < idxr):
            xint.append(x[idxl:idxr][k])
            yint.append(y[idxl:idxr][k])
            tint.append(t_etof[j])
    return xint, yint, tint


@njit
def __i_coincidence(tof_corr, pulse_corr, x, y, t, t_tof):
    xs = List()
    ys = List()
    ts = List()
    tofs = List()
    for [j, i] in enumerate(tof_corr):
        idxr = np.searchsorted(pulse_corr, i, side='right')
        idxl = np.searchsorted(pulse_corr, i, side='left')

        for k in range(idxl < idxr):
            xs.append(x[idxl:idxr][k])
            ys.append(y[idxl:idxr][k])
            ts.append(t[idxl:idxr][k])
            tofs.append(t_tof[j])
    return xs, ys, ts, tofs


def P_xy(x):
    return (np.sqrt(5.8029e-4)*(x))*np.sqrt(2*0.03675)


def P_z(t):
    # Ez = -2.433e9*t**5 + 1.482e8*t**4 - 2.937e6*t**3 + 8722*t**2 - 242*t + 0.04998
    Ez = 0.074850168*t**2*(t < 0)-0.034706593*t**2*(t > 0)+3.4926E-05*t**4*(t > 0)
    return np.sqrt(np.abs(Ez))*((Ez > 0)+0-(Ez < 0))*np.sqrt(2*0.03675)


t0 = args.t*1000
tof_range = np.array(args.itof)*1000
etof_range = args.etof
etof_range[1] = args.etof[1]+(0.26-(args.etof[1]-args.etof[0]) % 0.26)
etof_range = np.array(etof_range)
do_tof_gate = True

# %% Loading Data
print('Loading Data:', datetime.now().strftime("%H:%M:%S"))

with h5py.File(args.filename, mode='r') as f:
    x = f['Cluster']['x'][()]
    y = f['Cluster']['y'][()]
    pulse_corr = f['Cluster']['pulse_corr'][()]

    t_tof = f['t_tof'][()]
    t_tof = t_tof-args.t
    tof_corr = f['tof_corr'][()]

    t_etof = f['t_etof'][()]
    t_etof = t_etof-args.t
    etof_corr = f['etof_corr'][()]
# %% e-ToF Coincidence
print('Starting e-ToF Coincidence:', datetime.now().strftime("%H:%M:%S"))

etof_index = np.where(np.logical_and(
    t_etof > etof_range[0], t_etof < etof_range[1]))[0]

xint, yint, tint = __e_coincidence(etof_corr[etof_index], pulse_corr, x, y, t_etof[etof_index])

xint, yint, tint = np.array(xint), np.array(yint), np.array(tint)

# %% i-ToF Coincidence
if do_tof_gate:
    print('Starting i-ToF Coincidence:', datetime.now().strftime("%H:%M:%S"))
    tof_index = np.where(np.logical_and(
        t_tof > tof_range[0], t_tof < tof_range[1]))[0]

    xs, ys, ts, tofs = __i_coincidence(
        tof_corr[tof_index], pulse_corr, xint, yint, tint, t_tof[tof_index])

    xs = np.array(xs)
    ys = np.array(ys)
    ts = np.array(ts)
    tofs = np.array(tofs)
else:
    xs, ys, ts, tofs = xint, yint, tint, t_tof

# %% Conversion to Momentum
torem = np.where(np.logical_and(xs == 195, ys == 234))
torem = np.where(np.logical_and(xs == 0, ys == 0))
xs = np.delete(xs, torem)
ys = np.delete(ys, torem)
ts = np.delete(ts, torem)

noise = np.random.rand(len(ts))*0.26
ts = ts+noise

px = P_xy(xs-args.x0)
py = P_xy(ys-args.y0)
pz = P_z(ts-args.t0)


# %% Rotation
print('Rotating:', datetime.now().strftime("%H:%M:%S"))

width = .6
plot_range = [-width, width]
nbins = 512

h, edges = np.histogramdd((px, py, pz), bins=(nbins, nbins, nbins),
                          range=[plot_range, plot_range, plot_range])
Xc, Yc, Zc = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1])
h[0, 0, :] = np.zeros_like(h[0, 0, :])
h[-1, -1, :] = np.zeros_like(h[0, 0, :])
numbins = 256

xv = Xc[0, :, 0]
yv = Yc[:, 0, 0]
zv = Zc[0, 0, :]

theta = -1

hp = inter.interpn((xv, yv, zv), h, (Xc*np.cos(theta)+Yc*np.sin(theta),
                                     Yc*np.cos(theta)-Xc*np.sin(theta), Zc),
                   fill_value=0, bounds_error=False,)
# %% Plotting and Saving
print('Plotting and Saving:', datetime.now().strftime("%H:%M:%S"))
with matplotlib.backends.backend_pdf.PdfPages(args.output) as pdf:
    fsize = (10, 10)
    window = [0, 0, 1, 0.95]
    # %%% Page 1: Unprocessed VMI
    plt.figure(figsize=fsize)
    plt.suptitle('Unprocessed VMI')

    plt.hist2d(y, x, bins=256, range=[[0, 256], [0, 256]])
    plt.colorbar()
    plt.title("Full VMI Image (Log Scale)")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.tight_layout(rect=window)
    pdf.savefig()

    # %%% Page 2: Filtered and Rotated VMI
    plt.figure(figsize=fsize)
    plt.imshow(np.sum(hp, axis=2))
    plt.title("Rotated VMI Image")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.tight_layout(rect=window)

    # %%% Page 3: ToF Spectra
    plt.figure(figsize=fsize)
    plt.suptitle('ToF Spectra')
    plt.subplot(211)
    plt.hist(tofs, bins=300, range=tof_range)
    plt.title("Ion ToF Spectrum")
    plt.xlabel("TOF (ns)")
    plt.ylabel("Count")
    plt.tight_layout(rect=window)

    plt.subplot(212)
    plt.hist(ts, bins=300, range=etof_range)
    plt.title("Electron ToF Spectrum")
    plt.xlabel("TOA (ns)")
    plt.ylabel("Count")
    plt.tight_layout(rect=window)

    pdf.savefig()

    # %%% Page 4: Momentum Projections

    plt.figure(figsize=fsize)
    plt.suptitle('Momentum Projections')

    xyhist = np.sum(hp, axis=2)
    xzhist = np.sum(hp, axis=0)
    yzhist = np.sum(hp, axis=1)

    plt.subplot(223)
    plt.imshow(xyhist, extent=(-width, width, -width, width), origin='lower')
    plt.xlabel("$p_x$ (a.u.)")
    plt.ylabel("$p_y$ (a.u.)")
    plt.tight_layout(rect=window)

    plt.subplot(224)
    plt.imshow(yzhist, extent=(-width, width, -width, width), origin='lower')
    plt.xlabel("$p_z$ (a.u.)")
    plt.ylabel("$p_y$ (a.u.)")
    plt.tight_layout(rect=window)

    plt.subplot(221)
    plt.imshow(xzhist.transpose(), extent=(-width, width, -width, width), origin='lower')
    plt.xlabel("$p_x$ (a.u.)")
    plt.ylabel("$p_z$ (a.u.)")
    plt.tight_layout(rect=window)

    pdf.savefig()

    # %%% Page 5: Momentum Cuts

    plt.figure(figsize=fsize)
    plt.suptitle('Momentum Cuts')

    xyhist = np.sum(hp[:, :, nbins//2-2:nbins//2+2], axis=2)
    xzhist = np.sum(hp[nbins//2-2:nbins//2+2, :, :], axis=0)
    yzhist = np.sum(hp[:, nbins//2-2:nbins//2+2, :], axis=1)

    plt.subplot(223)
    plt.imshow(xyhist, extent=(-width, width, -width, width), origin='lower')
    plt.xlabel("$p_x$ (a.u.)")
    plt.ylabel("$p_y$ (a.u.)")
    plt.tight_layout(rect=window)

    plt.subplot(224)
    plt.imshow(yzhist, extent=(-width, width, -width, width), origin='lower')
    plt.xlabel("$p_z$ (a.u.)")
    plt.ylabel("$p_y$ (a.u.)")
    plt.tight_layout(rect=window)

    plt.subplot(221)
    plt.imshow(xzhist.transpose(), extent=(-width, width, -width, width), origin='lower')
    plt.xlabel("$p_x$ (a.u.)")
    plt.ylabel("$p_z$ (a.u.)")
    plt.tight_layout(rect=window)

    pdf.savefig()

    # %%% Page 5: Polarization Plane
    plt.figure(figsize=fsize)
    plt.suptitle('Polarization Plane')

    plt.imshow(np.sum(hp, axis=1).transpose(),
               extent=(-width, width, -width, width), origin='lower')

    plt.xlabel("$p_y$ (a.u.)")
    plt.ylabel("$p_z$ (a.u.)")
    plt.tight_layout(rect=window)

    t = np.linspace(0, 2*np.pi, num=100)
    pol_x, pol_y = .5*np.cos(t), .5*args.pol[1]*np.sin(t)

    (pol_x, pol_y) = (pol_x*np.sin(args.pol[0]*np.pi/180)+pol_y*np.cos(args.pol[0]*np.pi/180),
                      pol_y*np.sin(args.pol[0]*np.pi/180)-pol_x*np.cos(args.pol[0]*np.pi/180))

    plt.plot(pol_x, pol_y, 'r')
    plt.arrow(pol_x[40], pol_y[40], np.diff(pol_x)[40]/10,
              np.diff(pol_y)[40]/10, head_width=.03, color='r')
    plt.arrow(pol_x[90], pol_y[90], np.diff(pol_x)[90]/10,
              np.diff(pol_y)[90]/10, head_width=.03, color='r')

    plt.scatter(pol_x[[0, 50]], pol_y[[0, 50]], c='r', marker='o')

    pdf.savefig()
