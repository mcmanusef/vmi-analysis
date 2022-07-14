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
import scipy.io
from numba import njit
from numba.typed import List
import mayavi.mlab as mlab
from matplotlib.cm import ScalarMappable as SM
from skimage import measure


mpl.rc('image', cmap='viridis')
font = {'weight': 'bold',
        'size': 22}

plt.rc('font', **font)
plt.close('all')


parser = argparse.ArgumentParser(
    prog='Analyze Data', description="Analyze a clustered dataset and save the data to a pdf")
parser.add_argument('--out', dest='output',
                    default="output.pdf", help="Output Filename")
parser.add_argument('--t', dest='t', type=float,
                    default=500, help="time in us at which the laser is in the chamber")
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
parser.add_argument('--w', dest='width', type=float,
                    default=1, help="max momentum to plot")
parser.add_argument('--nbins', dest='nbins', type=int,
                    default=256, help="resolution of the plots")
parser.add_argument('--info', dest='info_file',
                    help="file containing run information to include")
parser.add_argument('--data', dest='data_file',
                    help="mat file to save raw data to")

parser.add_argument('filename')

args = parser.parse_args()


@njit
def __e_coincidence(etof_corr, pulse_corr, x, y, t_etof):
    xint = List()
    yint = List()
    tint = List()
    for [j, i] in enumerate(etof_corr):
        idxr = np.searchsorted(pulse_corr, i, side='right')
        idxl = np.searchsorted(pulse_corr, i, side='left')

        for k in range(-idxl + idxr):
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

        for k in range(-idxl + idxr):
            xs.append(x[idxl:idxr][k])
            ys.append(y[idxl:idxr][k])
            ts.append(t[idxl:idxr][k])
            tofs.append(t_tof[j])
    return xs, ys, ts, tofs


def P_xy(x):
    return (np.sqrt(5.8029e-4)*(x))*np.sqrt(2*0.03675)


def P_z(t):
    # Ez = -2.433e9*t**5 + 1.482e8*t**4 - 2.937e6*t**3 + 8722*t**2 - 242*t + 0.04998
    Ez = ((6.7984E-05*t**4+5.42E-04*t**3+1.09E-01*t**2)*(t < 0) +
          (-5.64489E-05*t**4+3.37E-03*t**3-6.94E-02*t**2)*(t > 0))
    return np.sqrt(np.abs(Ez))*((Ez > 0)+0-(Ez < 0))*np.sqrt(2*0.03675)


t0 = args.t*1000
tof_range = np.array(args.itof)*1000
etof_range = args.etof
etof_range[1] = args.etof[1]+(0.26-(args.etof[1]-args.etof[0]) % 0.26)
etof_range = np.array(etof_range)
do_tof_gate = True

data_dict = {}
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

width = args.width
plot_range = [-width, width]

nbins = args.nbins

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
phi = args.pol[0]*np.pi/180
print(np.sin(phi), np.cos(phi))

hp = inter.interpn((xv, yv, zv), h, (Xc*np.cos(theta)+Yc*np.sin(theta),
                                     Yc*np.cos(theta)-Xc*np.sin(theta), Zc),
                   fill_value=0, bounds_error=False)

hp = inter.interpn((xv, yv, zv), hp, (Yc*np.cos(phi)-Zc*np.sin(phi), Xc, Zc *
                                      np.cos(phi)+Yc*np.sin(phi)), fill_value=0, bounds_error=False)
data_dict['3d_hist'] = hp
data_dict['xv'] = xv
data_dict['yv'] = yv
data_dict['zv'] = zv
# %% Plotting and Saving
print('Plotting and Saving:', datetime.now().strftime("%H:%M:%S"))
with matplotlib.backends.backend_pdf.PdfPages(args.output) as pdf:
    fsize = (10, 10)
    window = [0, 0, 1, 0.95]
    # %%% Page 0: Run Information
    plt.figure(figsize=fsize)
    plt.suptitle('Run Information')
    INFO = """
        Filename: {filename} \n
        Polarization: {pol:.2f}$^\\circ$ counter-clockwise from s polarization \n
        Ellipticity: {elip:.2f}, {hand} \n
        Pulses: {pulses} \n
        Electrons (clusters): {ec} \n
        Electrons (ToF): {et} \n
        Ions: {it} \n
        Other info: \n
        {fileinfo} \n
        """.format(
        filename=args.filename.split('\\')[-1],
        pol=args.pol[0],
        elip=abs(args.pol[1]),
        hand="Left Handed" if args.pol[1] < 0 else "Right Handed",
        pulses=max(pulse_corr),
        ec=len(pulse_corr),
        et=len(t_etof),
        it=len(t_tof),
        fileinfo=open(args.info_file).read() if args.info_file else "NONE"
    )

    plt.figtext(0.1, 0.1, INFO)
    pdf.savefig()
    # %%% Page 1: Unprocessed VMI
    plt.figure(figsize=fsize)
    plt.suptitle('Unprocessed VMI')

    plt.hist2d(y, x, bins=256, range=[[0, 256], [0, 256]])
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
    pdf.savefig()

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

    data_dict["xy_proj"] = xyhist = np.sum(hp, axis=2)
    data_dict["xz_proj"] = xzhist = np.sum(hp, axis=0)
    data_dict["yz_proj"] = yzhist = np.sum(hp, axis=1)

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

    data_dict["xy_cut"] = xyhist = np.sum(hp[:, :, nbins//2-2:nbins//2+2], axis=2)
    data_dict["xz_cut"] = xzhist = np.sum(hp[nbins//2-2:nbins//2+2, :, :], axis=0)
    data_dict["yz_cut"] = yzhist = np.sum(hp[:, nbins//2-2:nbins//2+2, :], axis=1)

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
    pol_x, pol_y = width*0.8*np.cos(t), 0.8*width*args.pol[1]*np.sin(t)

    (pol_x, pol_y) = (pol_y, -pol_x)

    plt.plot(pol_x, pol_y, 'r')
    plt.arrow(pol_x[40], pol_y[40], np.diff(pol_x)[40]/10,
              np.diff(pol_y)[40]/10, head_width=.03*width, color='r')
    plt.arrow(pol_x[90], pol_y[90], np.diff(pol_x)[90]/10,
              np.diff(pol_y)[90]/10, head_width=.03*width, color='r')

    plt.scatter(pol_x[[0, 50]], pol_y[[0, 50]], c='r', marker='o')

    pdf.savefig()

    # %%% Page 5: p_x Slices
    plt.figure(figsize=(10, 50))
    plt.suptitle('$p_x$ slices')

    for j, i in enumerate(np.linspace(-0.3, 0.3, num=9)):
        c = (i, 0, 0)

        cx = nbins//2+int(c[0]/np.diff(xv)[0])

        w = 2
        data_dict["px{}".format(j)] = yzhist = np.sum(hp[:, cx-w:cx+w, :], axis=1)

        plt.subplot(911+j).set_title("$p_x={px:.2f}$".format(px=i))
        plt.imshow(yzhist.transpose(), extent=(-width, width, -width, width), origin='lower')
        plt.xlabel("$p_y$")
        plt.ylabel("$p_z$")
        plt.tight_layout(rect=window)
    pdf.savefig()


mlab.figure(figure="Rotated 3d Projection", bgcolor=(0, 0, 0))

minbin = 1
numbins = min(numbins, int(h.max())-minbin)
cm = SM().to_rgba(np.array(range(numbins))**1)
cm[:, 3] = (np.array(range(numbins))/numbins)**1.5

for i in range(numbins):
    iso_val = i*(int(hp.max())-minbin)/numbins+minbin
    verts, faces, _, _ = measure.marching_cubes_lewiner(
        hp, iso_val, spacing=(2*width/nbins, 2*width/nbins, 2*width/nbins))

    # ax.plot_trisurf(verts[:, 0]-width, verts[:, 1]-width, faces, verts[:, 2]-width,
    #                 color=cm[i], shade=True, zorder=numbins+100-i)

    mlab.triangular_mesh(verts[:, 0]-width, verts[:, 1]-width, verts[:, 2]-width,
                         faces, color=tuple(cm[i, :3]), opacity=cm[i, 3])
    if i == 0:
        mlab.axes()
mpl.rc('image', cmap='gray')
mlab.savefig(args.output[:-4]+".png")

scipy.io.savemat(args.data_file, data_dict) if args.data_file else 0
scipy.io.savemat(args.data_file, data_dict) if args.data_file else 0
scipy.io.savemat(args.data_file, data_dict) if args.data_file else 0
