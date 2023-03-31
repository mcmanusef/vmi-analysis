# %% Initializing

import itertools
import os
from multiprocessing import Pool

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.stats
from matplotlib.colors import ListedColormap
from scipy import fft
from scipy.io import loadmat
from scipy.optimize import curve_fit
from cv3_analysis import load_cv3
from scipy.stats import qmc
mpl.rc('image',cmap='jet')
mpl.use('Qt5Agg')

# %% Functions
def trans_jet(opaque_point=0.1):
    # Create a colormap that looks like jet
    cmap = plt.cm.jet

    # Create a new colormap that is transparent at low values
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:int(cmap.N*opaque_point), -1] = np.linspace(0, 1, int(cmap.N*opaque_point))
    return ListedColormap(cmap_colors)


def wrap_between(data,low=0,high=180):
    return (data-low) % (high-low) + low


def unwrap(data, start_between=(-90,90), period=180):
    data[0]= wrap_between(data[0], *start_between)
    return np.unwrap(data, period=period)


def pairwise(iterable):
    """Return an iterator that aggregates elements in pairs. Example: pairwise('ABCDEFG') --> AB BC CD DE EF FG"""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def organize(x):
    def filter_valid(x): return tuple(filter(lambda j: j[1] is not None, x))
    def transpose_tuple(x): return tuple(zip(*x))

    return tuple(map(lambda l: tuple(map(transpose_tuple, l)),
                     map(transpose_tuple,
                         map(filter_valid, zip(*x)))))


def blur2d(array, sigma, width):
    x = np.arange(-width, width + 1, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-width, width + 1, 1)
    xx, yy = np.meshgrid(x, y)
    return signal.convolve(array, np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)))[width:-width, width:-width]


def blurnd(array, sigma, width, dn=1):
    n = array.ndim
    coords = [np.arange(-width, width + 1, 1)]*n
    if isinstance(dn, (int, float)):
        coords = [c*dn for c in coords]
    else:
        coords = [c*d for c, d in zip(coords, dn)]
    grid = np.meshgrid(*coords)
    if isinstance(sigma, (int, float)):
        r2 = sum(x**2/(2*sigma**2) for x in grid)
    else:
        r2 = sum(x**2/(2*s**2) for x, s in zip(grid, sigma))
    gauss = np.exp(- r2) / np.sum(np.exp(- r2))
    return signal.convolve(array, gauss, mode='same')


def get_pol_angle(power_file, angle_file, plotting=False):
    def cos2(theta, delta, a, b): return a * np.cos((theta - delta) * np.pi / 90) + b

    angle = loadmat(angle_file)['angle'][0]
    p = next(v for k, v in loadmat(power_file).items() if not k.startswith('__'))[0]
    fit = curve_fit(cos2, angle, p, p0=[angle[p == max(p)][0] % 180, 1, 1],
                    bounds=(0, [180, np.inf, np.inf]))[0]
    if plotting:
        print(angle)
        plt.figure(power_file)
        plt.plot(angle, p)
        plt.plot(angle, cos2(angle, *fit))
    return fit[0]


def get_ell(power_file, angle_file, plotting=False):
    def cos2(theta, delta, a, b): return a * np.cos((theta - delta) * np.pi / 90) + b

    angle = loadmat(angle_file)['angle'][0]
    p = next(v for k, v in loadmat(power_file).items() if not k.startswith('__'))[0]
    fit = curve_fit(cos2, angle, p, p0=[angle[p == max(p)][0] % 180, 1, 1],
                    bounds=(0, [180, np.inf, np.inf]))[0]
    if plotting:
        print(angle)
        plt.figure(power_file)
        plt.plot(angle, p)
        plt.plot(angle, cos2(angle, *fit))
    return np.sqrt((fit[2]-fit[1])/(fit[1]+fit[2]))


def edges_to_centers(edges):
    centers = (edges[:-1] + edges[1:]) / 2
    return centers


def angular_average(angles, low=0, high=2 * np.pi, weights=None):
    adjusted = (angles - low) / (high - low) * (2 * np.pi)
    xavg = np.average(np.cos(adjusted), weights=weights)
    yavg = np.average(np.sin(adjusted), weights=weights)
    out = np.arctan2(yavg, xavg)
    return out / (2 * np.pi) * (high - low) + low


def get_peak_angles(hist, r, t, blurring=0, max_p=1, mode="mean"):
    tt, rr = np.meshgrid(t, r)
    rhist, r_edges = np.histogram(rr, weights=hist * rr, bins=len(r), range=[0, max_p], density=True)
    if blurring:
        rhist = blurnd(rhist, 0.03, 100, dn=r[1]-r[0])
    peaks = signal.find_peaks(rhist)[0]
    widths = np.asarray(signal.peak_widths(rhist, peaks)[0], dtype=int)
    for peak, width in itertools.compress(zip(peaks, widths), rhist[peaks] > 0.25):
        match mode:
            case "mean":
                mask = (np.abs(rr - r[peak]) < r[width] / 2)
                cm = angular_average(tt % np.pi, high=np.pi, weights=(hist * mask)**2)
                yield r[peak], cm, r[width]
            case "max":
                mask = (np.abs(rr - r[peak]) < r[width] / 10)
                thist,tedges = np.histogram(tt % np.pi, weights=mask*hist, bins=1000, range=[0,2*np.pi])
                yield r[peak], tedges[np.argmax(thist)], r[width]
            case "fourier":
                mask = (np.abs(rr - r[peak]) < r[width] / 10)
                thist,tedges = np.histogram(tt, weights=mask*hist, bins=1000, range=[0,2*np.pi])
                yield r[peak], -np.angle(fft.fft(fft.fftshift(thist))[2]) / 2 % np.pi, r[width]
            case _:
                raise NameError



def get_peak_profile(hist, rs, r, t,mode="mean"):
    tt, rr = np.meshgrid(t, r)
    dr = np.diff(rs)[0]
    for ri in rs:
        match mode:
            case "mean":
                mask = (np.abs(rr - ri) < dr / 2)
                cm = angular_average(tt % np.pi, high=np.pi, weights=(hist * mask)**1)
                yield ri, cm
            case "max":
                mask = (np.abs(rr - ri) < dr / 10)
                thist,tedges = np.histogram(tt % np.pi, weights=mask*hist, bins=1000, range=[0,2*np.pi])
                yield ri, tedges[np.argmax(thist)]
            case "fourier":
                mask = (np.abs(rr - ri) < dr / 10)
                thist,tedges = np.histogram(tt, weights=mask*hist, bins=1000, range=[0,2*np.pi])
                yield ri, -np.angle(fft.fft(fft.fftshift(thist))[2]) / 2 % np.pi
            case _:
                raise NameError


def get_profiles(data, i, n, pol=0., plotting=True, blurring=0, max_p=0.8,mode='mean'):
    print(f"{i}/{n}")
    px, py, pz = map(lambda x: x[i::n], data)

    print(f"{len(px)} Samples")

    pr, ptheta = np.sqrt(px ** 2 + py ** 2), np.arctan2(py, px)

    rt_hist, r_edges, t_edges = np.histogram2d(pr, ptheta, bins=256, range=[[0, max_p], [-np.pi, np.pi]])

    r = edges_to_centers(r_edges)
    if blurring:
        rt_hist = np.maximum(blurnd(rt_hist, blurring, 100, dn=(np.diff(r_edges)[0], np.diff(t_edges)[0])), 0)

    r_peak, theta_peak, width = zip(*get_peak_angles(rt_hist, edges_to_centers(r_edges),
                                    edges_to_centers(t_edges), blurring=blurring, max_p=max_p, mode=mode))

    rs = np.linspace(r_peak[0] - width[0] / 2, r_peak[0] + width[0] / 2, num=20)

    try:
        r_p, ang_p = zip(*get_peak_profile(rt_hist, rs, edges_to_centers(r_edges), edges_to_centers(t_edges)))
    except ValueError:
        print("No Peaks for intra")
        return (r_peak, theta_peak), (None, None)

    if plotting and i == 0:
        plot_diagnostics(px, py, r, rt_hist, r_p, ang_p, r_peak, theta_peak, pol, i, blurring, max_p=max_p)

    return (r_peak, theta_peak), (r_p, ang_p)


def plot_diagnostics(px, py, r, rt_hist, r_p, ang_p, r_peak, theta_peak, pol, i, blurring, max_p=1):
    plt.figure(f"{pol}: {i}")
    plt.subplot(221)
    plt.hist2d(px, py, bins=256, range=[[-max_p, max_p], [-max_p, max_p]], cmap=trans_jet())
    plt.subplot(212)
    plt.plot(r, blurnd(np.sum(rt_hist, 1) / max(np.sum(rt_hist, 1)), blurring, 100, dn=r[1] - r[0]))
    for rp in r_peak:
        plt.axvline(rp, color='r')
    plt.subplot(222)
    # plt.figure(f"{pol}")
    plt.imshow(rt_hist[:, ::int(-np.sign(pol))],
               extent=[-180, 180, 0, max_p],
               origin='lower',
               aspect='auto',
               cmap=trans_jet())
    plt.plot(unwrap(-np.sign(pol) * np.degrees(np.array(theta_peak)),start_between=(-45,135)), r_peak, color='m')
    # plt.plot(-np.sign(pol) * np.degrees(np.array(theta_peak)) - 180, r_peak, color='m')
    # plt.plot(-np.sign(pol) * np.degrees(np.array(ang_p)) - 180, r_p, color='r')
    plt.plot(unwrap(-np.sign(pol) * np.degrees(np.array(ang_p)),start_between=(-45,135)), r_p, color='k')


def get_angular_error(angles: list[list[float]]) -> tuple[float, float]:
    centers = [(np.mean(np.cos(theta)), np.mean(np.sin(theta))) for theta in angles]
    xs, ys = zip(*centers)
    cx = np.mean(xs)
    cy = np.mean(ys)
    error_matrix = np.cov(xs, ys)/len(xs)
    sample = qmc.MultivariateNormalQMC(mean=[cx, cy], cov=error_matrix).random(2048)
    sample_angles = np.arctan2(sample[:, 1], sample[:, 0])
    angle = scipy.stats.circmean(np.unwrap(sample_angles, discont=np.pi/2, period=np.pi))
    error = scipy.stats.circstd(np.unwrap(sample_angles, discont=np.pi/2, period=np.pi))
    return angle, error


def main(files=None,
         n=3,
         wdir=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613",
         calibrated=False,
         to_load=None,
         fig=None,
         mode='mean',
         blurring=(.005, 0.05)):

    if files is None:
        files = [('xe002_s', -0.1),
                 ('xe014_e', 0.2),
                 ('xe011_e', 0.3),
                 ('xe012_e', 0.5),
                 ('xe013_e', 0.6),
                 ('xe003_c', 0.9)]
    inter_lines = []
    intra_lines = []
    labels = []

    for name, pol in files:
        print(name)
        labels.append(pol)
        data = load_data(name, wdir, to_load, calibrated)

        def get_profiles_index(i): return get_profiles(data, i, n, pol=pol, plotting=True, blurring=blurring, mode=mode)

        profiles = list(map(get_profiles_index, range(n)))
        inter, intra = organize(profiles)
        r_inter, dr_inter = zip(*((np.mean(i), np.std(i) / np.sqrt(len(i))) for i in inter[0]))
        t_inter, dt_inter = zip(*(get_angular_error(i) for i in inter[1]))
        inter_lines.append(tuple(map(np.array, (r_inter, dr_inter, t_inter, dt_inter))))

        r_intra, dr_intra = zip(*((np.mean(i), np.std(i) / np.sqrt(len(i))) for i in intra[0]))
        t_intra, dt_intra = zip(*(get_angular_error(i) for i in intra[1]))
        intra_lines.append(tuple(map(np.array, (r_intra, dr_intra, t_intra, dt_intra))))

    # %% Plotting
    if not fig:
        f, ax = plt.subplots(2, 1, num='plots')
    else:
        f,ax=fig
    lines_colour_cycle = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    plt.sca(ax[0])
    # plt.figure()
    plt.title("inter_ring")
    for (r, dr, t, dt), l, c in zip(inter_lines, labels, lines_colour_cycle):
        t_adjust = (np.unwrap(t, period=np.pi) - 2 * np.pi) * -1 * np.sign(l) + np.pi * int(l < 0)
        plt.errorbar(r, np.degrees(t_adjust)%180, xerr=dr * 2, yerr=dt * 360 / np.pi,
                     label=l, linestyle='--', marker="o", markersize=3)
        # plt.plot(r, np.poly1d(np.polyfit(r,t_adjust,1,w=1/dt))(r), linestyle='', linewidth=1, color=c)
    plt.legend()
    plt.sca(ax[1])
    # plt.figure()
    plt.title("intra_ring")
    for (r, dr, t, dt), l in zip(intra_lines, labels):
        t_adjust = np.unwrap((t % np.pi - t[0] % np.pi) * -1 * np.sign(l), period=np.pi, discont=np.pi / 2)
        plt.errorbar(r, np.degrees(t_adjust), xerr=dr * 2, yerr=dt * 360 / np.pi, label=l)
    # plt.legend()
    plt.tight_layout()

    plt.figure("ati_order")
    for (r, dr, t, dt), l, c in zip(inter_lines, labels, lines_colour_cycle):
        t_adjust = (np.unwrap(t, period=np.pi) - 2 * np.pi) * -1 * np.sign(l) + np.pi * int(l < 0)
        plt.errorbar(range(len(r)), np.degrees(t_adjust)%180, xerr=dr * 2, yerr=dt * 360 / np.pi,
                     label=l, linestyle='--', marker="o", markersize=3)
    return f, ax


def load_data(name, wdir, to_load, calibrated):
    if not calibrated:
        angle = get_pol_angle(os.path.join(wdir, fr"Ellipticity measurements\{name}_power.mat"),
                              os.path.join(wdir, r"Ellipticity measurements\angle.mat")) + 4
        print(angle)
        return load_cv3(os.path.join(wdir, fr"clust_v3\{name}.cv3"),
                        pol=np.radians(angle), width=.05, to_load=to_load)
    else:
        with h5py.File(name) as f:
            if to_load is None:
                return (f["y"][()], f["x"][()], f["z"][()])
            else:
                return (f["y"][:to_load], f["x"][:to_load], f["z"][:to_load])



if __name__ == '__main__':
    plt.close("all")
    out = main(files=[('theory_03.h5', 0.3), ('theory_06.h5', 0.6)], wdir=r'C:\Users\mcman\Code\VMI', calibrated=True, n=5,mode='fourier')
    main([('xe011_e', 0.301), ('xe013_e', 0.601)], to_load=10000000, fig=out, wdir=r'C:\Users\mcman\Code\VMI', n=5,mode='fourier')
    print("DONE")


#%%
