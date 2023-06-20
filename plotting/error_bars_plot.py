# %% Initializing

import itertools
import os
import pickle

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
from scipy.stats import qmc

from cv3_analysis import load_cv3

mpl.rc('image', cmap='jet')
mpl.use('Qt5Agg')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


# %% Functions
def cache_results(func):
    def wrapper(*args, **kwargs):
        try:
            to_use=kwargs['use_cache']
            del kwargs['use_cache']
        except KeyError:
            to_use=True

        if not to_use:
            return func(*args, **kwargs)

        if not os.path.isdir('func_cache'):
            os.mkdir('func_cache')
        if func.__name__ in os.listdir("func_cache"):
            with open(os.path.join("func_cache", func.__name__, "dir.txt")) as f:
                for dir_string in [x for x in f.readlines() if str([args, kwargs]) in x]:
                    print("Loading from Cache")
                    file = dir_string.split("-->")[0]
                    with open(os.path.join("func_cache", func.__name__, file), 'rb') as f2:
                        return pickle.load(f2)
                print("Could not find saved results")
        else:
            os.mkdir(os.path.join("func_cache", func.__name__))
            with open(os.path.join("func_cache", func.__name__, "dir.txt"), 'w') as f:
                pass

        out = func(*args, **kwargs)

        filename = f"{hash(str([args, kwargs]))}.pickle"
        print(f"Saving to {filename}")
        with open(os.path.join("func_cache", func.__name__, "dir.txt"), 'a') as f:
            print(f"{filename}-->{str([args, kwargs])}", file=f)
        with open(os.path.join("func_cache", func.__name__, filename), 'wb') as f:
            pickle.dump(out, f)
        return out

    return wrapper


def trans_jet(opaque_point=0.1):
    # Create a colormap that looks like jet
    cmap = plt.cm.jet

    # Create a new colormap that is transparent at low values
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:int(cmap.N * opaque_point), -1] = np.linspace(0, 1, int(cmap.N * opaque_point))
    return ListedColormap(cmap_colors)


def wrap_between(data, low=0, high=180):
    return (data - low) % (high - low) + low


def unwrap(data, start_between=(-90, 90), period=180):
    data[0] = wrap_between(data[0], *start_between)
    return np.unwrap(data, period=period)


def pairwise(iterable):
    """Return an iterator that aggregates elements in pairs. Example: pairwise('ABCDEFG') --> AB BC CD DE EF FG"""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def organize(x):
    def filter_valid(y): return tuple(filter(lambda j: j[1] is not None, y))

    def transpose_tuple(y): return tuple(zip(*y))

    return tuple(map(lambda l: tuple(map(transpose_tuple, l)),
                     map(transpose_tuple,
                         map(filter_valid, zip(*x)))))


def blur2d(array, sigma, width):
    x = np.arange(-width, width + 1, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-width, width + 1, 1)
    xx, yy = np.meshgrid(x, y)
    return signal.convolve(array, np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)))[width:-width, width:-width]


def blurnd(array, sigma, width, dn=None):
    if not width:
        return array
    n = array.ndim
    coords = [np.arange(-width, width + 1, 1)] * n
    if isinstance(dn, (int, float)):
        coords = [c * dn for c in coords]
    else:
        coords = [c * d for c, d in zip(coords, dn)]
    grid = np.meshgrid(*coords)
    if isinstance(sigma, (int, float)):
        r2 = sum(x ** 2 / (2 * sigma ** 2) for x in grid)
    else:
        r2 = sum(x ** 2 / (2 * s ** 2) for x, s in zip(grid, sigma))
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
    return np.sqrt((fit[2] - fit[1]) / (fit[1] + fit[2]))


def edges_to_centers(edges):
    centers = (edges[:-1] + edges[1:]) / 2
    return centers


def angular_average(angles, low=0, high=2 * np.pi, weights=None):
    adjusted = (angles - low) / (high - low) * (2 * np.pi)
    xavg = np.average(np.cos(adjusted), weights=weights)
    yavg = np.average(np.sin(adjusted), weights=weights)
    out = np.arctan2(yavg, xavg)
    return out / (2 * np.pi) * (high - low) + low


def get_peak_angles(hist, r, t, blurring=0, max_p=1., mode="mean"):
    tt, rr = np.meshgrid(t, r)
    rhist, r_edges = np.histogram(rr, weights=hist * rr, bins=len(r), range=[0, max_p], density=True)
    if blurring:
        rhist = blurnd(rhist, 0.03, 100, dn=r[1] - r[0])
    peaks = signal.find_peaks(rhist)[0]
    widths = np.asarray(signal.peak_widths(rhist, peaks)[0], dtype=int)
    for peak, width in itertools.compress(zip(peaks, widths), rhist[peaks] > 0.25):
        match mode:
            case "mean":
                mask = (np.abs(rr - r[peak]) < r[width] / 2)
                cm = angular_average(tt % np.pi, high=np.pi, weights=(hist * mask) ** 2)
                yield r[peak], cm, r[width]
            case "max":
                mask = (np.abs(rr - r[peak]) < r[width] / 10)
                thist, tedges = np.histogram(tt % np.pi, weights=mask * hist, bins=1000, range=[0, 2 * np.pi])
                yield r[peak], tedges[np.argmax(thist)], r[width]
            case "fourier":
                mask = (np.abs(rr - r[peak]) < r[width] / 10)
                thist, tedges = np.histogram(tt, weights=mask * hist, bins=1000, range=[0, 2 * np.pi])
                yield r[peak], -np.angle(fft.fft(fft.fftshift(thist))[2]) / 2 % np.pi, r[width]
            case _:
                raise NameError


def get_peak_profile(hist, rs, r, t, mode="mean"):
    tt, rr = np.meshgrid(t, r)
    dr = np.diff(rs)[0]
    for ri in rs:
        match mode:
            case "mean":
                mask = (np.abs(rr - ri) < dr / 2)
                cm = angular_average(tt % np.pi, high=np.pi, weights=(hist * mask) ** 1)
                yield ri, cm
            case "max":
                mask = (np.abs(rr - ri) < dr / 10)
                thist, tedges = np.histogram(tt % np.pi, weights=mask * hist, bins=1000, range=[0, 2 * np.pi])
                yield ri, tedges[np.argmax(thist)]
            case "fourier":
                mask = (np.abs(rr - ri) < dr / 10)
                thist, tedges = np.histogram(tt, weights=mask * hist, bins=1000, range=[0, 2 * np.pi])
                yield ri, -np.angle(fft.fft(fft.fftshift(thist))[2]) / 2 % np.pi
            case _:
                raise NameError


def get_profiles(data, i, n, pol=0., plotting=True, blurring=0, max_p=0.6, mode='mean', label="", num=11):
    print(f"{i}/{n}")
    px, py, pz = map(lambda x: x[i::n], data)

    print(f"{len(px)} Samples")

    pr, ptheta = np.sqrt(px ** 2 + py ** 2), np.arctan2(py, px)

    rt_hist, r_edges, t_edges = np.histogram2d(pr, ptheta, bins=256, range=[[0, max_p], [-np.pi, np.pi]])

    r = edges_to_centers(r_edges)
    if blurring:
        rt_hist = np.maximum(blurnd(rt_hist, blurring, 100, dn=(np.diff(r_edges)[0], np.diff(t_edges)[0])), 0)

    r_peak, theta_peak, width = zip(*get_peak_angles(rt_hist, edges_to_centers(r_edges),
                                                     edges_to_centers(t_edges), blurring=blurring, max_p=max_p,
                                                     mode=mode))

    rs = np.linspace(r_peak[0] - width[0] / 2, r_peak[0] + width[0] / 2, num=num)

    try:
        r_p, ang_p = zip(*get_peak_profile(rt_hist, rs, edges_to_centers(r_edges), edges_to_centers(t_edges)))
    except ZeroDivisionError:
        print("No Peaks for intra")
        return (r_peak, theta_peak), (None, None)

    if plotting and i == 0:
        plot_diagnostics(px, py, r, rt_hist, r_p, ang_p, r_peak, theta_peak, pol, blurring, max_p=max_p, label=label)

    return (r_peak, theta_peak), (r_p, ang_p)


def plot_diagnostics(px, py, r, rt_hist, r_p, ang_p, r_peak, theta_peak, pol, blurring, max_p=1., label=""):
    plt.figure(f"{pol} {label}")
    plt.subplot(221)
    plt.hist2d(px, py, bins=256, range=[[-max_p, max_p], [-max_p, max_p]], cmap=trans_jet())

    x_peak, y_peak = r_peak * np.cos(np.array(theta_peak) % np.pi), r_peak * np.sin(np.asarray(theta_peak) % np.pi)
    plt.scatter(x_peak, y_peak, color='r')

    plt.subplot(212)
    plt.plot(r, blurnd(np.sum(rt_hist, 1) / max(np.sum(rt_hist, 1)), blurring, 100, dn=r[1] - r[0]))
    for rp in r_peak:
        plt.axvline(rp, color='r')
    plt.subplot(222)
    plt.imshow(rt_hist[:, ::int(-np.sign(pol))],
               extent=[-180, 180, 0, max_p],
               origin='lower',
               aspect='auto',
               cmap=trans_jet())
    plt.plot(unwrap(-np.sign(pol) * np.degrees(np.array(theta_peak)), start_between=(-45, 135)), r_peak, color='m')
    plt.plot(unwrap(-np.sign(pol) * np.degrees(np.array(ang_p)), start_between=(-45, 135)), r_p, color='k')


def get_angular_error(angles: list[list[float]]) -> tuple[float, float]:
    centers = [(np.mean(np.cos(theta)), np.mean(np.sin(theta))) for theta in angles]
    xs, ys = zip(*centers)
    cx = np.mean(xs)
    cy = np.mean(ys)
    error_matrix = np.cov(xs, ys) / len(xs)

    try:
        sample = qmc.MultivariateNormalQMC(mean=[cx, cy], cov=error_matrix).random(2048)
        sample_angles = np.arctan2(sample[:, 1], sample[:, 0])
        angle = scipy.stats.circmean(np.unwrap(sample_angles, discont=np.pi / 2, period=np.pi))
        error = scipy.stats.circstd(np.unwrap(sample_angles, discont=np.pi / 2, period=np.pi))
        return angle, error
    except ValueError as e:
        print(error_matrix)
        print(e)
        return np.arctan2(cy, cx), 0


def main(files,
         n=3,
         wdir=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613",
         calibrated=False,
         to_load=None,
         fig=None,
         mode='mean',
         blurring=(.005, 0.05),
         electrons='all',
         label="", symmetrize=True,
         max_error=None,
         send_data=False,
         **kwargs):
    inter_lines = []
    intra_lines = []
    labels = []
    data_list = []

    for name, pol in files:
        print(name)
        labels.append(pol)
        data = load_data(name, wdir, to_load, calibrated, electrons=electrons, symmetrize=symmetrize, use_cache=(not calibrated))
        if send_data:
            data_list.append(data)

        def get_profiles_index(i):
            return get_profiles(data, i, n, pol=pol, plotting=True, blurring=blurring, mode=mode, label=label, num=16)

        profiles = list(map(get_profiles_index, range(n)))
        inter, intra = organize(profiles)
        r_inter, dr_inter = zip(*((np.mean(i), np.std(i) / np.sqrt(len(i))) for i in inter[0]))
        t_inter, dt_inter = zip(*(get_angular_error(i) for i in inter[1]))
        inter_lines.append(tuple(map(np.array, (r_inter, dr_inter, t_inter, dt_inter))))

        r_intra, dr_intra = zip(*((np.mean(i), np.std(i) / np.sqrt(len(i))) for i in intra[0]))
        t_intra, dt_intra = zip(*(get_angular_error(i) for i in intra[1]))
        intra_lines.append(tuple(map(np.array, (r_intra, dr_intra, t_intra, dt_intra))))

    if send_data:
        return data_list, inter_lines, intra_lines

    # %% Plotting
    if not fig:
        f, ax = plt.subplots(2, 1, num='plots', figsize=(6, 10))
    else:
        f, ax = fig
    lines_colour_cycle = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    marker, linestyle = ("", "-") if calibrated else ("+", ":")
    plt.sca(ax[0])
    # plt.figure()
    plt.title("inter_ring")
    for (r, dr, t, dt), l, c in zip(inter_lines, labels, lines_colour_cycle):
        print(marker, linestyle)
        t_adjust = unwrap(-np.degrees(t), start_between=(0, 180))
        plt.errorbar(r, t_adjust, xerr=dr * 2, yerr=dt * 360 / np.pi,
                     label=f"{l}: {label}",
                     linestyle=linestyle, marker=marker, markersize=10, color=c)

    ax[0].set_xlabel(r"$p_r$")
    ax[0].set_ylabel(r"$\theta$")

    # plt.plot(r, np.poly1d(np.polyfit(r,t_adjust,1,w=1/dt))(r), linestyle='', linewidth=1, color=c)
    plt.legend()
    plt.sca(ax[1])
    # plt.figure()
    plt.title("intra_ring")
    for (r, dr, t, dt), l, (r_inter, _, t_inter, _), c in zip(intra_lines, labels, inter_lines, lines_colour_cycle):
        if max_error is not None:
            r, dr, t, dt = tuple(
                map(np.asarray, zip(*(filter(lambda x: x[3] < np.radians(max_error), zip(r, dr, t, dt))))))

        t_adjust = unwrap((t % np.pi - t_inter[0] % np.pi) * -1 * np.sign(l), period=np.pi, start_between=[0, np.pi])

        plt.errorbar(r - r_inter[0], np.degrees(t_adjust), xerr=dr * 2, yerr=dt * 180 / np.pi,
                     label=f"{l}: {label}",
                     color=c, linestyle=linestyle, marker=marker)
    ax[1].set_xlabel(r"$\Delta p_r$")
    ax[1].set_ylabel(r"$\Delta \theta$")
    # plt.legend()
    plt.tight_layout()

    f,(ax,ax2)=plt.subplots(2,1, sharex=True, num="ati_order",  height_ratios=[2,1])

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(-10, 10)  # outliers only
    ax.set_ylim(70, 110)
    ax.axhline(90, linestyle=":", color='silver')
    ax2.axhline(0, linestyle=":", color='silver')
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - 2*d, 1 + 2*d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - 2*d, 1 + 2*d), **kwargs)

    for (r, dr, t, dt), l, c in zip(inter_lines, labels, lines_colour_cycle):
        t_adjust = unwrap(t * -1 * np.sign(l), period=np.pi, start_between=[0,np.pi])
        ax.errorbar(range(1,len(r)+1), np.degrees(t_adjust), xerr=dr, yerr=dt * 180 / np.pi,
                     label=f"ε={np.abs(l)}",
                     linestyle=linestyle, marker=marker, markersize=3)
        ax2.errorbar(range(1,len(r)+1), np.degrees(t_adjust), xerr=dr, yerr=dt * 180 / np.pi,
                     label=f"ε={np.abs(l)}",
                     linestyle=linestyle, marker=marker, markersize=3)
    plt.gca().set_xlabel(r"ATI order")
    ax.set_ylabel(r"$\theta$",y=80)
    plt.xticks([1,2,3])
    plt.sca(ax)
    plt.legend()
    plt.tight_layout()

    plt.figure("Slopes")
    for (r, dr, t, dt), l, (r_inter, _, t_inter, _), c in zip(intra_lines, labels, inter_lines, lines_colour_cycle):
        if max_error is not None:
            r, dr, t, dt = tuple(
                map(np.asarray, zip(*(filter(lambda x: x[3] < np.radians(max_error), zip(r, dr, t, dt))))))

        t_adjust = np.unwrap((t % np.pi - t_inter[0] % np.pi) * -1 * np.sign(l), period=np.pi, discont=np.pi / 2)

        plt.plot(r[:-1], np.diff(np.degrees(t_adjust))/np.diff(r),
                     label=f"{l}: {label}",
                     color=c, linestyle=linestyle, marker=marker)
    plt.legend()
    plt.gca().set_xlabel(r"$\Delta p_r$")
    plt.gca().set_ylabel(r"$d\theta/dr$")
    plt.tight_layout()

    plt.figure("Absolute r")
    for (r, dr, t, dt), l, (r_inter, _, t_inter, _), c in zip(intra_lines, labels, inter_lines, lines_colour_cycle):
        if max_error is not None:
            r, dr, t, dt = tuple(
                map(np.asarray, zip(*(filter(lambda x: x[3] < np.radians(max_error), zip(r, dr, t, dt))))))

        t_adjust = np.unwrap((t % np.pi - t_inter[0] % np.pi) * -1 * np.sign(l), period=np.pi, discont=np.pi / 2)

        plt.errorbar(r, np.degrees(t_adjust), xerr=dr * 2, yerr=dt * 180 / np.pi,
                     label=f"{l} {label}",
                     color=c, linestyle=linestyle, marker=marker)
    plt.gca().set_xlabel(r"$p_r$")
    plt.gca().set_ylabel(r"$\Delta \theta$")
    plt.legend()
    plt.tight_layout()

    plt.figure("Dispersion")
    for (r, dr, t, dt), l, (r_inter, _, t_inter, _), c in zip(intra_lines, labels, inter_lines, lines_colour_cycle):
        if max_error is not None:
            r, dr, t, dt = tuple(
                map(np.asarray, zip(*(filter(lambda x: x[3] < np.radians(max_error), zip(r, dr, t, dt))))))

        t_adjust = np.unwrap((t % np.pi - t_inter[0] % np.pi) * -1 * np.sign(l), period=np.pi, discont=np.pi / 2)

        plt.errorbar(np.degrees(t_adjust), r**2/2,  xerr=dt * 180 / np.pi, yerr=dr * 2 * r ,
                     label=f"{l}: {label}",
                     color=c, linestyle=linestyle, marker=marker)
    plt.gca().set_ylabel(r"$E$")
    plt.gca().set_xlabel(r"$\Delta \theta$")
    plt.legend()
    plt.tight_layout()
    return f, ax


@cache_results
def load_data(name, wdir, to_load, calibrated, electrons="all", symmetrize=True, **kwargs):
    print("Unused:", kwargs)
    if not calibrated:
        angle = get_pol_angle(os.path.join(wdir, fr"Ellipticity measurements\{name}_power.mat"),
                              os.path.join(wdir, r"Ellipticity measurements\angle.mat")) + 4
        print(angle)
        if not symmetrize:
            return load_cv3(os.path.join(wdir, fr"clust_v3\{name}.cv3"),
                            pol=float(np.radians(angle)), width=.05, to_load=to_load, electrons=electrons)
        else:
            px, py, pz = load_cv3(os.path.join(wdir, fr"clust_v3\{name}.cv3"),
                                  pol=float(np.radians(angle)), width=.05, to_load=to_load, electrons=electrons)
            return tuple(map(np.array, ([*px, *(-px)], [*py, *(-py)], [*pz, *(-pz)])))
    else:
        with h5py.File(os.path.join(wdir, name)) as f:
            if to_load is None:
                return f["y"][()], f["x"][()], f["z"][()]
            else:
                return f["y"][:to_load], f["x"][:to_load], f["z"][:to_load]


if __name__ == '__main__':
    plt.close("all")
    # out = main(files=[('theory_03.h5', 0.3), ('theory_06.h5', 0.6)],
    #            wdir=r'C:\Users\mcman\Code\VMI\Data',
    #            calibrated=True,
    #            n=3,
    #            mode='fourier',
    #            label='theory'
    #            )
    #
    # out = main([('xe011_e', 0.3), ('xe013_e', 0.6)],
    #            to_load=None,
    #            fig=out,
    #            wdir=r'C:\Users\mcman\Code\VMI\Data',
    #            n=3,
    #            mode='fourier',
    #            electrons='down',
    #            label="experiment"
    #            )
    # #
    # files = [
    #     # ('xe002_s', -0.1),
    #     ('xe014_e', 0.2),
    #     ('xe011_e', 0.3),
    #     ('xe012_e', 0.5),
    #     ('xe013_e', 0.6),
    #     # ('xe003_c', 0.9),
    # ]
    # main(files,
    #      to_load=None,
    #      blurring=(.005, 0.05),
    #      # fig=out,
    #      wdir=r'C:\Users\mcman\Code\VMI\Data',
    #      n=3,
    #      mode='mean',
    #      electrons='down',
    #      symmetrize=True,
    #      label="",
    #      max_error=5
    #      )
    # f=plt.figure("ATI order")
    # plt.axhline(90,linestyle=':', c='k')

    #
    files = [
        ('theory_03_0.h5', 22),
        # ('theory_H_0.h5', 2)
        ('theory_03_1.h5', 23),
        ('theory_03_2.h5', 24),
        ('theory_03_3.h5', 25),
        ('theory_03_4.h5', 26),
        ('theory_03_5.h5', 27),

    ]

    main(files,
          wdir=r'C:\Users\mcman\Code\VMI\Data',
          calibrated=True,
          n=3,
          mode='fourier',
          label='TW/cm$^2$',
          )


    print("DONE")

#%%
