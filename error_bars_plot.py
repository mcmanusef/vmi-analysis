#%% Initializing

import itertools
import os
from multiprocessing import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.stats
from scipy.io import loadmat
from scipy.optimize import curve_fit
from cv3_analysis import load_cv3
from scipy.stats import qmc

mpl.rc('image', cmap='jet')
mpl.use('Qt5Agg')
# %% Functions
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

def blurnd(array, sigma, width,dn=1):
    n = array.ndim
    coords = [np.arange(-width, width + 1, 1)]*n
    if isinstance(dn,(int, float)):
        coords = [c*dn for c in coords]
    else:
        coords = [c*d for c,d in zip(coords,dn)]
    grid = np.meshgrid(*coords)
    r2 = sum(map(lambda x: x**2, grid))
    gauss=np.exp(- r2 / (2 * sigma ** 2))/np.sum(np.exp(- r2 / (2 * sigma ** 2)))
    return signal.convolve(array, gauss,mode='same')

def get_pol_angle(power_file, angle_file, plotting=False):
    def cos2(theta, delta, a, b): return a * np.cos((theta - delta) * np.pi / 90) + b

    angle = loadmat(angle_file)['angle'][0]
    p = next(v for k, v in loadmat(power_file).items() if not k.startswith('__'))[0]
    fit = curve_fit(cos2, angle, p, p0=[angle[p == max(p)][0] % 180, 1, 1],
                    bounds=(0, [180, np.inf, np.inf]))[0]
    if plotting:
        plt.figure(power_file)
        plt.plot(angle,p)
        plt.plot(angle,cos2(angle,*fit))
    return fit[0]


def edges_to_centers(edges):
    centers = (edges[:-1] + edges[1:]) / 2
    return centers


def angular_average(angles, low=0, high=2 * np.pi, weights=None):
    adjusted = (angles - low) / (high - low) * (2 * np.pi)
    xavg = np.average(np.cos(adjusted), weights=weights)
    yavg = np.average(np.sin(adjusted), weights=weights)
    out = np.arctan2(yavg, xavg)
    return out / (2 * np.pi) * (high - low) + low


def get_peak_angles(hist, r, t, blurring=0):
    tt, rr = np.meshgrid(t, r)
    rhist, r_edges = np.histogram(rr, weights=hist * rr, bins=len(r), range=[0, 1], density=True)
    if blurring:
        rhist=blurnd(rhist,0.03,10,dn=r[1]-r[0])
    peaks = signal.find_peaks(rhist)[0]
    widths = np.asarray(signal.peak_widths(rhist, peaks)[0], dtype=int)
    proms = np.asarray(signal.peak_prominences(rhist, peaks)[0])
    for peak, width in itertools.compress(zip(peaks, widths), rhist[peaks] > 0.25):
                                          # np.logical_and(proms / rhist[peaks] > 0.1, rhist[peaks] > 0.25)):
        mask = (np.abs(rr - r[peak]) < r[width] / 2)
        if np.sum(hist * mask) > 0:
            cm = angular_average(tt % np.pi, high=np.pi, weights=(hist * mask)**2)
            yield r[peak], cm, r[width]


def get_peak_profile(hist, rs, r, t):
    tt, rr = np.meshgrid(t, r)
    dr = np.diff(rs)[0]
    for ri in rs:
        mask = (np.abs(rr - ri) < dr / 2)
        # print(hist*mask)
        if np.sum(hist * mask) > 0:
            with np.errstate(invalid='raise'):
                try:
                    cm = angular_average(tt % np.pi, high=np.pi, weights=(hist * mask) ** 0.5)
                except Exception as E:
                    with np.printoptions(threshold=np.inf):
                        print(hist * mask)
                    raise E
            # print(ri, cm)
            yield ri, cm


def get_profiles(data, i, n, pol=0., plotting=True, blurring=0):
    print(i)
    px, py, pz = map(lambda x: x[i::n], data)
    print(f"{len(px)} Samples")
    pr, ptheta = np.sqrt(px ** 2 + py ** 2), np.arctan2(py, px)

    rt_hist, r_edges, t_edges = np.histogram2d(pr, ptheta, bins=500, range=[[0, 1], [-np.pi, np.pi]])

    r = edges_to_centers(r_edges)
    if blurring:
        rt_hist = np.maximum(blurnd(rt_hist, blurring, 10, dn=(np.diff(r_edges)[0], np.diff(t_edges)[0])), 0)
    
    r_peak, theta_peak, width = zip(*get_peak_angles(rt_hist, edges_to_centers(r_edges), edges_to_centers(t_edges),blurring=blurring))

    rs = np.linspace(r_peak[0] - width[0] / 3, r_peak[0] + width[0] / 3, num=10)

    if plotting and i==0:
        plt.figure(f"{pol}: {i}")
        plt.subplot(221)
        plt.hist2d(px, py, bins=256, range=[[-1, 1], [-1, 1]])
        plt.subplot(212)
        plt.plot(r, blurnd(np.sum(rt_hist,1) / max(np.sum(rt_hist, 1)), blurring+0.0001, 10, dn=r[1]-r[0]))
        for rp in r_peak:
            plt.axvline(rp, color='r')
        plt.subplot(222)
        plt.imshow(rt_hist ** 0.5, extent=[-np.pi, np.pi, 0, 1], origin='lower', aspect='auto')
        plt.plot(np.array(theta_peak), r_peak)

    try:
        r_p, ang_p = zip(*get_peak_profile(rt_hist, rs, edges_to_centers(r_edges), edges_to_centers(t_edges)))
    except ValueError as E:
        print("No Peaks for intra")
        return (r_peak, theta_peak), (None, None)

    if plotting and i==0:
        plt.plot(np.array(ang_p), r_p)
        
    return (r_peak, theta_peak), (r_p, ang_p)

def get_angular_error(angles: list[list[float]]) -> tuple[float, float]:
    centers=[(np.mean(np.cos(theta)),np.mean(np.sin(theta))) for theta in angles]
    xs,ys=zip(*centers)
    cx=np.mean(xs)
    cy=np.mean(ys)
    error_matrix=np.cov(xs,ys)/len(xs)
    sample=qmc.MultivariateNormalQMC(mean=[cx, cy], cov=error_matrix).random(2048)
    sample_angles=np.arctan2(sample[:,1],sample[:,0])
    angle=scipy.stats.circmean(np.unwrap(sample_angles,discont=np.pi/2,period=np.pi))
    error=scipy.stats.circstd(np.unwrap(sample_angles,discont=np.pi/2,period=np.pi))
    return angle, error

# %% Calculations
if __name__ == '__main__':
    n = 5
    use=n
    wdir = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613"
    # inputs = [  ('xe003_c', 0.5)]
    inputs = [('xe002_s', -0.1), ('xe014_e', 0.2), ('xe011_e', 0.3), ('xe012_e', 0.5), ('xe013_e', 0.6), ('xe003_c', 0.9)]
    inter_lines = []
    intra_lines = []
    labels = []

    for name, pol in inputs:
        print(name)
        labels.append(pol)

        angle = get_pol_angle(os.path.join(wdir, fr"Ellipticity measurements\{name}_power.mat"),
                              os.path.join(wdir, r"Ellipticity measurements\angle.mat"))
        print(angle)
        data = load_cv3(os.path.join(wdir, fr"clust_v3\{name}.cv3"), pol=angle*-np.pi/180, width= .05)
        with Pool(4) as p:
            def get_profiles_index(i): return get_profiles(data, i, n, pol=pol, plotting=True, blurring=0.05)
            profiles=list(map(get_profiles_index, range(use)))
            inter, intra = organize(profiles)


        # def cm(i): return scipy.stats.circmean(np.asarray(i) % np.pi, high=np.pi)
        # def cs(i): return scipy.stats.circstd(np.asarray(i) % np.pi, high=np.pi)
        def fix(angles, cutoff=np.pi/2): return np.where(angles>cutoff, angles%np.pi-np.pi, angles)
        
        r_inter, dr_inter = zip(*((np.mean(i), np.std(i)/np.sqrt(len(i))) for i in inter[0]))
        t_inter, dt_inter = zip(*(get_angular_error(i) for i in inter[1]))
        inter_lines.append(tuple(map(np.array, (r_inter, dr_inter, t_inter, dt_inter))))

        r_intra, dr_intra = zip(*((np.mean(i), np.std(i)/np.sqrt(len(i))) for i in intra[0]))
        t_intra, dt_intra = zip(*(get_angular_error(i)  for i in intra[1]))
        intra_lines.append(tuple(map(np.array, (r_intra, dr_intra, t_intra, dt_intra))))
    # %% Plotting
    f, ax = plt.subplots(2, 1)
    lines_colour_cycle = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    plt.sca(ax[0])
    plt.title("inter_ring")
    
    for (r, dr, t, dt), l, c in zip(inter_lines, labels, lines_colour_cycle):
        t_adjust=(np.unwrap(t,period=np.pi)-2*np.pi)*-1*np.sign(l)+np.pi*int(l<0)
        plt.errorbar(r, t_adjust, xerr=dr*2, yerr=dt*2 , label=l,color=c, linestyle='--', marker="o", markersize=3)
        # plt.plot(r, np.poly1d(np.polyfit(r,t_adjust,1,w=1/dt))(r), linestyle='', linewidth=1, color=c)
        
    plt.legend()
    plt.sca(ax[1])
    plt.title("intra_ring")
    for (r, dr, t, dt), l in zip(intra_lines, labels):
        plt.errorbar(r, np.unwrap((t % np.pi - t[0] % np.pi)*-1*np.sign(l),period=np.pi,discont=np.pi/2), xerr=dr*2, yerr=dt*2, label=l)
    plt.tight_layout()
