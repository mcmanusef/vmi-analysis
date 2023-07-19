
import cluster_v3 as cv3
import h5py
import itertools as it
import functools as ft
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


def partial_mean(x, y):
    return ((x[0]*x[1]+y[0]*y[1])/(y[1]+x[1]), y[1]+x[1]) if x is not None and x[0] is not None else y


def list_accum(l, to_add, /, func=partial_mean, default=(None, 0)):
    try:
        l[to_add[0]] = func(l[to_add[0]], to_add[1])
        return l
    except IndexError:
        while len(l) < to_add[0]+1:
            l.append(default)
        print(f"Lengthened to {len(l)}")
        l[to_add[0]] = func(l[to_add[0]], to_add[1])
        return l


def monitor(a, b):
    if b[0] % 10000 == 0:
        print(b[0])
    return list_accum(a, b[1])


filename = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\air002.h5"
cutoff = 100
data_range = [0, 1e3]

with h5py.File(filename, mode='r') as f_in:
    (tdc_time, tdc_type, x, y, tot, toa) = cv3.iter_file(f_in)

    pulse_times = cv3.correct_pulse_times_iter(cv3.get_times_iter(tdc_time, tdc_type, 'pulse', cutoff))
    pixel_corr, t_pixel = cv3.split_iter(cv3.get_t_iter(pulse_times, toa), 2)

    formatted = ((tot_i, (t_i/1000, 1)) for tot_i, t_i in zip(tot, t_pixel) if 0 < t_i < 1e6)

    means, counts = zip(*ft.reduce(monitor, enumerate(it.islice(formatted, 5000000)), [(None, 0)]))

    x, y = tuple(map(np.asarray, zip(*filter(lambda x: x[1], enumerate(means)))))
    c = np.array(list(it.compress(counts, means)))

    y2 = np.array([a[0] for a in map(partial_mean, zip(gaussian_filter(y, 30), it.repeat(1)), zip(y, np.sqrt(c/x)))])
    fun = interp1d(x, y2-y2[-1], fill_value=0, bounds_error=False)

    plt.figure()
    plt.plot(x, y, color='blue')
    plt.fill_between(x, y*(1+c**-.5), y*(1-c**-.5), alpha=0.2, color='blue')
    # plt.plot(x, y2, color='red')
    plt.plot(np.arange(500), fun(np.arange(500)), color='red')
    plt.ylim(0, 500)
    plt.twinx(plt.gca())
    plt.plot(x, c, color='black')


with open("ToT_Correction.txt", mode='w') as f:
    print(0, file=f)
    for i in range(1, 1000):
        print(fun(i), file=f)
