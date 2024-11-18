import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from matplotlib.backends.backend_pdf import PdfPages

from Old import coincidence_v4
import minor_utils.autoconvolve
from Old.calibrations import calibration_20240208
from plotting.plotting_utils import dp_filter, filter_coords


def analyze_to_pdf(file, output, dead_pixels=None, center=(123.48, 131.01, 749053.403), angle=0, calibration=None,
                   p_range=(-0.6, 0.6), polarization_angle=0.0, hole_coords=(0, 0), hole_radius=0.0):
    # Generates coincidence and non-coincidence plots for a given file, saving to pdf
    plt.close("all")
    if dead_pixels is None:
        dead_pixels = []
    coincidence_data = coincidence_v4.load_file(file)
    print(f"Loaded {file}")
    with PdfPages(output) as pdf:
        x, y, toa, etof, itof = coincidence_data
        print(f"x: {len(x)}, y: {len(y)}, toa: {len(toa)}, etof: {len(etof)}, itof: {len(itof)}")
        # Basic Preprocessing
        t = etof + 0.26 * np.random.random_sample(len(etof))
        x, y, toa, t, itof = filter_coords([x, y, toa, t, itof], [(0, 256), (0, 256), (0, 1e6), (0, 1e6), (0, 1e6)])
        if dead_pixels:
            x, y, toa, t, itof = dp_filter(dead_pixels, x, y, toa, t, itof)

        # Zoomed out ToF plots
        print("Calculating e-ToF Histogram")
        full_tof = plt.figure("Raw ToF Spectra (Full Range)", figsize=(12, 10))

        full_tof.add_subplot(311)
        etof_spectrum, etof_edges = np.histogram(etof, bins=1_000_000, range=[0, 1e6], density=True)
        etof_spectrum = etof_spectrum / (etof_edges[-1] - etof_edges[0])
        plt.plot(etof_edges[:-1], etof_spectrum)
        plt.xlabel("e-ToF (ns)")
        plt.ylabel("Counts (Normalized)")
        plt.title("Raw e-ToF Spectrum (Full Range)")
        print("Finding e-ToF peaks")
        etof_peak_index, etof_peak_properties = scipy.signal.find_peaks(etof_spectrum,
                                                                        prominence=0.5 * np.max(etof_spectrum))
        etof_peak_widths, etof_peak_width_heights, *_ = scipy.signal.peak_widths(etof_spectrum, etof_peak_index)
        print(f"Peaks: {etof_peak_index}, Widths: {etof_peak_widths}")
        plt.vlines(etof_edges[etof_peak_index], 0, max(etof_spectrum), colors='r', linestyles='dashed')
        plt.hlines(etof_peak_width_heights,
                   [etof_edges[epi - int(epw / 2)] for epi, epw in zip(etof_peak_index, etof_peak_widths)],
                   [etof_edges[epi + int(epw / 2)] for epi, epw in zip(etof_peak_index, etof_peak_widths)],
                   colors='r', linestyles='dashed')

        full_tof.add_subplot(312)
        itof_spectrum, itof_edges = np.histogram(itof, bins=100000, range=[0, 1e6], density=True)
        itof_spectrum = itof_spectrum / (itof_edges[-1] - itof_edges[0])
        plt.plot(itof_edges[:-1], itof_spectrum)
        plt.xlabel("i-ToF")
        plt.ylabel("Counts")
        plt.title("Raw i-ToF Spectrum (Full Range)")

        full_tof.add_subplot(313)
        toa_spectrum, toa_edges = np.histogram(toa, bins=100000, range=[0, 1e6], density=True)
        toa_spectrum = toa_spectrum / (toa_edges[-1] - toa_edges[0])
        plt.plot(toa_edges[:-1], toa_spectrum)
        toa_peak_index, toa_peak_properties = scipy.signal.find_peaks(toa_spectrum,
                                                                      prominence=0.5 * np.max(toa_spectrum))
        toa_peak_widths, toa_peak_width_heights, toa_le, toa_re = scipy.signal.peak_widths(toa_spectrum, toa_peak_index)
        plt.vlines(toa_edges[toa_peak_index], 0, max(toa_spectrum), colors='r', linestyles='dashed')
        plt.xlabel("ToA")
        plt.ylabel("Counts")
        plt.title("Raw ToA Spectrum (Full Range)")
        plt.tight_layout()
        pdf.savefig()

        # Filtering to the largest etof peak

        max_peak = np.argmax(etof_peak_properties['prominences'])
        etof_range = (etof_edges[etof_peak_index[max_peak]] - 30,
                      etof_edges[etof_peak_index[max_peak]] + 30)
        print(f"e-ToF range: {etof_range}")
        itof_range = (etof_range[0], etof_range[0] + 3e4)

        toa_max_peak = np.argmax(toa_peak_properties['prominences'])
        toa_range = (toa_edges[toa_peak_index[toa_max_peak]] - 100,
                     toa_edges[toa_peak_index[toa_max_peak]] + 100)
        print(f"{toa_range=}")
        print(f"Mean ToA: {np.mean(toa)}")
        print(f"ToA Sample: {toa[:10]}")
        print(f"Toa count in range: {np.sum((toa > toa_range[0]) & (toa < toa_range[1]))}")

        x, y, t, itof, toa = filter_coords([x, y, t, itof, toa],
                                           [(0, 256), (0, 256), etof_range, itof_range, toa_range])

        # Zoomed in ToF plots

        kernel_size = 10000
        gaussian_kernel = scipy.signal.windows.gaussian(kernel_size * 5, std=kernel_size)
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        mean = np.mean(t)
        try:
            center=scipy.io.loadmat(output.replace(".pdf", ".mat"))['center'].flatten()
        except FileNotFoundError:
            pass
        except KeyError:
            pass
        adjust = center[2] - mean
        t = t - scipy.signal.fftconvolve(t - mean, gaussian_kernel, mode='same', ) + adjust
        t_range = (etof_range[0] + adjust, etof_range[1] + adjust)
        print(f"t range: {t_range}, mean: {mean}, adjusted mean: {np.mean(t)}")

        tof_zoomed = plt.figure("Raw ToF Spectra (Zoomed)", figsize=(12, 10))
        tof_zoomed.add_subplot(311)
        t_hist, t_edges, _ = plt.hist(t, bins=1000, range=t_range, density=True)
        plt.xlabel("e-ToF (ns)")
        plt.ylabel("Counts (Normalized)")
        plt.title("Raw e-ToF Spectrum (Zoomed)")
        smoothed_t_hist = scipy.signal.savgol_filter(t_hist, 25, 3)
        highly_smoothed_t_hist = scipy.signal.savgol_filter(t_hist, 300, 3)
        plt.plot(t_edges[:-1], smoothed_t_hist, linestyle='dashed', color='k')
        plt.plot(t_edges[:-1], highly_smoothed_t_hist, linestyle='dotted', color='k')
        t_peak_index, t_peak_properties = scipy.signal.find_peaks(smoothed_t_hist, prominence=0.01 * np.max(t_hist))
        t_peak_widths, t_peak_width_heights, t_le, t_re = scipy.signal.peak_widths(smoothed_t_hist, t_peak_index,
                                                                                   rel_height=0.9)
        plt.vlines(t_edges[t_peak_index], 0, smoothed_t_hist[t_peak_index], colors='r', linestyles='dashed',
                   linewidth=1.5)
        plt.hlines(t_peak_width_heights,
                   [t_edges[round(ile)] for ile in t_le],
                   [t_edges[round(ire)] for ire in t_re],
                   colors='r', linestyles='dashed')
        tof_zoomed.add_subplot(312)
        filtered_itof_hist, filtered_itof_edges, _ = plt.hist(itof, bins=3000, range=itof_range, density=True)
        plt.xlabel("i-ToF (ns)")
        plt.ylabel("Counts (Normalized)")
        plt.title("Raw i-ToF Spectrum (Zoomed)")
        smoothed_itof_hist = scipy.signal.savgol_filter(filtered_itof_hist, 25, 3)
        plt.plot(filtered_itof_edges[:-1], smoothed_itof_hist, linestyle='dashed', color='k', linewidth=1.5)

        itof_peak_index, itof_peak_properties = scipy.signal.find_peaks(
            smoothed_itof_hist, prominence=0.1 * np.max(filtered_itof_hist))
        itof_peak_widths, itof_peak_width_heights, itof_le, itof_re = scipy.signal.peak_widths(smoothed_itof_hist,
                                                                                               itof_peak_index,
                                                                                               rel_height=0.9)

        print(itof_peak_index, itof_peak_widths)
        plt.vlines(filtered_itof_edges[itof_peak_index], 0, smoothed_itof_hist[itof_peak_index], colors='r',
                   linestyles='dashed', linewidth=1.5)
        plt.hlines(itof_peak_width_heights,
                   [filtered_itof_edges[round(ile)] for ile in itof_le],
                   [filtered_itof_edges[round(ire)] for ire in itof_re],
                   colors='r', linestyles='dashed')

        tof_zoomed.add_subplot(313)
        toa_hist, toa_edges, _ = plt.hist(toa, bins=1000, range=toa_range, density=True)
        plt.xlabel("ToA (ns)")
        plt.ylabel("Counts (Normalized)")
        plt.title("Raw ToA Spectrum (Zoomed)")

        plt.tight_layout()
        pdf.savefig()

        # ToF-ToA plot
        tof_toa = plt.figure("ToF-ToA", figsize=(12, 10))
        ax = tof_toa.add_subplot(111)
        ax.hist2d(t, toa, bins=1000, range=[t_range, toa_range], cmap='jet')
        plt.xlabel("ToF (ns)")
        plt.ylabel("ToA (ns)")
        plt.title("ToF-ToA")
        # individual line histograms on the sides
        ax2 = ax.twinx()
        h, *_ = ax2.hist(t, bins=1000, range=t_range, color='r', histtype='step', orientation='vertical')
        plt.ylim(0, np.max(h) * 10)
        ax3 = ax.twiny()
        h, *_ = ax3.hist(toa, bins=1000, range=toa_range, color='r', histtype='step', orientation='horizontal')
        plt.xlim(0, np.max(h) * 10)
        plt.tight_layout()
        pdf.savefig()

        etof_range = (etof_edges[etof_peak_index[max_peak]] - 30,
                      etof_edges[etof_peak_index[max_peak]] + 30)
        print(f"e-ToF range: {etof_range}")

        x, y, t, itof = filter_coords([x, y, t, itof], [(0, 256), (0, 256), etof_range, itof_range])

        print(f"Filtered {file}: {len(x)}")
        # Derivative plots of t
        plt.figure("Peak Analysis", figsize=(12, 10))
        plt.subplot(211)
        plt.plot(t_edges[:-1], smoothed_t_hist)
        t_p = scipy.signal.savgol_filter(np.gradient(smoothed_t_hist, t_edges[:-1]), 45, 3)
        t_pp = scipy.signal.savgol_filter(np.gradient(t_p, t_edges[:-1]), 25, 3)
        peakiness = -np.minimum(0, t_pp)
        peak_index = find_peaks(peakiness)
        [plt.axvline(t_edges[peak], color='r', linestyle='dashed', ) for peak in peak_index]
        [plt.text(t_edges[peak], 0, f"{i}", fontsize=12) for i, peak in enumerate(peak_index)]
        plt.text(0.1, 0.6,
                 f"Peak times:\n" + "\n".join([f"{i}: {t_edges[peak]:.2f} ns" for i, peak in enumerate(peak_index)]),
                 fontsize=12, transform=plt.gca().transAxes)
        plt.twinx(plt.gca())
        plt.plot(t_edges[:-1], -np.minimum(0, t_pp), linestyle='dotted', color='k')

        plt.subplot(212)
        xf, yf, tf = filter_coords([x, y, t], [(0, 256), (0, 256), (center[2] - 1, center[2] + 1)])
        r = np.sqrt((xf - center[0]) ** 2 + (yf - center[1]) ** 2)
        r_hist, r_edges, *_ = np.histogram(r, bins=1000, range=(0, 256), density=True)
        r_smoothed = scipy.signal.savgol_filter(r_hist, 25, 3)
        plt.plot(r_edges[:-1], r_smoothed, linestyle='dashed', color='k')
        r_p = scipy.signal.savgol_filter(np.gradient(r_smoothed, r_edges[:-1]), 25, 3)
        r_pp = scipy.signal.savgol_filter(np.gradient(r_p, r_edges[:-1]), 25, 3)
        peakiness = -np.minimum(0, r_pp)
        peak_index = find_peaks(peakiness)
        [plt.axvline(r_edges[peak], color='r', linestyle='dashed', ) for peak in peak_index]
        [plt.text(r_edges[peak], 0, f"{i}", fontsize=12) for i, peak in enumerate(peak_index)]
        plt.twinx(plt.gca())
        plt.plot(r_edges[:-1], -np.minimum(0, r_pp), linestyle='dotted', color='k')
        plt.text(0.9, 0.6, f"Peak Radii:\n" + "\n".join(
            [f"{i}: {r_edges[peak]:.2f} pixels" for i, peak in enumerate(peak_index)]), fontsize=12,
                 transform=plt.gca().transAxes)
        plt.tight_layout()
        pdf.savefig()

        center = minor_utils.autoconvolve.find_center(x, y, t, t_range, calibration=calibration, n=512)
        try:
            center=scipy.io.loadmat(output.replace(".pdf", ".mat"))['center'].flatten()
            print(f"Loaded center from file: {center=}")
        except FileNotFoundError:
            pass
        except KeyError:
            pass


        # X-Y-T plots
        xyt_full = plt.figure("X-Y-T (Full)", figsize=(12, 10))
        xyt_plot(x, y, t, itof, t_range, itof_range, itof_range, xyt_full, mark_center=center)
        print(f"{center=}")
        # center=(120.5,134.5,749055.4)
        # plt.sca(xyt_full.get_axes()[0])
        plt.tight_layout()
        pdf.savefig()
        if calibration:
            print("Calibrating")
            px, py, pz = fix_hole_calibrated(x, y, t, center, angle, calibration)
            py, pz = coincidence_v4.rotate_data(py, pz, polarization_angle)
            xyt_calibrated = plt.figure("Calibrated Momentum", figsize=(12, 10))
            calibrated_plot(px, py, pz, p_range, xyt_calibrated)
            plt.tight_layout()
            pdf.savefig()
            p_plane_slice = filter_coords([px, py, pz], [(-0.03, 0.03), p_range, p_range])
            calibrated_plot(*p_plane_slice, p_range, plt.figure("Calibrated Momentum Slice", figsize=(12, 10)))
            plt.tight_layout()
            pdf.savefig()
        main_peak = np.argmax(smoothed_itof_hist[itof_peak_index])

        for i, (peak, left, right) in enumerate(zip(itof_peak_index, itof_le, itof_re)):
            print(f"{i} -- peak: {peak}, left: {left}, right: {right}")
            xyt_filtered = plt.figure(f"X-Y-T {i}", figsize=(12, 10))
            xf, yf, tf = xyt_plot(x, y, t, itof, t_range, itof_range,
                                  (filtered_itof_edges[round(left)], filtered_itof_edges[round(right)]), xyt_filtered)
            plt.tight_layout()
            pdf.savefig()
            if calibration:
                px, py, pz = fix_hole_calibrated(xf, yf, tf, center, angle, calibration)
                py, pz = coincidence_v4.rotate_data(py, pz, polarization_angle)
                xyt_calibrated = plt.figure(f"Calibrated Momentum {i}", figsize=(12, 10))
                calibrated_plot(px, py, pz, p_range, xyt_calibrated)
                plt.tight_layout()
                pdf.savefig()
                p_plane_slice = filter_coords([px, py, pz], [(-0.03, 0.03), p_range, p_range])
                calibrated_plot(*p_plane_slice, p_range, plt.figure(f"Calibrated Momentum {i} Slice", figsize=(12, 10)))
                plt.tight_layout()
                pdf.savefig()
                if i == main_peak:
                    save_data_mike(output.replace(".pdf", f'_mike.mat'), px, py, pz, file, center, angle,
                                   polarization_angle, left, right, etof_range,
                                   itof_range)

                    data_len = len(x)

                    reduction_factor = 5
                    beginning_uncalibrated = [x[:data_len // reduction_factor], y[:data_len // reduction_factor],
                                              t[:data_len // reduction_factor], itof[:data_len // reduction_factor]]
                    end_uncalibrated = [x[-data_len // reduction_factor:], y[-data_len // reduction_factor:],
                                        t[-data_len // reduction_factor:], itof[-data_len // reduction_factor:]]

                    beginning = fix_hole_calibrated(beginning_uncalibrated[0], beginning_uncalibrated[1],
                                                    beginning_uncalibrated[2], center, angle, calibration)
                    end = fix_hole_calibrated(end_uncalibrated[0], end_uncalibrated[1], end_uncalibrated[2], center,
                                              angle, calibration)

                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    plt.suptitle('Beginning vs End')
                    axes[0, 0].hist2d(beginning[1], beginning[2], bins=256, range=[p_range, p_range], cmap='jet')
                    axes[0, 0].title.set_text('Beginning (Projection)')
                    axes[0, 1].hist2d(end[1], end[2], bins=256, range=[p_range, p_range], cmap='jet')
                    axes[0, 1].title.set_text('End (Projection)')

                    beg_slice = filter_coords(beginning, [(-0.03, 0.03), p_range, p_range])
                    end_slice = filter_coords(end, [(-0.03, 0.03), p_range, p_range])
                    axes[1, 0].hist2d(beg_slice[1], beg_slice[2], bins=256, range=[p_range, p_range], cmap='jet')
                    axes[1, 0].title.set_text('Beginning (Slice)')
                    axes[1, 1].hist2d(end_slice[1], end_slice[2], bins=256, range=[p_range, p_range], cmap='jet')
                    axes[1, 1].title.set_text('End (Slice)')
                    plt.tight_layout()
                    pdf.savefig()


def save_data_mike(output, px, py, pz, file, center, angle, polarization_angle, left, right, etof_range, itof_range):
    parameters_orient = np.array([
        ('file name', file),
        ('cluster range (us)', np.array([[0, 1000]], dtype='uint16')),
        ('etof range (us)', np.array([etof_range], dtype='uint16')),
        ('tof range (us)', np.array([itof_range], dtype='uint16')),
        ('ion coincidence?', 'y'),
        ('coincidence range (us)', np.array([[left, right]])),
        ('3D hist size', 0),
        ('CoM_x (px)', center[0]),
        ('CoM_y (px)', center[1]),
        ('t_0 (us)', center[2]),
        ('xy rotate?', 'y'),
        ('xy rotation (deg)', np.degrees(angle)),
        ('polarization rotate?', 'y'),
        ('polarization rotation (deg)', np.degrees(polarization_angle)),
        ('# laser shots', 0),
        ('# 3D electrons', 0),
        ('notes', np.array(['laser prop=+y', 'major ax=z', 'minor ax=x'], dtype=object))
    ], dtype=object)
    data_dict = {'px': pz, 'py': px, 'pz': py, 'parameters_orient': parameters_orient}
    scipy.io.savemat(output, data_dict)


def xyt_plot(x, y, t, itof, etof_range, full_itof_range, itof_range, figure, mark_center=None):
    x, y, t, filter_itof = filter_coords([x, y, t, itof], [(0, 256), (0, 256), etof_range, itof_range])
    print(f"X: {len(x)}, Y: {len(y)}, T: {len(t)}, i-ToF: {len(itof)}")
    plt.suptitle("X-Y-T")
    figure.add_subplot(221)
    plt.hist2d(y, x, bins=256, range=[[0, 256], [0, 256]])
    if mark_center is not None:
        plt.scatter(mark_center[1], mark_center[0], c='r', marker='x')
    plt.xlabel("Y")
    plt.ylabel("X")
    figure.add_subplot(223)
    plt.hist2d(x, t, bins=256, range=[[0, 256], etof_range])
    if mark_center is not None:
        plt.scatter(mark_center[0], mark_center[2], c='r', marker='x')
    plt.xlabel("X")
    plt.ylabel("e-ToF")
    figure.add_subplot(222)
    plt.hist2d(t, y, bins=256, range=[etof_range, [0, 256]])
    if mark_center is not None:
        plt.scatter(mark_center[2], mark_center[1], c='r', marker='x')
    plt.xlabel("e-ToF")
    plt.ylabel("Y")
    figure.add_subplot(426)
    plt.hist(t, bins=1000, range=etof_range)
    if mark_center is not None:
        plt.axvline(mark_center[2], c='r', ls='--')
    plt.xlabel("e-ToF (ns)")
    figure.add_subplot(428)
    bin_size = 1
    plt.hist(np.asarray(itof) / 1000, bins=int((full_itof_range[1] - full_itof_range[0]) / bin_size),
             range=(full_itof_range[0] / 1000, full_itof_range[1] / 1000))
    if itof_range != full_itof_range:
        plt.hist(np.asarray(filter_itof) / 1000, bins=int((itof_range[1] - itof_range[0]) / bin_size), color='r',
                 range=(itof_range[0] / 1000, itof_range[1] / 1000))
    plt.xlabel("i-ToF (us)")
    return x, y, t


def calibrated_plot(px, py, pz, ranges, figure):
    plt.suptitle("Calibrated Momentum")
    figure.add_subplot(221)
    plt.hist2d(px, py, bins=256, range=[ranges, ranges])
    plt.xlabel("Propagation Axis")
    plt.ylabel("Major Axis")
    figure.add_subplot(223)
    plt.hist2d(px, pz, bins=256, range=[ranges, ranges])
    plt.xlabel("Propagation Axis")
    plt.ylabel("Minor Axis")
    figure.add_subplot(222)
    plt.hist2d(pz, py, bins=256, range=[ranges, ranges])
    plt.xlabel("Minor Axis")
    plt.ylabel("Major Axis")
    figure.add_subplot(224)
    plt.hist(np.sqrt(px ** 2 + py ** 2 + pz ** 2), bins=1000, range=[0, ranges[1]])
    plt.xlabel("Pr")
    plt.tight_layout()


def find_peaks(data, min_prominence=0.01):
    """
    Find the peak index of each positive region in the data.

    Parameters:
    - data: A list of numbers.

    Returns:
    - A list of indices representing the peak of each positive region.
    """
    peaks = []
    positive_region = False
    start_index = 0

    for i in range(len(data)):
        # Start of a positive region
        if data[i] > 0 and not positive_region:
            positive_region = True
            start_index = i

        # End of a positive region
        if data[i] <= 0 and positive_region:
            positive_region = False
            peak_index = start_index + np.argmax(data[start_index:i])
            if max(data[start_index:i]) > min_prominence * max(data):
                peaks.append(peak_index)

        # If we are at the end of the list and still in a positive region
        if i == len(data) - 1 and positive_region:
            peak_index = start_index + np.argmax(data[start_index:])
            if max(data[start_index:]) > min_prominence * max(data):
                peaks.append(peak_index)

    return peaks


def fix_hole(x, y, t, center, hole_center, hole_radius):
    return x,y,t
    hole_mask = (x - hole_center[0]) ** 2 + (y - hole_center[1]) ** 2 < hole_radius ** 2
    opposing_center = (2 * center[0] - hole_center[0], 2 * center[1] - hole_center[1])
    opposing_mask = (x - opposing_center[0]) ** 2 + (y - opposing_center[1]) ** 2 < hole_radius ** 2
    x_opp, y_opp, t_opp = 2 * center[0] - x[opposing_mask], 2 * center[1] - y[opposing_mask], t[opposing_mask]
    x_fixed, y_fixed, t_fixed = np.concatenate([x[~hole_mask], x_opp]), np.concatenate(
        [y[~hole_mask], y_opp]), np.concatenate([t[~hole_mask], t_opp])
    return x_fixed, y_fixed, t_fixed


def fix_hole_calibrated(x, y, t, center, angle, calibration):
    p_range = (-0.6, 0.6)
    r = 0.03
    px, py, pz = calibration(x, y, t, center, angle, symmetrize=False)
    idx=np.argwhere(pz>0).flatten()
    px, py, pz = px[idx], py[idx], pz[idx]
    return np.append(px, -px), np.append(py, -py), np.append(pz, -pz)
    xy_hist, xe, ye = np.histogram2d(px, py, bins=1024, range=(p_range, p_range), density=True)
    smoothed = scipy.ndimage.gaussian_filter(xy_hist, sigma=8)
    laplace = scipy.ndimage.laplace(smoothed)
    laplace = np.maximum(laplace, 0)
    xc, yc = np.argwhere(laplace == np.max(laplace)).flatten()
    mask = np.argwhere(((px - xe[xc]) ** 2 + (py - ye[yc]) ** 2) > r ** 2).flatten()
    opp_mask = np.argwhere(((px + xe[xc]) ** 2 + (py + ye[yc]) ** 2) < r ** 2).flatten()
    x_fix, y_fix, z_fix = np.append(px[mask], -px[opp_mask]), np.append(py[mask], -py[opp_mask]), np.append(pz[mask],
                                                                                                            -pz[
                                                                                                                opp_mask])
    x_fix, y_fix, z_fix = np.append(x_fix, -x_fix), np.append(y_fix, -y_fix), np.append(z_fix, -z_fix)
    print(xe[xc], ye[yc])
    return x_fix, y_fix, z_fix


if __name__ == '__main__':
    dir = r"D:\Data"
    for file in os.listdir(dir):
        if file.endswith("20deg.cv4"):
            try:
                analyze_to_pdf(os.path.join(dir, file), os.path.join(dir, file.replace(".cv4", ".pdf")), dead_pixels=[
                    (206, 197),
                    (197, 206),
                    (191, 197),
                    (196, 194),
                    (98.2, 163.3),
                    (0, 0)], angle=1.197608, calibration=calibration_20240208, polarization_angle=0.0,
                               hole_coords=(109.25, 136.5), hole_radius=5)
            except np.linalg.LinAlgError:
                print(f"Error in {file}")
                continue
# sort_key = lambda x: int(x.split('_')[1]) if x.endswith('.cv4') else 0
# if __name__ == '__main__':
#     matplotlib.rc('image', cmap='jet')
#     if os.name == 'nt':
#         dir= r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240208"
#         matplotlib.use('pdf')
#     else:
#         dir= r"/mnt/NAS/ctgroup/DATA/UCONN/VMI/VMI/20240208"
#         matplotlib.use('pdf')
#
#     plt.close("all")
#
#     for file in sorted(os.listdir(dir), key=sort_key):
#         if file.endswith(".cv4") and sort_key(file) ==3:
#             match file.split("_"):
#                 case _, _, "a.cv4":
#                     polarization_angle = 0.038 + 4 * np.pi / 180
#                 case _, _, "b.cv4":
#                     polarization_angle = 0.306 + 4 * np.pi / 180
#                 case _, _, "s.cv4":
#                     polarization_angle = np.pi / 2
#                 case _, _, "p.cv4":
#                     polarization_angle = 0
#
#             analyze_to_pdf(os.path.join(dir, file), os.path.join(dir, file.replace(".cv4", ".pdf")), dead_pixels=[
#                 (206, 197),
#                 (197,  206),
#                 (191, 197),
#                 (196, 194),
#                 (0, 0)
#             ], angle=1.197608, calibration=calibration_20240208, polarization_angle=polarization_angle,
#                            hole_coords=(109.25, 136.5), hole_radius=5)
#             # np.arctan2(91-168,183-66)
#             # break
#         plt.close(fig="all")
#     plt.show()
# %%
