import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numba
import numpy as np
from numba import njit


@njit
def pw_jit(lst):
    out = []
    for i in range(len(lst) - 1):
        out.append((lst[i], lst[i + 1]))
    return out


@njit
def split_int(num: int, ranges: list[tuple[int, int]]):
    out = []
    for r in ranges:
        out.append((num & (1 << r[1]) - 1) >> r[0])
    return out


@njit
def addr_to_coords(pix_addr):
    dcol = (pix_addr & 0xFE00) >> 8
    spix = (pix_addr & 0x01F8) >> 1
    pix = (pix_addr & 0x0007)
    x = (dcol + pix // 4)
    y = (spix + (pix & 0x3))
    return x, y


@njit
def process_packet(packet: int):
    header = packet >> 60
    reduced = packet & 0x0fff_ffff_ffff_ffff
    if header == 7:
        return process_pixel(reduced)

    elif header == 2:
        return process_tdc(reduced)

    else:
        return -1, (0, 0, 0, 0)


# @njit
def process_tdc_old(packet):
    split_points = (5, 9, 44, 56, 60)
    f_time, c_time, _, tdc_type = split_int(packet, pw_jit(split_points))
    c_time = c_time & 0x1ffffffff  # Remove 2 most significant bit to loop at same point as pixels
    return 0, (tdc_type, c_time, f_time, 0)
    # c_time in units of 3.125 f_time in units of 260 ps,
    # tdc_type: 10->TDC1R, 15->TDC1F, 14->TDC2R, 11->TDC2F


@njit
def process_tdc(packet: int):
    split_points = (5, 9, 44, 56, 60)
    f_time, c_time, _, tdc_type = split_int(packet, pw_jit(split_points))
    c_time = c_time & 0x1ffffffff  # Remove 2 most significant bit to loop at same point as pixels

    return 0, (tdc_type, c_time, f_time - 1, 0)
    # c_time in units of 3.125 f_time in units of 260 ps,
    # tdc_type: 10->TDC1R, 15->TDC1F, 14->TDC2R, 11->TDC2F


@njit
def process_pixel(packet: int):
    split_points = (0, 16, 20, 30, 44, 61)
    c_time, f_time, tot, m_time, pix_add = split_int(packet, pw_jit(split_points))
    toa = numba.uint64(c_time * 2 ** 18 + m_time * 2 ** 4)
    toa = numba.uint64(toa - f_time)
    x, y = addr_to_coords(pix_add)
    return 1, (toa, x, y, tot)  # x,y in pixels, toa in units of 25ns/2**4, tot in units of 25 ns


def cluster_pixels(pixels, dbscan):
    if not pixels:
        return []
    toa, x, y, tot = map(np.asarray, zip(*pixels))
    cluster_index = dbscan.fit(np.column_stack((x, y))).labels_
    return cluster_index, toa, x, y, tot


# @njit
def average_over_clusters(cluster_index, toa, x, y, tot):
    clusters = []
    if len(cluster_index) > 0 and max(cluster_index) >= 0:
        for i in range(max(cluster_index) + 1):
            clusters.append((
                np.average(toa[cluster_index == i], weights=tot[cluster_index == i]),
                np.average(x[cluster_index == i], weights=tot[cluster_index == i]),
                np.average(y[cluster_index == i], weights=tot[cluster_index == i]),
            ))
    return clusters


# @njit
def sort_tdcs(cutoff: float | int, tdcs: list[tuple[int, int, int, int]]):
    start_time = 0
    pulses, etof, itof = [], [], []
    for tdc_type, c_time, ftime, _ in tdcs:
        tdc_time = 3.125 * c_time + .260 * ftime
        if tdc_type == 15:
            start_time = tdc_time
        elif tdc_type == 14:
            etof.append(tdc_time)
        elif tdc_type == 10:
            pulse_len = (tdc_time - start_time)
            if pulse_len > cutoff:
                pulses.append(start_time) if start_time > 0 else None
            else:
                itof.append(start_time) if start_time > 0 else None
    return etof, itof, pulses


# @njit
def process_chunk(chunk):
    tdcs = []
    pixels = []
    for packet in chunk:
        data_type, packet_data = process_packet(packet)
        if data_type == -1:
            continue
        if data_type == 1:
            pixels.append(packet_data)
        elif data_type == 0:
            tdcs.append(packet_data)
    return pixels, tdcs


# @njit
def sort_clusters(clusters):
    period = 25 * 2 ** 30
    t0 = clusters[0][0]
    c_adjust = [((t - t0 + period / 2) % period - period / 2, x, y) for t, x, y in clusters]
    return [((t + t0) % period, x, y) for t, x, y in sorted(c_adjust)]


# @njit
def fix_toa(toa):
    if max(toa) - min(toa) < 2 ** 32:
        return toa
    period = 2 ** 34
    t0 = toa[0]
    return (toa - t0 + period / 2) % period + t0 - period / 2


# @njit
def apply_timewalk(pixels, timewalk_correction):
    for i, (toa, x, y, tot) in enumerate(pixels):
        if tot >= len(timewalk_correction):
            pixels[i] = (toa - timewalk_correction[-1], x, y, tot)
        pixels[i] = (toa - timewalk_correction[tot], x, y, tot)
    return pixels


# @njit
def toa_correction(pixels, correction):
    for i, (toa, x, y, tot) in enumerate(pixels):
        pixels[i] = (toa - correction, x, y, tot)
    return pixels
