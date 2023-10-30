import numba
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


@njit
def process_tdc(packet):
    split_points = (5, 9, 44, 56, 60)
    f_time,  c_time, _, tdc_type = split_int(packet, pw_jit(split_points))
    c_time = c_time & 0x1ffffffff  # Remove 2 most significant bit to loop at same point as pixels

    return 0, (tdc_type, c_time, f_time-1, 0)
    # c_time in units of 3.125 f_time in units of 260 ps,
    # tdc_type: 10->TDC1R, 15->TDC1F, 14->TDC2R, 11->TDC2F


@njit
def process_pixel(packet):
    split_points = (0, 16, 20, 30, 44, 61)
    c_time, f_time, tot, m_time, pix_add = split_int(packet, pw_jit(split_points))
    toa = numba.uint64(c_time * 2 ** 18 + m_time * 2 ** 4)
    toa = numba.uint64(toa - f_time)
    x, y = addr_to_coords(pix_add)
    return 1, (x, y, toa, tot)  # x,y in pixels, toa in units of 25ns/2**4, tot in units of 25 ns


@njit
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
