import numpy as np

import coincidence_v4


def itof_calibration(itof, t0=670.54):
    return ((itof - t0)/1803.29)**2

def general_calibration(x, y, t, center, E_xy, E_t, angle=0., symmetrize=True, cutup=False):
    px = np.sqrt(2 * 0.0367493 * E_xy(x - center[0])) * np.sign(x - center[0])
    py = np.sqrt(2 * 0.0367493 * E_xy(y - center[1])) * np.sign(y - center[1])
    pz = np.sqrt(2 * 0.0367493 * E_t(t - center[2])) * np.sign(t - center[2])
    px, py = coincidence_v4.rotate_data(px, py, angle)
    if cutup:
        idx = np.argwhere(pz > 0).flatten()
        px, py, pz = px[idx], py[idx], pz[idx]
    if symmetrize:
        px, py, pz = np.column_stack([px, -px]).flatten(), np.column_stack([py, -py]).flatten(), np.column_stack([pz, -pz]).flatten()
    return px, py, pz


def calibration_20240208(x, y, t, center, angle=0, symmetrize=True, cutup=False):
    E_xy = lambda x: 0.000519 * x ** 2
    E_t = lambda t: np.where(t < 0, 0.1137238 * t ** 2 + 0.00291691 * t ** 3, 0.03621571 * t ** 2 - 0.00082477 * t ** 3)
    return general_calibration(x, y, t, center, E_xy, E_t, angle, symmetrize, cutup)


def calibration_20240806(x, y, t, center, angle=0, symmetrize=True, cutup=False):
    E_xy = lambda x: 0.000534601619665414 * x ** 2
    E_t = lambda t: np.where(t > 0,
                             0.037600543 * t ** 2 - 0.000492239 * t ** 3 - 7.57991E-12 * t ** 4 - 2.38291E-10 * t ** 5 - 3.37876E-08 * t ** 6,
                             0.102536983 * t ** 2 - 0.000867618 * t ** 3 + 2.95764E-13 * t ** 4 - 2.59601E-12 * t ** 5 + 1.11493E-11 * t ** 6)
    return general_calibration(x, y, t, center, E_xy, E_t, angle, symmetrize, cutup)


def calibration_20241120(x, y, t, center=(133.2, 131.7, 496), angle=-0.43, symmetrize=True, cutup=False):
    E_xy = lambda x: 0.000519 * x ** 2
    z = lambda t: np.where(
            t < 0,
            -np.polyval([3.1336553375552114, 181.91303568011313, 0, 0], t) ** 0.5,
            np.polyval([0.13548162025504634, -6.827514146748386, 125.8002806866771, 0, 0], t) ** 0.5
    )
    return general_calibration(x, y, t, center, E_xy, lambda t: E_xy(z(t)), angle, symmetrize, cutup)


def calibration_20250123(x,y,t, center=(127.75, 128.75, 495.42), angle=-0.5, symmetrize=True, cutup=False):
    E_xy=lambda x: 0.000519 * x**2
    z = lambda t: np.where(
        t < 0,
        -np.polyval([-2.180124512511416, 152.71609026513565, 0, 0],t)**0.5,
        np.polyval([-0.04885889138712628, -2.0134037673354435, 92.1302612227862, 0, 0],t)**0.5
    )
    return general_calibration(x,y,t, center ,E_xy,lambda t: E_xy(z(t)), angle, symmetrize, cutup)

def calibration_20250228(x,y,t, center=(121.3, 128.7, 495.9), angle=-0.53, symmetrize=True, cutup=False):
    E_xy=lambda x: 0.000519 * x**2
    z = lambda t: np.where(
            t < 0,
            -np.polyval([-15.939239037763162, 44.81692906680434, 0, 0],t)**0.5,
            np.polyval([0.09487404130881483, -3.7896129598378003, 92.55227231395205, 0, 0],t)**0.5
    )
    return general_calibration(x,y,t, center ,E_xy,lambda t: E_xy(z(t)), angle, symmetrize, cutup)

# def calibration_20250303(x,y,t, center=(121.375, 128.25, 495.42), angle=-0.54, symmetrize=True, cutup=False):
#     E_xy=lambda x: 0.000519 * x**2
#     z = lambda t: np.where(
#             t < 0,
#             -np.polyval([-3.9696366262625324, 150.7251571326864, 0, 0],t)**0.5,
#             np.polyval([-0.1370448112323239, 0.5805970723710808, 71.025514133672, 0, 0],t)**0.5
#     )
#     return general_calibration(x,y,t, center ,E_xy,lambda t: E_xy(z(t)), angle, symmetrize, cutup)

# def calibration_20250303(x,y,t, center=(121.625, 127.625, 495.92757346365676), angle=2.630162514320607, symmetrize=True, cutup=False):
#     E_xy=lambda x: 0.000519 * x**2
#     z = lambda t: np.where(
#             t < 0,
#             -np.polyval([0.9257842027376331, 129.3299219963953, 0, 0],-t)**0.5,
#             np.polyval([0.06087596354321943, -5.34959474544479, 116.76233863558174, 0, 0],t)**0.5
#     )
#     return general_calibration(x,y,t, center ,E_xy,lambda t: E_xy(z(t)), angle, symmetrize, cutup)
def calibration_20250303(x,y,t, center=(121.875, 127.125, 495.9345297060858), angle=2.630162514320607, symmetrize=True, cutup=False):
    E_xy=lambda x: 0.000519 * x**2
    z = lambda t: np.where(
            t < 0,
            -np.polyval([1.0067789324603167, 129.95544764952695, 0, 0],-t)**0.5,
            np.polyval([0.12542165461684168, -6.387291956347507, 121.53401756837005, 0, 0],t)**0.5
    )
    return general_calibration(x,y,t, center ,E_xy,lambda t: E_xy(z(t)), angle, symmetrize, cutup)
#%%
