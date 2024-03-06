import numpy as np

import coincidence_v4


def calibration_20240208(x, y, t, center, angle=0, symmetrize=True):
    E_xy=lambda x: 0.000519 * x**2
    E_t=lambda t: np.where(t < 0, 0.1137238 * t ** 2 + 0.00291691 * t ** 3, 0.03621571 * t ** 2 - 0.00082477 * t ** 3)

    px=np.sqrt(2*0.0367493* E_xy(x-center[0]))*np.sign(x-center[0])
    py=np.sqrt(2*0.0367493* E_xy(y-center[1]))*np.sign(y-center[1])
    pz=np.sqrt(2*0.0367493* E_t(t-center[2]))*np.sign(t-center[2])
    px,py=coincidence_v4.rotate_data(px, py, angle)
    if symmetrize:
        px,py,pz=np.column_stack([px,-px]).flatten(),np.column_stack([py,-py]).flatten(),np.column_stack([pz,-pz]).flatten()
    return px,py,pz
