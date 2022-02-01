import ast

import numpy as np
from numba import njit
import numba.types as nbt

@njit(cache=True)
def calculate_tof(tdc, toa, tdc_index):
    
    tof = np.zeros_like(toa)
    
    for i, t in enumerate(toa):
        
        # Search correct TDC pulse
        while tdc_index + 1 < len(tdc) and tdc[tdc_index + 1] < t:
            tdc_index += 1
         
        assert tdc_index < len(tdc)
        
        tof[i] = t - tdc[tdc_index]
    
    return tof, tdc_index

def empty_image(file_):
    board_layout = ast.literal_eval(file_.attrs['BoardLayout'])

    if 'Gaps' in file_.attrs:
        gaps = int(file_.attrs['Gaps'])
    else:
        gaps = 0
     
    max_X = max([chip['X'] for chip in board_layout])
    max_Y = max([chip['Y'] for chip in board_layout])
     
    return np.zeros(((max_X+1) * 256 + max_X*gaps, (max_Y+1) * 256 + max_Y*gaps), np.uint64)


