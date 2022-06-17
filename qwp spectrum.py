# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:56:02 2022

@author: mcman
"""

import matplotlib.pyplot as plt
import numpy as np

wvln = np.zeros(2048)
spectrum = np.zeros((10, 2048))
for i in range(10):
    with open("J:\\ctgroup\\Edward\\qwp\\gg{}.txt".format(i*5)) as f:
        data = f.readlines()
    data = data[17:-1]
    print(len(data))
    for j, st in enumerate(data):
        wvln[j] = float(st.split()[0])
        spectrum[i, j] = float(st.split()[1])
    norm = max(spectrum[i])
    spectrum[i] = spectrum[i]/norm
plt.plot(wvln, spectrum.transpose())
