
import matplotlib.pyplot as plt  # -*- coding: utf-8 -*-
import itertools
from scipy.io import loadmat
import scipy.interpolate as inter
import scipy.signal as signal
import scipy.stats as stats
import numpy as np
import os
import matplotlib
plt.close("all")
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \boldmath \bfseries"
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.weight'] = 'bold'


def threshold(array, n, default=0, fraction=False):
    return np.where(array < n, default, array) if not fraction else np.where(array < n*np.max(array), default, array)


source = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clustered_new"

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
fig.set_size_inches(12, 4)

files = ['xe011_e.mat', 'xe015_e.mat', 'xe013_e.mat', 'theory.mat']
titles = [r'$\varepsilon = 0.3 \:L$', r'$\varepsilon = 0.3 \:R$', r'$\varepsilon = 0.6 \:L$', r'$\varepsilon = 0.6 \:R$']

datas = [loadmat(os.path.join(source, d)) for d in files]
plt.sca(axes[0])
plt.ylabel("$p_y$")

for ax, title, data, let in zip(axes, titles, datas, ['$a$', '$b$', '$c$', '$d$']):
    plt.sca(ax)
    ax.annotate(let, xy=(0.9, 0.88), xycoords="axes fraction", fontweight='bold')
    xv = data['xv'][0]+0.00001
    cut_unnorm = np.sum(data['hist'][:, (np.abs(data['zv']) < 0.05)[0], :], 1)
    cut = cut_unnorm/np.max(cut_unnorm)*256
    # plt.title(title, fontweight='bold')
    plt.xlabel("$p_x$")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.imshow(threshold(cut, 0.003, fraction=True, default=-1)**0.5,
               extent=[min(xv), max(xv), min(xv), max(xv)], cmap='jet')

plt.tight_layout()
plt.savefig("cuts_figure.png")
