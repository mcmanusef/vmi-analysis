
import scipy.stats as stats
from scipy.stats import qmc

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def get_angular_error(angles: list[list[float]]) -> tuple[float, float]:
    centers=[(np.mean(np.cos(theta)),np.mean(np.sin(theta))) for theta in thetas]
    xs,ys=zip(*centers)
    cx=np.mean(xs)
    cy=np.mean(ys)
    error_matrix=np.cov(xs,ys)/len(xs)
    sample=qmc.MultivariateNormalQMC(mean=[cx, cy], cov=error_matrix).random(2048)
    sample_angles=np.arctan2(sample[:,1],sample[:,0])
    angle=stats.circmean(sample_angles)
    error=stats.circstd(sample_angles)
    return angle, error


n=10000
npart=1000
ni=n//npart

ratios=[]

def dist(i): return random.gauss(1,2)%(2*np.pi) 

for k in range(100):
    thetas=[np.array([dist(i) for i in range(ni)]) for j in range(npart)]
    
    angle, error= get_angular_error(thetas)
    ratios.append(min(abs(1-angle),abs(angle-1))/error)
    
plt.figure(1)
plt.hist(ratios, bins=100, density=True)

# f,ax=plt.subplots(3,1)
# # ax[0].hist(thetas, bins=1000, range=[-np.pi,np.pi],histtype='step')
# # ax[0].twinx()
# plt.hist(sample_angles, bins=1000, range=[-np.pi,np.pi],histtype='step')

# plt.sca(ax[1])
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# ax[1].scatter(np.cos(thetas), np.sin(thetas), s=1)
# ax[1].scatter(xs,ys)
# ax[1].scatter(sample[:,0], sample[:,1], s=1)

# plt.sca(ax[2])
# plt.xlim(-.02,.02)
# plt.ylim(-.02,.02)
# ax[2].scatter(sample[:,0]-cx, sample[:,1]-cy)

