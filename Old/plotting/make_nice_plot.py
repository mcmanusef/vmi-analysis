import itertools
import numpy as np
import scipy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from plotting import plotting_utils
import calibrations
#%%
file="o2_b_01.mat"
data=scipy.io.loadmat(r"o2_b_01.mat",squeeze_me=True,struct_as_record=False)

x,y,t,etof=data['x'],data['y'],data['t'],data['etof']
idx=np.argwhere(np.logical_or(t>748800,t<748900)).flatten()
x,y,etof=x[idx],y[idx],etof[idx]+0.26*np.random.random_sample(len(etof[idx]))
x,y,etof=plotting_utils.dp_filter([(191, 197), (196, 194),(98.2, 163.3), (0, 0)],x,y,etof)

x,y,etof = plotting_utils.filter_coords((x,y,etof),((0,256),(0,256),(0,1e6)))


center=(120.5,134.5,749055.4)
n=256
f,ax=plt.subplots(2,2)
ax[0,0].hist2d(x,y,bins=n,range=((0,256),(0,256)))
ax[0,0].scatter(center[0],center[1],c='r',marker='x')
ax[0,1].hist2d(etof,y,bins=n,range=((749030,749080),(0,256)))
ax[0,1].scatter(center[2],center[1],c='r',marker='x')
ax[1,0].hist2d(x,etof,bins=n,range=((0,256),(749030,749080)))
ax[1,0].scatter(center[0],center[2],c='r',marker='x')
ax[1,1].hist(etof,bins=1000,range=(749030,749080))
ax[1,1].axvline(center[2],c='r')

x, y, etof = plotting_utils.filter_coords((x,y,etof),((0,256),(0,256),(center[2],749075)))

#%%
for pxr in itertools.pairwise(np.linspace(-0.25, 0.25, 6)):
    pxr=tuple(pxr)
    px,py,pz=calibrations.calibration_20240208(x,y,etof,center=center,angle=1.197608,symmetrize=True)


    px,py,pz= plotting_utils.filter_coords((px,py,pz),(pxr,(-1,1),(-1,1)))
    hist,xe,ye=np.histogram2d(py,pz,bins=2048,range=((-0.5,0.5),(-0.5,0.5)))

    hist=scipy.ndimage.gaussian_filter(hist,0.5)
    # scipy.io.savemat("o2_b_01_pe.mat",{"hist":hist,"xe":xe,"ye":ye})
    plt.rcParams["font.weight"] = "medium"
    plt.rcParams["font.size"] = 14
    f,ax=plt.subplots()
    gamma=1
    maxed_hist=np.minimum(hist.T/np.max(hist),0.2)**gamma
    plt.grid(zorder=-1, alpha=0.5)
    ax.imshow(maxed_hist,extent=(xe[0],xe[-1],ye[0],ye[-1]),cmap=plotting_utils.trans_jet(opaque_point=0.15),origin='lower', interpolation='bicubic', zorder=0)
    plt.title("O$_2$ P.E. Distribution, $p_z$={:.2f} to {:.2f}".format(*pxr))
    plt.xlabel("$p_x$ (a.u.)")
    plt.ylabel("$p_y$ (a.u.)")
    plt.tight_layout()

    plt.savefig(f"o2_{np.mean(pxr):.2f}.png",dpi=300)

#%%
