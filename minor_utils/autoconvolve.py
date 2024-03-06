import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from calibrations import calibration_20240208


def find_center(x, y, etof, etof_range=(749000,749100), calibration=calibration_20240208, iterations=3, n=256):
    xy_hist,xe,ye=np.histogram2d(x, y, bins=n, range=((0, 256), (0, 256)))
    xy_autoconvolve=scipy.signal.fftconvolve(xy_hist,xy_hist,mode='full')
    auto_xe=np.linspace(xe[0]*2,xe[-1]*2,len(xy_autoconvolve))
    auto_ye=np.linspace(ye[0]*2,ye[-1]*2,len(xy_autoconvolve[0]))
    center=np.argwhere(xy_autoconvolve==xy_autoconvolve.max()).flatten()
    cx,cy=auto_xe[center[0]]/2,auto_ye[center[1]]/2

    etof_hist,etof_edges=np.histogram(etof,bins=1000,range=etof_range)
    auto_etof=scipy.signal.convolve(etof_hist,etof_hist,mode='full')
    auto_etof_xe=np.linspace(etof_edges[0]*2,etof_edges[-1]*2,len(auto_etof))
    center=np.argwhere(auto_etof==auto_etof.max()).flatten()
    ct=auto_etof_xe[center[0]]/2

    for i in range(iterations):
        px,py,pz=calibration(x,y,etof,center=(cx,cy,ct), angle=1.197608,symmetrize=False)
        hist_xz,xe,ze=np.histogram2d(px, pz, bins=n, range=((-1, 1), (-1, 1)))
        xz_autoconvolve=scipy.signal.fftconvolve(hist_xz,hist_xz,mode='full')
        auto_ze=np.linspace(ze[0]*2,ze[-1]*2,len(xz_autoconvolve[0]))
        center=np.argwhere(xz_autoconvolve==xz_autoconvolve.max()).flatten()
        cz=auto_ze[center[1]]/2

        pz_sample=calibration(x,y,t_sample:=np.linspace(-10,10,num=100),center=(0,0,0),symmetrize=False)[2]
        ct += t_sample[np.argmin(np.abs(pz_sample - cz))]
    return cx,cy,ct

if __name__=='__main__':
    n = 512
    matplotlib.rc('image', cmap='jet')
    matplotlib.use('Qt5Agg')
    plt.close("all")
    data=scipy.io.loadmat(r'J:\ctgroup\DATA\UCONN\VMI\VMI\20240208\xe_01_p.mat',squeeze_me=True,struct_as_record=False)
    x,y,t,etof=data['x'],data['y'],data['t'],data['etof']
    idx=np.argwhere(np.logical_and(t>748800,t<748900)).flatten()
    x,y,etof=x[idx],y[idx],etof[idx]+0.26*np.random.random_sample(len(etof[idx]))

    plt.figure()
    hist,xe,ye,_=plt.hist2d(x, y, bins=n, range=((0, 256), (0, 256)), cmap='jet', density=True)

    autoconvolve=scipy.signal.fftconvolve(hist,hist,mode='full')
    auto_xe=np.linspace(xe[0]*2,xe[-1]*2,len(autoconvolve))
    auto_ye=np.linspace(ye[0]*2,ye[-1]*2,len(autoconvolve[0]))
    center=np.argwhere(autoconvolve==autoconvolve.max()).flatten()
    cx,cy=auto_xe[center[0]]/2,auto_ye[center[1]]/2
    print(auto_xe[center[0]]/2,auto_ye[center[1]]/2)
    plt.scatter(auto_xe[center[0]]/2,auto_ye[center[1]]/2,c='r',marker='x')
    plt.figure()
    plt.imshow(autoconvolve.T,extent=(xe[0],xe[-1],ye[0],ye[-1]),cmap='jet',origin='lower')
    plt.scatter(auto_xe[center[0]]/2,auto_ye[center[1]]/2,c='r',marker='x')

    plt.figure()
    etof_hist,etof_xe,_=plt.hist(etof,bins=1000,range=(749000,749100),density=True)
    auto_etof=scipy.signal.convolve(etof_hist,etof_hist,mode='full')
    auto_etof_xe=np.linspace(etof_xe[0]*2,etof_xe[-1]*2,len(auto_etof))
    center=np.argwhere(auto_etof==auto_etof.max()).flatten()
    ct=auto_etof_xe[center[0]]/2
    print(auto_etof_xe[center[0]]/2)
    plt.plot(auto_etof_xe/2,auto_etof)
    plt.axvline(auto_etof_xe[center[0]]/2,c='r',ls='--')

    for i in range(3):
        px,py,pz=calibration_20240208(x, y, etof, center=(cx, cy, ct), angle=1.197608, symmetrize=False)
        plt.figure()
        hist_xz,xe,ze,_=plt.hist2d(py, pz, bins=n, range=((-1, 1), (-1, 1)), cmap='jet', density=True)
        xz_autoconvolve=scipy.signal.fftconvolve(hist_xz,hist_xz,mode='full')
        auto_xe=np.linspace(xe[0]*2,xe[-1]*2,len(xz_autoconvolve))
        auto_ze=np.linspace(ze[0]*2,ze[-1]*2,len(xz_autoconvolve[0]))
        center=np.argwhere(xz_autoconvolve==xz_autoconvolve.max()).flatten()
        cz=auto_ze[center[1]]/2
        print(auto_xe[center[0]]/2,auto_ze[center[1]]/2)
        plt.scatter(auto_xe[center[0]]/2,auto_ze[center[1]]/2,c='r',marker='x')
        plt.figure()
        plt.imshow(xz_autoconvolve.T,extent=(xe[0],xe[-1],ze[0],ze[-1]),cmap='jet',origin='lower')
        plt.scatter(auto_xe[center[0]]/2,auto_ze[center[1]]/2,c='r',marker='x')


        pz_sample=calibration_20240208(x, y, t_sample:=np.linspace(-10, 10, num=100), center=(0, 0, 0), symmetrize=False)[2]
        plt.figure()
        plt.plot(t_sample,pz_sample)
        plt.axhline(cz,c='k',ls='--')
        t_0=t_sample[np.argmin(np.abs(pz_sample-cz))]
        plt.axvline(t_0,c='r',ls='--')
        print(t_0+ct)
        ct+=t_0

    px,py,pz=calibration_20240208(x, y, etof, center=(cx, cy, ct), angle=1.197608, symmetrize=False)
    px,py,pz=np.column_stack([px,-px]).flatten(),np.column_stack([py,-py]).flatten(),np.column_stack([pz,-pz]).flatten()
    f,ax=plt.subplots(2,2,figsize=(5,5))
    ax[0,0].hist2d(px, py, bins=n, range=((-0.6, 0.6), (-0.6,0.6)), cmap='jet', density=True)
    ax[0,1].hist2d(px, pz, bins=n, range=((-0.6, 0.6), (-0.6,0.6)), cmap='jet', density=True)
    ax[1,0].hist2d(py, pz, bins=n, range=((-0.6, 0.6), (-0.6,0.6)), cmap='jet', density=True)
    ax[1,1].hist(np.sqrt(px**2+py**2), bins=n, range=(0, 1), density=True, histtype='step')
    # plt.hist(-pz,bins=1000,range=(-1,1),density=True,histtype='step')

    plt.show()