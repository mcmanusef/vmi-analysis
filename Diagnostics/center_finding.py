import numpy as np
import scipy
from calibrations import calibration_20240208
import matplotlib.pyplot as plt
import matplotlib
from minor_utils import autoconvolve
matplotlib.use('Qt5Agg')
def calibrate_data(x,y,etof,center):
    px,py,pz=calibration_20240208(x,y,etof,center=center,angle=1.197608,symmetrize=True)
    return px,py,pz

if __name__ == '__main__':
    filename=r"D:\Data\xe_s_8W.mat"
    # filename=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240208\kr_06_b.mat"
    data=scipy.io.loadmat(filename,squeeze_me=True,struct_as_record=False)
    x,y,t,etof=data['x'],data['y'],data['t'],data['etof']
    n=1000000
    # x,y,etof=x[:n],y[:n],etof[:n]
    etof=etof+0.26*np.random.random_sample(len(etof))
    etof_range=(749040,749080)
    calculated_center=autoconvolve.find_center(x,y,etof,etof_range=etof_range, iterations=10)
    xw=10
    idx=np.argwhere(abs(y-calculated_center[1])<xw).flatten()
    x,y,etof=x[idx],y[idx],etof[idx]


    f,ax=plt.subplots(1,2,squeeze=False)
    hist,xe,ye,plot=ax[0,0].hist2d(x, etof, bins=256, range=((0, 256), etof_range), cmap='jet', density=True)
    marker=ax[0,0].scatter(calculated_center[0],calculated_center[2],c='r',marker='x')

    #add slider
    axcolor = 'lightgoldenrodyellow'
    axcenter = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    z_center_slider = matplotlib.widgets.Slider(axcenter, 'Z Center', etof_range[0],etof_range[1], valinit=calculated_center[2])
    ax_center2 = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    x_center_slider = matplotlib.widgets.Slider(ax_center2, 'X Center', 0,256, valinit=calculated_center[0])
    button_ax=plt.axes([0.8, 0.025, 0.1, 0.04])
    button=matplotlib.widgets.Button(button_ax,'Save')
    def update(val):
        z_center=z_center_slider.val
        x_center=x_center_slider.val
        marker.set_offsets((x_center,z_center))
        px,py,pz=calibrate_data(x,y,etof,(x_center,calculated_center[1],z_center))
        idx=np.argwhere(pz>0).flatten()
        px,py,pz=px[idx],py[idx],pz[idx]
        px,py,pz=np.hstack((px,-px)),np.hstack((py,-py)),np.hstack((pz,-pz))
        hist,xe,ye=np.histogram2d(py,pz,bins=512,range=((-1,1),(-1,1)))
        ax[0,1].clear()
        ax[0,1].imshow(hist.T/np.max(hist),extent=(xe[0],xe[-1],ye[0],ye[-1]),cmap='jet',origin='lower',interpolation='nearest')

    def save(val):
        center=z_center_slider.val
        full_center=(x_center_slider.val,calculated_center[1],center)
        data['center']=full_center
        scipy.io.savemat(filename,data)


    z_center_slider.on_changed(update)
    x_center_slider.on_changed(update)
    button.on_clicked(save)
    update(0)




#%%
