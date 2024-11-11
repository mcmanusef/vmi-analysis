import os

import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl
def trans_jet(opaque_point=0.15):
    # Create a colormap that looks like jet
    cmap = plt.cm.jet

    # Create a new colormap that is transparent at low values
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:int(cmap.N * opaque_point), -1] = np.linspace(0, 1, int(cmap.N * opaque_point))
    return mpl.colors.ListedColormap(cmap_colors)
def main(file_path, p_max=1, unsymmetrize=False, slice_width=0.05):
    data=scipy.io.loadmat(file_path,struct_as_record=False,squeeze_me=True)

    major=data['pz']
    minor=data['px']
    propagation=data['py']
    if unsymmetrize:
        major=major[:major.size//2]
        minor=minor[:minor.size//2]
        propagation=propagation[:propagation.size//2]

    major, minor, propagation=major[np.abs(propagation) < slice_width], minor[np.abs(propagation) < slice_width], propagation[np.abs(propagation) < slice_width]

    p_r=np.sqrt(major**2+minor**2+propagation**2)
    angle=np.arctan2(minor,major)

    radial_spectrum, r_edges=np.histogram(p_r,bins=256,range=(0,p_max))
    r_centers=(r_edges[1:]+r_edges[:-1])/2

    angular_spectrum, a_edges=np.histogram(angle,bins=256,range=(-np.pi,np.pi))
    a_centers=(a_edges[1:]+a_edges[:-1])/2

    radial_spectrum=scipy.signal.savgol_filter(radial_spectrum, 21, 3)
    angular_spectrum=scipy.signal.savgol_filter(angular_spectrum, 21, 3)

    r_peaks=scipy.signal.find_peaks(radial_spectrum)[0]
    r_prominences=scipy.signal.peak_prominences(radial_spectrum,r_peaks)[0]
    r_widths=scipy.signal.peak_widths(radial_spectrum,r_peaks)[0]
    r_widths=np.round(r_widths).astype(int)

    r_peaks, r_widths, r_peak_heights=r_centers[r_peaks], r_centers[r_widths], radial_spectrum[r_peaks]
    hist2d, _,_=np.histogram2d(angle,p_r,bins=256,range=((-np.pi,np.pi),(0,p_max)))
    hist2d=scipy.ndimage.gaussian_filter(hist2d, sigma=1)

    fig, ax=plt.subplots(1,1)
    plt.title(file_path)
    plt.grid(linestyle='--',zorder=-1)
    plt.imshow(hist2d.T, extent=(-np.pi,np.pi,0,p_max), cmap=trans_jet(), aspect='auto', origin='lower', zorder=0)

    line=[weighted_circular_mean(a_centers,hist2d[:,i],period=np.pi) for i in range(len(hist2d))]
    line=np.unwrap(line,period=np.pi)
    for i in [-2,-1,0,1,2]:
        plot_multicolored_line(line+i*np.pi, r_centers,np.sum(hist2d,axis=0)/np.max(np.sum(hist2d,axis=0)),cmap='inferno')
    plt.xlim(-np.pi,np.pi)

    plt.xlabel('Angle (rad)')
    plt.ylabel('Momentum (a.u.)')
    plt.twiny(ax)
    plt.plot(radial_spectrum, r_centers, color='r',ls='--')
    for i in range(len(r_peaks)):
        index=np.logical_and(r_centers>r_peaks[i]-r_widths[i]/2,r_centers<r_peaks[i]+r_widths[i]/2)
        plt.fill_betweenx(r_centers[index],0,radial_spectrum[index], color='r',alpha=0.3)
    plt.xlim(0,radial_spectrum.max()*10)
    plt.xticks([])
    plt.twinx(ax)
    plt.plot(a_edges[:-1],angular_spectrum, color='g',ls='--')
    plt.fill_between(a_centers,0,angular_spectrum, color='g',alpha=0.5)
    plt.ylim(0,angular_spectrum.max()*10)
    plt.yticks([])
    plt.show()




def weighted_circular_mean(angles, weights, period=2*np.pi):
    angles=angles%period*2*np.pi/period
    x=np.sum(weights*np.cos(angles))
    y=np.sum(weights*np.sin(angles))
    return np.arctan2(y,x)%(2*np.pi)*period/(2*np.pi)

def plot_multicolored_line(x,y,c,ax=None, cmap='viridis', outline='k', **kwargs):
    if ax is None:
        ax=plt.gca()
    if outline is not None:
        ax.plot(x,y,lw=2,color=outline,**kwargs)
    for i in range(len(x)-1):
        ax.plot(x[i:i+2],y[i:i+2],color=plt.get_cmap(cmap)(c[i]), lw=1,**kwargs)
    return ax






if __name__ == "__main__":
    mpl.use('Qt5Agg')
    wdir=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240208"
    wdir=r"J:\ctgroup\Edward\DATA\VMI\20240424"
    wdir=r"D:\Data"
    for file in os.listdir(wdir):
    # for file in sorted(os.listdir(wdir), key=lambda x: int(x.split('_')[2])if x.endswith('_mike.mat') else -1):
        if not file.endswith('_mike.mat'):
            continue
        print(file)
        file=os.path.join(wdir,file)
        main(file,p_max=0.6, unsymmetrize=True, slice_width=0.6)
    plt.show()
#%%