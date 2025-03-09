import scipy
from vmi_analysis import coincidence_v4
from vmi_analysis import calibrations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("pdf")
import scipy.io
import os

def analyze_file(fname, coincidence=False, gate='', calibration = calibrations.calibration_20250123):
    data = coincidence_v4.load_file(fname, coincidence=coincidence)
    print("Data Loaded")
    # data = coincidence_v4.load_file(r"D:\Data\c2h4_p_5W.cv4")
    x,y,t,etof,itof=data
    etof+=0.26*np.random.random_sample(len(etof))
    n=1024//2
    gamma=0.25
    gate_dict = {
        '': None,
        'CH4+': (7000, 9000),
        'C2H4+': (9000, 11000),
        'O2+': (10200, 11500),
        'N2O+': (12000, 13500),
    }

    itof_gate=gate_dict[gate]
    gate_title=gate

    try:
        import cmasher as cmr
        cmap=cmr.rainforest
    except ImportError:
        cmap='viridis'
    name=fname[:-4] if not itof_gate else fname[:-4]+'_'+gate_title if gate_title else fname[:-4]+'_'+str(itof_gate[0])+'_'+str(itof_gate[1])
    title=fname

    if gate_title:
        title+=f' (Gated on {gate_title})'
    elif itof_gate:
        title+=f' (Gated from {itof_gate[0]}-{itof_gate[1]} ns)'

    fig = plt.figure(figsize=(9, 11))
    plt.suptitle(title)
    plt.subplot(311)
    plt.gca().set_axisbelow(True)
    plt.title('Time of Flight Spectra (Log Scale, Density)')
    plt.semilogy()
    plt.xlabel('Time (ns)')
    plt.ylabel('Count')
    plt.grid()

    plt.hist(t, bins=3000, range=(0, 20000), color='b', alpha=0.7, label='ToA')
    plt.hist(etof, bins=3000, range=(0, 20000), color='r', alpha=0.7, label='e-ToF')
    plt.hist(itof, bins=3000, range=(0, 20000), color='g', alpha=0.7, label='i-ToF')
    if itof_gate:
        plt.axvline(itof_gate[0], color='k', linestyle='--')
        plt.axvline(itof_gate[1], color='k', linestyle='--')
    plt.legend()

    mask1 = ((t > 500) & (t<700)) | ((t > 200) & (t<400))| (t<2000)
    # mask1 = (t > 200) & (t<400)
    mask2 = (etof > 495) & (etof<497) if calibration is None else (etof > 0) & (etof<20000)
    mask = mask1 & mask2
    if itof_gate:
        mask = mask & (itof > itof_gate[0]) & (itof < itof_gate[1])

    plt.text(0.25, 0.8, f'Coincidence Rate: {np.sum(mask)/len(mask)*100:.2f}%\nCoincidence Count: {np.sum(mask)}', transform=plt.gca().transAxes, fontsize=12, ha='center')

    if calibration is None:
        xc=x-133.2
        yc=y-131.7
        theta = 0.43
        x_rot = xc*np.cos(theta) - yc*np.sin(theta)
        y_rot = xc*np.sin(theta) + yc*np.cos(theta)

        plt.subplot(323)
        plt.xlabel('Polarization Axis (pixels)')
        plt.ylabel('Propagation Axis (pixels)')
        plt.title('VMI Slice (2 ns e-ToF Window)')

        hist,xe,ye= np.histogram2d(x_rot[mask],y_rot[mask], bins=n, range=((-128, 128), (-128, 128)))
        plt.imshow(hist.T, extent=[xe[0],xe[-1],ye[0],ye[-1]], cmap=cmap, origin='lower', norm=matplotlib.colors.PowerNorm(gamma))
        plt.colorbar()

        plt.subplot(324)
        plt.xlabel('Polarization Axis (pixels)')
        plt.ylabel('Propagation Axis (pixels)')
        plt.title('VMI Slice (2 ns e-ToF Window, Log Scale)')

        plt.imshow(hist.T, extent=[xe[0],xe[-1],ye[0],ye[-1]], cmap=cmap, origin='lower',norm=matplotlib.colors.LogNorm())
        plt.colorbar()

        plt.subplot(313)
        plt.title('Radial Distribution')
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Counts')
        plt.grid()

        r=np.sqrt(x_rot[mask]**2+y_rot[mask]**2)
        hist,re= np.histogram(r, bins=n, range=(0, 128))
        plt.plot(re[:-1], hist)

    else:
        name+='_calibrated'
        pmax=0.7
        px,py,pz=calibration(x,y,etof,symmetrize=False)
        scipy.io.savemat(name+'.mat', {'px':px[mask], 'py':py[mask], 'pz':pz[mask]})

        plt.subplot(323)
        plt.xlabel('Polarization Axis (AU)')
        plt.ylabel('Propagation Axis (AU)')
        plt.title('VMI Slice (Detector Plane, 0.1 AU width)')
        pz_mask= np.abs(pz)<0.05
        plt.hist2d(px[mask&pz_mask], py[mask&pz_mask], bins=n, range=((-pmax, pmax), (-pmax, pmax)), cmap=cmap, norm=matplotlib.colors.PowerNorm(gamma))
        plt.gca().set_aspect('equal')
        plt.colorbar()

        plt.subplot(324)
        plt.xlabel('Polarization Axis (AU)')
        plt.ylabel('Detector Axis (AU)')
        plt.title('VMI Slice (Polarization Plane, 0.1 AU width)')
        py_mask= np.abs(py)<0.05
        plt.hist2d(px[mask&py_mask], pz[mask&py_mask], bins=n, range=((-pmax, pmax), (-pmax, pmax)), cmap=cmap, norm=matplotlib.colors.PowerNorm(gamma))
        plt.gca().set_aspect('equal')
        plt.colorbar()

        plt.subplot(313)
        plt.title('Radial Distribution')
        plt.xlabel('Radius (AU)')
        plt.ylabel('Counts')
        plt.grid()

        r=np.sqrt(px[mask]**2+py[mask]**2+pz[mask]**2)
        hist,re= np.histogram(r, bins=n, range=(0, pmax))
        plt.plot(re[:-1], hist)


    plt.tight_layout()
    plt.show()
    plt.savefig(name+'.png')

if __name__ == '__main__':
    fname=r"J:\ctgroup\Edward\DATA\VMI\20250121\xe_ellipticity"
    for f in os.listdir(fname):
        if f.endswith('.cv4'):
            analyze_file(os.path.join(fname, f))