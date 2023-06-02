import matplotlib.pyplot as plt
import numpy as np

import plotting.niceplot as niceplot
import plotting.error_bars_plot as ebp


def main():
    wdir = r'C:\Users\mcman\Code\VMI\Data'
    ellipticity = 0.6
    theory_file = 'theory_06.h5'
    experimental_file = 'xe013_e'

    print("Loading Theory")
    theory_data = get_theory_data(ellipticity, theory_file, wdir)

    print("Loading Experimental Data")
    experimental_data = get_exp_data(ellipticity, experimental_file, wdir)

    print("Unpacking")
    
    print("Plotting")
    fig=plt.figure("Cut Figure",figsize=(15,5))
    ax0=plt.subplot(131)
    ax0.set_xlabel("Minor Axis (a.u.)")
    ax0.set_ylabel("Major Axis (a.u.)")

    ax1=plt.subplot(132)
    ax1.set_xlabel("Minor Axis (a.u.)")
    ax1.set(yticklabels=[])

    ax2=plt.subplot(133)
    ax2.set_xlabel("Angle (Degrees)")
    ax2.set_ylabel("$p_r$ (a.u.)")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    gen_cartesian_plot(theory_data, ax0, ellipticity)
    gen_cartesian_plot(experimental_data, ax1, ellipticity)
    gen_polar_plot(experimental_data,ax2)
    print("Done")


def gen_polar_plot(data, axis):
    inter_r, inter_theta, intra_r, intra_theta, px, py = unpack(data)
    pr,theta=np.sqrt(px**2+py**2),np.degrees(np.arctan2(py,px))
    axis.hist2d(theta, pr, bins=256, range=[[-180, 180], [0, 0.6]], cmap=niceplot.trans_jet())
    axis.set_aspect(360/0.6, 'box')
    plt.plot(np.degrees(inter_theta),inter_r, color='m')
    plt.plot(np.degrees(ebp.unwrap(intra_theta, start_between=(-np.pi, 0), period=np.pi)),intra_r, color='k')
    axis.grid()
    axis.set_axisbelow(True)


def gen_cartesian_plot(data, axis, ellipticity):
    inter_r, inter_theta, intra_r, intra_theta, px, py = unpack(data)
    niceplot.make_fig(axis, py, px, ellipse=True, bins=256, text=False, ell=ellipticity, width=0.6)

    axis.plot(inter_r * np.sin(inter_theta), inter_r * np.cos(inter_theta),color='m')
    inter_theta_unwrapped = ebp.unwrap(intra_theta, start_between=(np.pi, 2 * np.pi), period=np.pi)
    axis.plot(intra_r * np.sin(inter_theta_unwrapped), intra_r * np.cos(inter_theta_unwrapped), color='k')
    axis.grid()
    axis.set_axisbelow(True)


def unpack(data):
    coords, inter_data, intra_data = data
    px, py, pz = to_flat_arrays(coords[0])
    inter_r, _, inter_theta, _ = to_flat_arrays(zip(*inter_data))
    intra_r, _, intra_theta, _ = to_flat_arrays(zip(*intra_data))
    return inter_r, inter_theta, intra_r, intra_theta, px, py


def to_flat_arrays(iter):
    out = tuple(map(lambda x: x.flatten(), map(np.asarray, iter)))
    return out


@ebp.cache_results
def get_exp_data(ellipticity, experimental_file, wdir):
    experimental_data = ebp.main([(experimental_file, ellipticity)],
                                 to_load=None,
                                 wdir=wdir,
                                 n=3,
                                 mode='fourier',
                                 electrons='down',
                                 send_data=True,
                                 label='experiment'
                                 )
    return experimental_data


@ebp.cache_results
def get_theory_data(ellipticity, theory_file, wdir):
    theory_data = ebp.main(files=[(theory_file, ellipticity)],
                           wdir=wdir,
                           calibrated=True,
                           n=3,
                           mode='fourier',
                           send_data=True,
                           label="theory"
                           )
    return theory_data


if __name__ == "__main__":
    main()

#%%
