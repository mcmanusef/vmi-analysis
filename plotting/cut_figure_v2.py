import matplotlib.pyplot as plt
import numpy as np
import scipy

import plotting.niceplot as niceplot
import plotting.error_bars_plot as ebp

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def main():
    wdir = r'C:\Users\mcman\Code\VMI\Data'
    ellipticity = 0.6
    theory_file = 'theory_06.h5'
    experimental_file = 'xe013_e'
    option_1=False
    option_2=False
    option_3=True

    print("Loading Theory")
    print("Loading Experimental Data")
    print("Unpacking")
    
    print("Plotting")
    if option_1:
        fig=plt.figure("Cut Figure",figsize=(15,5))

        ax0=plt.subplot(131)
        plt.text(-0.5,0.45, "$a$",size=24,ma='center')
        ax0.set_ylabel("Minor Axis (a.u.)")
        ax0.set_xlabel("Major Axis (a.u.)")

        ax1=plt.subplot(132)
        plt.text(-0.5,0.45, "$b$",size=24, ma='center')

        ax1.set_xlabel("Major Axis (a.u.)")
        ax1.set(yticklabels=[])

        ax2=plt.subplot(133)
        plt.text(-150,0.525, "$c$",size=24,ma='center')
        ax2.set_xlabel("Angle (Degrees)")
        ax2.set_ylabel("$p_r$ (a.u.)")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()

        gen_cartesian_plot(get_theory_data(ellipticity, theory_file, wdir), ax0, ellipticity)
        gen_cartesian_plot(get_exp_data(ellipticity, experimental_file, wdir), ax1, ellipticity)
        gen_polar_plot(get_exp_data(ellipticity, experimental_file, wdir), ax2)

    if option_2:
        fig2=plt.figure("Cut Figure Option 2",figsize=(15,5))

        ax0=plt.subplot(141)
        plt.text(-0.5,0.45, "$a$",size=24,ma='center')
        ax0.set_ylabel("Minor Axis (a.u.)")
        ax0.set_xlabel("Major Axis (a.u.)")

        ax1=plt.subplot(142)
        plt.text(-0.5,0.45, "$b$",size=24, ma='center')

        ax1.set_xlabel("Major Axis (a.u.)")
        ax1.set(yticklabels=[])

        ax2=plt.subplot(143)
        plt.text(-0.5,0.45, "$c$",size=24, ma='center')
        ax2.set_xlabel("Major Axis (a.u.)")
        ax2.set(yticklabels=[])

        ax3=plt.subplot(144)
        plt.text(-0.5,0.45, "$d$",size=24, ma='center')
        ax3.set_xlabel("Major Axis (a.u.)")
        ax3.set(yticklabels=[])
        plt.tight_layout()

        gen_cartesian_plot(get_theory_data(ellipticity, theory_file, wdir), ax0, ellipticity)
        gen_cartesian_plot(get_exp_data(ellipticity, experimental_file, wdir), ax1, ellipticity)

        gen_cartesian_plot(get_theory_data(0.3, "theory_03_3.h5", wdir), ax2, 0.3)
        gen_cartesian_plot(get_exp_data(0.3, "xe011_e", wdir), ax3, 0.3)

    if option_3:
        plt.figure("Row Normalized")
        ax=plt.subplot(111)
        ax.set_xlabel("Angle (Degrees)")
        ax.set_ylabel("$p_r$ (a.u.)")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        gen_row_norm_plot(get_theory_data(ellipticity, theory_file, wdir), ax)
    print("Done")


def gen_polar_plot(data, axis, row_norm=False):
    inter_r, inter_theta, intra_r, intra_theta, px, py = unpack(data)
    pr,theta=np.sqrt(px**2+py**2),np.degrees(np.arctan2(py,px))
    hist,xe,ye=np.histogram2d(theta, pr, bins=256, range=[[-180, 180], [0, 0.6]], density=True)
    hist=scipy.ndimage.gaussian_filter(hist, sigma=3)
    axis.pcolormesh(xe,ye,hist.T, cmap=niceplot.trans_jet())
    axis.set_aspect(360/0.6, 'box')
    plt.plot(np.degrees(inter_theta),inter_r, color='k', linewidth=3)
    plt.plot(np.degrees(ebp.unwrap(intra_theta, start_between=(-np.pi, 0), period=np.pi)),intra_r, color='k', linewidth=3)
    axis.grid(visible=True)
    axis.set_axisbelow(True)


def gen_cartesian_plot(data, axis, ellipticity):
    inter_r, inter_theta, intra_r, intra_theta, px, py = unpack(data)
    niceplot.make_fig(axis, py, px, ellipse=True, bins=256, text=False, ell=ellipticity, width=0.6,blurring=1)

    axis.plot(inter_r * np.sin(inter_theta), inter_r * np.cos(inter_theta),color='k', linewidth=3)
    inter_theta_unwrapped = ebp.unwrap(intra_theta, start_between=(np.pi, 2 * np.pi), period=np.pi)
    axis.plot(intra_r * np.sin(inter_theta_unwrapped), intra_r * np.cos(inter_theta_unwrapped), color='k', linewidth=3)
    axis.grid(visible=True)
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

def gen_row_norm_plot(data, axis):
    inter_r, inter_theta, intra_r, intra_theta, px, py = unpack(data)
    pr,theta=np.sqrt(px**2+py**2),np.degrees(np.arctan2(py,px))%180
    hist,xe,ye=np.histogram2d(theta, pr, bins=256, range=[[0, 180], [0, 1]], density=True)
    hist=scipy.ndimage.gaussian_filter(hist, sigma=1)
    plt.axvline(90, color='grey')
    # plt.plot(45*np.sum(hist, axis=0)/np.nanmax(np.sum(hist, axis=0)),ye[:-1], color='k')
    hist2 = np.sqrt(hist/np.max(hist, axis=0, keepdims=True))

    axis.pcolormesh(xe,ye,hist2.T, alpha=(hist.T/np.max(hist))**0.5,cmap='jet')

    axis.set_aspect(180, 'box')
    axis.grid(visible=True)
    axis.set_axisbelow(True)


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
