import mayavi.mlab as mlab
import numpy as np
from skimage import measure
from matplotlib.cm import ScalarMappable as SM
import scipy.io
def axes(lower, upper, center=(0, 0, 0)):
    xx = yy = zz = np.arange(lower, upper, 0.1)
    xc = np.zeros_like(xx) + center[0]
    yc = np.zeros_like(xx) + center[1]
    zc = np.zeros_like(xx) + center[2]
    mlab.plot3d(xc, yy + yc, zc, line_width=0.01, tube_radius=0.005)
    # mlab.text3d(center[0]+upper+0.05, center[1], center[2], "y", scale=0.05)
    mlab.plot3d(xc, yc, zz + zc, line_width=0.01, tube_radius=0.005)
    # mlab.text3d(center[0], center[1]+upper, center[2], "z", scale=0.05)
    mlab.plot3d(xx + zc, yc, zc, line_width=0.01, tube_radius=0.005)
    # mlab.text3d(center[0]+0.025, center[1], center[2]+upper, "x", scale=0.05)


def plot_3d_contours(data, width=1, min_bin=None, max_bin=None, num_bins=10, add_axes=False):
    nbins = len(data[0, 0])
    mlab.figure()
    if max_bin is None:
        max_bin = np.max(data)
    if min_bin is None:
        min_bin = np.min(data)

    cm = SM(cmap='jet').to_rgba(np.array(range(num_bins)) ** 0.7)
    cm[:, 3] = (np.array(range(num_bins)) / num_bins) ** 0.5

    for i in range(num_bins):
        iso_val = (i+1) * (max_bin - min_bin) / (num_bins+1) + min_bin
        print(iso_val)
        verts, faces, _, _ = measure.marching_cubes(
            data, iso_val, spacing=(2 * width / nbins, 2 * width / nbins, 2 * width / nbins))

        mlab.triangular_mesh(verts[:, 0] - width, verts[:, 1] - width, verts[:, 2] - width,
                             faces, color=tuple(cm[i, :3]), opacity=cm[i, 3])
    if add_axes:
        axes(-width*1.2, width*1.2)


if __name__ == "__main__":
    a=scipy.io.loadmat(r"J:\ctgroup\Edward\D3_C_2p5_Ze_1.mat")
    data= np.abs(a['W'][1]) ## Do whatever you need to do to load in your data here as a numpy array
    width = 1  ## Half width of the data (should be the maximum of x/y/z vector, assuming centered at 0)

    plot_3d_contours(data, width=width)
    mlab.axes()
    mlab.show()
