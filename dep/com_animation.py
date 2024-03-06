import itertools
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import scipy.stats
from PyQt5 import QtWidgets
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

mpl.rc('image', cmap='jet')
mpl.use('Qt5Agg')

from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """

    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                              [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h / 2 / (r + w / 2)):
                    return np.sqrt((r + w / 2) ** 2 + (np.tan(a) * (r + w / 2)) ** 2)
                else:
                    c = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                    T = np.arcsin(c * np.cos(np.pi / 2 - a + np.arcsin(h / 2 / c)) / r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w / 2, h / 2])
                    return np.sqrt(np.sum(xy ** 2))

            def R(a, r, w, h):
                aa = (a % (np.pi / 4)) * ((a % (np.pi / 2)) <= np.pi / 4) + \
                     (np.pi / 4 - (a % (np.pi / 4))) * ((a % (np.pi / 2)) >= np.pi / 4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2 * a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X - s / 2), 0))[0] * 72
            self.text.set_position([offs * np.cos(angle), offs * np.sin(angle)])


def trans_jet(opaque_point=0.1):
    # Create a colormap that looks like jet
    cmap = plt.cm.jet

    # Create a new colormap that is transparent at low values
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:int(cmap.N * opaque_point), -1] = np.linspace(0, 1, int(cmap.N * opaque_point))
    return ListedColormap(cmap_colors)


def unpack(data):
    coords, inter_data, intra_data = data
    px, py, pz = to_flat_arrays(coords[0])
    inter_r, _, inter_theta, _ = to_flat_arrays(zip(*inter_data))
    intra_r, _, intra_theta, _ = to_flat_arrays(zip(*intra_data))
    return inter_r, inter_theta, intra_r, intra_theta, px, py


def get_datapoints(angle): return (0.1 * np.cos(angle), 0.1 * np.sin(angle))


def to_flat_arrays(iter):
    out = tuple(map(lambda x: x.flatten(), map(np.asarray, iter)))
    return out


def main():
    wdir = r'C:\Users\mcman\Code\VMI\Data'
    theory_file = 'theory_03.h5'
    to_load = 100000
    with h5py.File(os.path.join(wdir, theory_file)) as f:
        px, py = f["y"][::100], f["x"][::100]
    px, py = tuple(zip(*((x, y) for x, y in zip(px, py) if 0.15 < np.sqrt(x ** 2 + y ** 2) < 0.3)))
    r, theta = to_flat_arrays(tuple(zip(*(((x ** 2 + y ** 2) ** 0.5, np.arctan2(y, x)) for x, y in zip(px, py)))))

    alpha = 0
    fig, ax = plt.subplots(1)
    artists = []

    for i in range(5):
        plot_full(px, py, ax, artists)

    for i in range(5):
        plot_line(r, theta, ax, artists, alpha)

    for i in range(20):
        plot_cut(r, theta, ax, artists, 1 + i / 19, alpha)

    for i in range(20):
        plot_cut(r, theta, ax, artists, 2 - i / 19, alpha, line=True)

    for i in range(5):
        plot_line(r, theta, ax, artists, alpha, line=True)

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=40)
    plt.show()
    ani.save(filename=os.path.join(wdir, "video.html"), writer="html")


def plot_full(px, py, ax, artists, first=False, n=256):
    ax.set(xticks=[], yticks=[])
    bounds = [(-0.5, 0.5), ] * 2
    *_, hist = ax.hist2d(px, py, bins=n, range=bounds, cmap=trans_jet())
    # l1 = ax.axhline(0, color='black', linewidth=1)
    # l2 = ax.axvline(0, color='black', linewidth=1)
    if first:
        artists.append(ax.get_children())
    else:
        artists.append([hist])


def plot_line(r, theta, ax, artists, cut_angle, line=False, cut_line=True, n=256, ret=False):
    arts = []
    ax.set(xticks=[], yticks=[])
    bounds = [(-0.5, 0.5), ] * 2
    r_cut,theta_cut = to_flat_arrays(zip(*((a,(b-cut_angle)%(2*np.pi)+cut_angle) for a,b in zip(r,theta) if 0 < (b-cut_angle)%(2*np.pi) < np.pi))) # (theta - cut_angle) % np.pi + cut_angle
    *_, hist = ax.hist2d(r * np.cos(theta), r * np.sin(theta), bins=n, range=bounds, cmap=trans_jet())
    arts.append(hist)
    if cut_line:
        arts.append(ax.plot([-0.4 * np.cos(cut_angle), 0.4 * np.cos(cut_angle)],
                            [-0.4 * np.sin(cut_angle), 0.4 * np.sin(cut_angle)],
                            color='black', linewidth=3)[0])
    # arts.append(ax.axhline(0, color='black', linewidth=1))
    # arts.append(ax.axvline(0, color='black', linewidth=1))
    if line:
        mean_angle = scipy.stats.circmean(theta_cut * 2)/2
        arts.append(ax.plot([-0.4 * np.cos(mean_angle), 0.4 * np.cos(mean_angle)],
                            [-0.4 * np.sin(mean_angle), 0.4 * np.sin(mean_angle)],
                            color='red', linewidth=3)[0])

    artists.append(arts)
    if ret:
        return mean_angle


def plot_cut(r, theta, ax, artists, scale, cut_angle, line=False, n=256):
    arts = []
    ax.set(xticks=[], yticks=[])
    bounds = [(-0.5, 0.5), ] * 2
    r_cut,theta_cut = to_flat_arrays(zip(*((a,(b-cut_angle)%(2*np.pi)+cut_angle) for a,b in zip(r,theta) if 0 < (b-cut_angle)%(2*np.pi) < np.pi)))  # (theta - cut_angle) % np.pi + cut_angle

    *_, hist = ax.hist2d(r_cut * np.cos(theta_cut * scale), r_cut * np.sin(theta_cut * scale), bins=n, range=bounds,
                         cmap=trans_jet())
    arts.append(hist)
    arts.append(
        ax.plot([0, 0.4 * np.cos(scale * cut_angle)], [0, 0.4 * np.sin(scale * cut_angle)], color='black', linewidth=3)[
            0])
    arts.append(ax.plot([0, 0.4 * np.cos(scale * (cut_angle + np.pi))], [0, 0.4 * np.sin(scale * (cut_angle + np.pi))],
                        color='black', linewidth=3)[0])
    # arts.append(ax.axhline(0, color='black', linewidth=1))
    # arts.append(ax.axvline(0, color='black', linewidth=1))
    if line:
        mean_angle = scipy.stats.circmean(theta_cut * 2) / 2
        arts.append(ax.plot([0, 0.4 * np.cos(scale * mean_angle)], [0, 0.4 * np.sin(scale * mean_angle)], color='red',
                            linewidth=3)[0])
    artists.append(arts)


def plot_hist(r, theta, ax, artists, cut_angle, lines=False, line=False, n=256):
    arts = []
    ax.set(xticks=[], yticks=[])
    bounds = [(-0.5, 0.5), ] * 2
    r_cut,theta_cut = to_flat_arrays(zip(*((a,(b-cut_angle)%(2*np.pi)+cut_angle) for a,b in zip(r,theta) if 0 < (b-cut_angle)%(2*np.pi) < np.pi)))  # (theta - cut_angle) % np.pi + cut_angle
    *_, hist = ax.hist2d(r_cut * np.cos(theta_cut * 2), r_cut * np.sin(theta_cut * 2), bins=n, range=bounds,
                         cmap=trans_jet())
    arts.append(hist)

    xhist,xe=np.histogram(r_cut * np.cos(theta_cut * 2), bins=n, range=bounds[0], density=True)
    yhist,ye=np.histogram(r_cut * np.sin(theta_cut * 2), bins=n, range=bounds[1], density=True)

    xhist=xhist/max(xhist)
    yhist=xhist/max(yhist)

    x=np.asarray([a/2+b/2 for a,b in itertools.pairwise(xe)])
    y=np.asarray([a/2+b/2 for a,b in itertools.pairwise(xe)])

    arts.append(ax.plot(x,0.2*xhist-0.5, color='red', linewidth=2)[0])
    arts.append(ax.plot(0.2*yhist-0.5,y, color='red', linewidth=2)[0])

    if lines:
        xm=np.mean(r_cut * np.cos(theta_cut * 2))
        ym=np.mean(r_cut * np.sin(theta_cut * 2))
        arts.append(ax.axvline(xm,color='red',linewidth=1))
        arts.append(ax.axhline(ym,color='red',linewidth=1))

    # arts.append(ax.axhline(0, color='black', linewidth=1))
    # arts.append(ax.axvline(0, color='black', linewidth=1))

    if line:
        mean_angle = scipy.stats.circmean(theta_cut * 2) / 2
        arts.append(ax.plot([0, 0.4 * np.cos(2 * mean_angle)], [0, 0.4 * np.sin(2 * mean_angle)], color='red',
                            linewidth=3)[0])
    artists.append(arts)


if __name__ == "__main__":
    wdir = r'C:\Users\mcman\Code\VMI\Data'
    theory_file = 'theory_06.h5'
    with h5py.File(os.path.join(wdir, theory_file)) as f:
        px, py = f["y"][::100], f["x"][::100]

    px, py = tuple(zip(*((x, y) for x, y in zip(px, py) if 0.15 < np.sqrt(x ** 2 + y ** 2) < 0.3)))
    px=np.asarray((*px,*-np.asarray(px)))
    py=np.asarray((*py,*-np.asarray(py)))
    r, theta = to_flat_arrays(tuple(zip(*(((x ** 2 + y ** 2) ** 0.5, np.arctan2(y, x)) for x, y in zip(px, py)))))

    n=256
    m=1
    alpha = 1.6
    fig = mpl.figure.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.subplots(1)
    artists = []
    #
    # for i in range(m):
    #     plot_full(px, py, ax, artists,n=n)
    # print('a')
    #
    # for i in range(m):
    #     plot_line(r, theta, ax, artists, alpha,n=n)
    # print('b')
    #
    # for i in range(4*m):
    #     plot_cut(r, theta, ax, artists, 1 + i / (4*m-1), alpha,n=n)
    # print('c')
    #
    # for i in range(m):
    #     plot_cut(r, theta, ax, artists, 2, alpha,n=n)
    # print('d')
    #
    # for i in range(m):
    #     plot_hist(r,theta, ax, artists, alpha,n=n)
    #
    # for i in range(m):
    #     plot_hist(r,theta, ax, artists, alpha, lines=True,n=n)
    #
    # for i in range(m):
    #     plot_hist(r,theta, ax, artists, alpha, lines=True, line=True,n=n)
    #
    # for i in range(m):
    #     plot_cut(r, theta, ax, artists, 2, alpha, line=True,n=n)
    # print('e')
    #
    # for i in range(4*m):
    #     plot_cut(r, theta, ax, artists, 2 - i / (4*m-1), alpha, line=True,n=n)
    # print('f')
    #
    # for i in range(m):
    #     plot_cut(r, theta, ax, artists, 1, alpha, line=True,n=n)
    # print('g')
    #
    # for i in range(m):
    #     plot_line(r, theta, ax, artists, alpha, line=True,n=n)
    #
    # for i in range(m):
    #     plot_line(r, theta, ax, artists, alpha, line=True, cut_line=False,n=n)
    # print('h')
    #
    # ani = animation.ArtistAnimation(canvas.figure, artists, interval=1000)
    # # canvas.show()
    # # app = QtWidgets.QApplication(sys.argv)
    # # app.exec()
    #
    # #
    # # # plt.show()
    # ani.save(filename=os.path.join(wdir, "a_crit.mp4"), writer="ffmpeg")
    # print("Done")
#%%
    fig,axes= plt.subplots(3,4)
    for i,ax in zip(np.linspace(0,np.pi,num=12), axes.flatten()):
        angle=plot_line(r, theta, ax, artists, i, line=True,n=n, ret=True)
        print(i,angle)

        ax.text(-.45, .45, f"Cut at {np.degrees(i):0.2f}°")
        ax.text(-.45, .40, f"CoM at {np.degrees(angle):0.2f}°")
        # ax.text(-.45, .35, f"(angles measured from major axis)")

        plt.show()
    fig.tight_layout()


#%%
