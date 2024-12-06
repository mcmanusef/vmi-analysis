#%%
import numpy as np

np.bool = np.bool_
import scipy
import pyvista as pv
from skimage import measure
import matplotlib.pyplot as plt
import cmasher as cmr

cmap = cmr.rainforest

fname = r"D:\Data\11_20\c2h4_p_1,5W_C2H4+_calibrated.mat"
data = scipy.io.loadmat(fname, squeeze_me=True)
px, py, pz = data['px'], data['py'], data['pz']
# px,py,pz=np.vstack([px,-px]),np.vstack([py,-py]),np.vstack([pz,-pz])
pmax = 0.7
mask = (px ** 2 + py ** 2 + pz ** 2) < pmax ** 2
px, py, pz = px[mask], py[mask], pz[mask]
print(px.shape)

hist, *e = np.histogramdd((px, py, pz), bins=256, range=((-pmax, pmax), (-pmax, pmax), (-pmax, pmax)))

hist_smooth = scipy.ndimage.gaussian_filter(hist, 2)
hist = hist_smooth
hist /= hist.max()
hist = hist ** 0.3
print(hist.shape)

data_to_plot = []
#marching cubes
for level in np.linspace(0.1, 0.9, 20):
    verts, faces, _, _ = measure.marching_cubes(hist, level, spacing=(2 * pmax / 1024, 2 * pmax / 1024, 2 * pmax / 1024))
    faces = [(len(f), *f) for f in faces]
    faces = list(np.concatenate(faces))
    data_to_plot.append((level, verts, faces))

plotter = pv.Plotter()
# Create a PyVista mesh
for level, verts, faces in data_to_plot:
    surf = pv.PolyData(verts, faces)
    plotter.add_mesh(surf, color=cmap(level ** 1.5), opacity=level ** 3.5)
# plotter.add_axes_at_origin()
plotter.show()
