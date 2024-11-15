import numpy as np
import scipy.ndimage
from scipy.io import loadmat
from mayavi import mlab
from matplotlib.cm import ScalarMappable as SM
from skimage import measure


file=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\Analyzed pdfs and matlab workspaces\xe002_s.mat"
mat_data = loadmat(file)
hist3D = mat_data['3d_hist']

smoothed=scipy.ndimage.gaussian_filter(hist3D,.5)
x,y,z=np.meshgrid(mat_data['xv'],mat_data['yv'],mat_data['zv'])


nbins = len(smoothed[0, 0])
width=np.max(x)
minbin = 1
numbins = 20
numbins = min(numbins, int(smoothed.max())-minbin)

cm = SM(cmap='jet').to_rgba(np.array(range(numbins))**0.7)

cm[:, 3] = (np.array(range(numbins))/numbins)**0.5
mlab.figure(size=(800, 800))
for i in range(numbins):
    iso_val = i*(int(smoothed.max())-minbin)/numbins+minbin

    verts, faces, _, _ = measure.marching_cubes(
        smoothed, iso_val, spacing=(2*width/nbins, 2*width/nbins, 2*width/nbins))

    mlab.triangular_mesh(verts[:, 0]-width, verts[:, 1]-width, verts[:, 2]-width,
                         faces, color=tuple(cm[i, :3]), opacity=cm[i, 3])
mlab.show()
