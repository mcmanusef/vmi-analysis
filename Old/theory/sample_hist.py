import datetime
import itertools
import os
import random
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat

mpl.use('Qt5Agg')
plt.ion()


def norm(array):
    return array / np.max(array)


def main(input_file="theory_03_0.mat", source="", output_file="out.h5", sample_factor=1000):
    data = loadmat(os.path.join(source, input_file))
    cut = norm(np.sum(data['hist'][:, (np.abs(data['zv']) < 0.1)[0], :], 1))

    xv = data['xv'][0]
    dx = np.diff(xv)[0]

    yv = data['yv'][0]
    dy = np.diff(yv)[0]

    xs, ys = [], []
    for (x_index, x), (y_index, y) in itertools.product(enumerate(xv), enumerate(yv)):
        xs.extend(random.uniform(x - dx / 2, x + dx / 2) for _ in range(round(sample_factor * cut[y_index, x_index])))
        ys.extend(random.uniform(y - dy / 2, y + dy / 2) for _ in range(round(sample_factor * cut[y_index, x_index])))
    # plt.hist2d(xs,ys,bins=2048,range=[[min(xv),max(xv)],[min(xv),max(xv)]],cmap='jet')
    # plt.show()
    with h5py.File(os.path.join(source, output_file), 'w') as f:
        print(f"Saving {len(xs)} points")
        f.create_dataset('x', data=xs)
        f.create_dataset('y', data=ys)
        f.create_dataset('z', data=np.zeros_like(xs))


if __name__ == "__main__":
    start = datetime.datetime.now()
    print(start)
    for e, i in itertools.product([0.6],range(2)):
        if not os.path.exists(os.path.join(r"C:\Users\mcman\Code\VMI",rf"Data\theory_H_{i}.h5")):
            print(e,i)
            main(input_file=rf"theory\theory_H_{i}.mat", output_file=rf"Data\theory_H_{i}.h5",
                 source=r"C:\Users\mcman\Code\VMI")
    end = datetime.datetime.now()
    print((end - start))
