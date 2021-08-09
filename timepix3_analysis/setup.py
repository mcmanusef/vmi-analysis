from distutils.core import setup

setup(
    name="timepix3_analysis",
    version="0.2",
    author="Amsterdam Scientific Instruments",
    description="Program to convert Tpx3 raw files to HDF5 files",
    author_email="info@amscins.com",
    scripts=["tpx3-analyze"],
    packages=["tpx3_analysis"],
    install_requires=["h5py", "numba", "numpy", "matplotlib"]
)

