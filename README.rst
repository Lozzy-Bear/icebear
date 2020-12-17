README
======
ICEBEAR-3D processing, imaging and plotting python package.

DEPENDENCIES
============
numpy
scipy
matplotlib
time
h5py
ctypes
pyfftw
numba

=========

Install Instructions:

It is recommended to use a virtual environment for the ICEBEAR software.  Navigate to the package directory and run:

```
pip3 install virtualenv
```

```
virtualenv icebear_devel
```

This then creates the virtual environment in the directory.  To activate the environment, one can use:

```
source icebear_devel/bin/activate
```

This then activates the virtual environment.  The packages that are installed are now only implemented with this environment.  This helps keep software dependencies modularized and prevents issues with different software packages.  The following command installs the required packages for the icebear software package.

```
pip3 install numpy h5py scipy pyyaml python-dateutil matplotlib install-qt-binding imageio
```

```
pip3 install .
```
