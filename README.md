[![Build](https://img.shields.io/travis/kadrlica/dsphsim.svg)](https://travis-ci.org/kadrlica/dsphsim)
[![PyPI](https://img.shields.io/pypi/v/dsphsim.svg)](https://pypi.python.org/pypi/dsphsim)
[![Release](https://img.shields.io/github/release/kadrlica/dsphsim.svg)](../../releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../)

Simulate dwarf galaxies
=======================

This tool relies on the [Ultra-faint Galaxy Likelihood (ugali)](https://github.com/DarkEnergySurvey/ugali) toolkit for simulating the spatial and color-magnitude distributions of stars in dwarf galaxies. Stellar velocities are simulated via a numerical integration of the Eddington equation.

Installation
------------

Installation is not the easiest thing in the world, but you can check out the [travis.yml](.travis.yml).

Supported Instruments
---------------------

* `DEIMOS` on Keck
* `IMACS` on Magellan
* `GIRAFFE` on VLT
* `AAOmega/2dF` on AAT
* `M2FS` on Magellan
* `GMACS` for GMT

Output Column Descriptions
---------------------------

The output of the `dsphsim` executable is an ascii table with the following columns.

| Column | Unit | Description | 
| ------ | ---- | ----------- |
| RA | deg | Right ascension |
| DEC | deg | Declination |
| MAG_G | mag | DES g-band magnitude | 
| MAG_I | mag | DES i-band magnitude |
| ANGSEP | deg | Angular separtion from dwarf centroid |
| RPROJ | kpc | Projected radial separation from dwarf centroid |
| SNR |  | Simulated signal-to-noise ratio for spectroscopy |
| VTRUE | km/s | True simulated random velocity drawn from the underlying distribution |
| VSTAT | km/s | Adjustment to true velocity from statistical measurement uncertainty related to the brightness of each star. |
| VSYS | km/s | Adjustment to true velocity from instrumental systematic uncertainty |
| VMEAS | km/s | Velocity measured by instrument <br/> VMEAS = VTRUE+VSTAT+VSYS |
| VMEASERR | km/s | Measured statistical velocity error |
| VSYSERR | km/s | Assumed instrumental systematic velocity error |
| VERR | km/s | Quadrature sum of VMEASERR and VSYSERR <br/> VERR = sqrt(VMEASERR<sup>2</sup>+VSYSERR<sup>2</sup>)|

The code used to generate these data products can be found in [dsphsim/simulator.py](dsphsim/simulator.py).

This table can be read into a `numpy` array using [`numpy.genfromtxt`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html):
```
import numpy as np
filename = "<your_filename.dat>"
data = np.genfromtxt(filename,names=True,dtype=None)
```
