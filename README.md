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