#!/usr/bin/env python
"""
Test fitting the velocity dispersion.
"""

import numpy as np
import pylab as plt

import dsphsim.instruments
import dsphsim.dwarf

import dsphsim.vdisp
import dsphsim.simulator

from ugali.utils.projector import angsep

np.random.seed(0)

kwargs = dict(distance_modulus=17.5,extension=0.1,rs=1.0,richness=1e4)
kwargs.update(vmax=50)

dwarf = dsphsim.dwarf.Dwarf(**kwargs)
tac   = dsphsim.tactician.ObstimeTactician(instrument='M2FS')

simulator = dsphsim.simulator.Simulator()
data = simulator.simulate(dwarf,tac)

for i in range(10):
    samples = dsphsim.vdisp.mcmc(data['VTRUE'],data['VERR'])
    print samples.peak('sigma')
