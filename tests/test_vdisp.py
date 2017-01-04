#!/usr/bin/env python
"""
Test fitting the velocity dispersion.
"""

import numpy as np
import pylab as plt
import logging

import dsphsim.instruments
import dsphsim.dwarf

import dsphsim.vdisp
import dsphsim.simulator

# More verbose output...
logging.getLogger().setLevel(logging.DEBUG)

def test_vdisp():
    np.random.seed(0)
     
    # Is the output independent of richness? 
    # It should be, but doesn't look like it is...
    kwargs = dict(distance_modulus=17.5,extension=0.1,rs=1.0,richness=1e5)
    kwargs.update(vmax=50)
     
    dwarf = dsphsim.dwarf.Dwarf(**kwargs)
    tac   = dsphsim.tactician.ObstimeTactician(instrument='M2FS')
     
    simulator = dsphsim.simulator.Simulator()
    data = simulator.simulate(dwarf,tac)
     
    peaks = []
    for i in range(10):
        samples,sampler = dsphsim.vdisp.mcmc(data['VTRUE'],data['VERR'])
        peaks.append(dsphsim.vdisp.kde_peak(samples['sigma']))
        print peaks[-1]
     
    assert np.allclose(peaks,3.3,atol=0.1)

if __name__ == "__main__":
    test_vdisp()
