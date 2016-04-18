#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import numpy as np
import pylab as plt

import dsphsim.tactician
import dsphsim.dsphsim
import dsphsim.instruments
import dsphsim.dwarf

dwarf = dsphsim.dwarf.Dwarf()
mag_1, mag_2, lon, lat = dwarf.simulate()
mags = mag_1
data = np.rec.fromarrays([mag_1,lon,lat],names=['mag','lon','lat'])
inst = dsphsim.instruments.factory('GMACS')
obstime = 3600.

basetac = dsphsim.tactician.Tactician(inst)
maxtac = dsphsim.tactician.MaxStarsTactician(inst)
smartac = dsphsim.tactician.SmartTactician(inst)

out = []
times = np.linspace(1000,36000,36)
x,y,z = [],[],[]
for obstime in times:
    print "Observation time: %s"%obstime
    x += [(basetac.schedule(data,obstime)>5).sum()]
    y += [(maxtac.schedule(data,obstime)>5).sum()]
    z += [(smartac.schedule(data,obstime)>5).sum()]

plt.figure()
plt.plot(times,x,label='Single Exposure')
plt.plot(times,y,label='Variable Exposure')
plt.plot(times,z,label='Equal Exposure')
plt.legend(loc='lower right',fontsize=12)
plt.xlabel("Observation Time (s)")
plt.xlabel("Number of Stars (S/N > 5)")
plt.ion()
