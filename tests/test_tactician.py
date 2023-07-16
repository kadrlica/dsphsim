#!/usr/bin/env python
"""
Test the exposure tacticians
"""

import os
import numpy as np
import logging

import dsphsim.tactician
import dsphsim.instruments
import dsphsim.dwarf

# More verbose output...
logging.getLogger().setLevel(logging.DEBUG)

def test_tactician():
    dwarf = dsphsim.dwarf.Dwarf()
    mag_1, mag_2, lon, lat, vel = dwarf.simulate()
    mags = mag_1
    data = np.rec.fromarrays([mag_1,lon,lat],names=['mag','lon','lat'])
     
    obstime = 3600.
    times = np.linspace(500,36000,72)
    times = np.logspace(1,5,25)
     
    inst = 'GMACS'
    obstac = dsphsim.tactician.ObstimeTactician(inst)
    dyntac = dsphsim.tactician.DynamicTimeTactician(inst)
    equtac = dsphsim.tactician.EqualTimeTactician(inst)
    numtac = dsphsim.tactician.NStarsTactician(inst)
     
    x,y,z,w = [],[],[],[]
    for obstime in times:
        print("Observation time: %s"%obstime)
        x += [obstac.schedule(data,obstime=obstime)]
        y += [dyntac.schedule(data,obstime=obstime)]
        z += [equtac.schedule(data,obstime=obstime)]
        w += [numtac.schedule(data,nstars=25)]
     
    if os.environ.get('DISPLAY'):
        import pylab as plt
     
        plt.figure()
        plt.loglog(times,[(_x>5).sum() for _x in x],label='Single Exposure')
        plt.loglog(times,[(_x>5).sum() for _x in y],label='Dynamic Exposure')
        plt.loglog(times,[(_x>5).sum() for _x in z],label='Equal Exposure')
        plt.legend(loc='upper left',fontsize=12)
        plt.xlabel("Observation Time (s)")
        plt.ylabel("Number of Stars (S/N > 5)")
        plt.savefig('nstars_vs_exp.png',bbox_inches='tight')
         
        plt.figure()
        plt.loglog(times,[np.nansum(_x) for _x in x],label='Single Exposure')
        plt.loglog(times,[np.nansum(_x) for _x in y],label='Dynamic Exposure')
        plt.loglog(times,[np.nansum(_x) for _x in z],label='Equal Exposure')
        plt.legend(loc='upper left',fontsize=12)
        plt.xlabel("Observation Time (s)")
        plt.ylabel("Sum of S/N")
        plt.savefig('totsnr_vs_exp.png',bbox_inches='tight')
         
        plt.ion()

if __name__ == "__main__":
    test_tactician()
