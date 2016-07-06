#!/usr/bin/env python
"""
Test generating the velocity distribution
"""

import os
import numpy as np

import dsphsim.tactician
import dsphsim.instruments
import dsphsim.dwarf
import dsphsim.velocity

from ugali.utils.projector import angsep

np.random.seed(0)

kwargs = dict(distance_modulus=17.5,extension=0.1,rs=1.0,richness=1e4,
              kinematics='PhysicalVelocity')
kwargs.update(vmax=20)

dwarf = dsphsim.dwarf.Dwarf(**kwargs)
vdist = dwarf.kinematics
mag_1, mag_2, lon, lat, vel = dwarf.simulate()
angsep = dwarf.kernel.angsep(lon,lat)
velocities = dwarf.kinematics.velocities
print "Simulated %s stars"%len(vel)

# Angular separation (deg)
a = np.linspace(angsep.min(),angsep.max())
# Projected radial separation (kpc)
r = np.tan(np.radians(a))*dwarf.distance
# Scaled projected radius (unitless)
x = r/dwarf.rs

v = np.linspace(velocities.min(),velocities.max())
i = np.linspace(0,1)
xx,vv = np.meshgrid(x,v)
junk,ii = np.meshgrid(x,i)

#pdf = dwarf.vdist.interp_pdf(r,v)
#cdf = dwarf.vdist.interp_cdf(r,v)
#icdf = dwarf.vdist.interp_icdf(r,i)

pdf = dwarf.kinematics.interp_pdf(xx,vv)
cdf = dwarf.kinematics.interp_cdf(xx,vv)
icdf = dwarf.kinematics.interp_icdf(xx,ii)

if os.environ.get('DISPLAY'):
    import pylab as plt

    plt.figure()
    plt.pcolormesh(a,v,pdf)
    plt.title('PDF')
    plt.colorbar(label='f(v)')
    plt.xlabel('Angular Separation (deg)')
    plt.ylabel('Velocity (km/s)')
    plt.plot(angsep,vel,'o',color='0.5',mec='0.5',markersize=4,alpha=0.5)
    plt.draw()
    plt.xlim(a.min(),a.max())
    plt.ylim(v.min(),v.max())
     
    # Bins in fraction of the angular extension
    bins = [ (0.0,0.5),
             (0.5,1.0),
             (1.0,2.0),
             (2.0,20.0),
    ]
    kwargs = dict(histtype='step',bins=v,lw=2,normed=True)
    plt.figure()
    for (amin,amax) in bins:
        sel = (angsep > amin*dwarf.extension) & (angsep < amax*dwarf.extension)
        print sel.sum()
        plt.hist(vel[sel],label=r'$%g < r/r_h < %g$'%(amin,amax),**kwargs)
     
    plt.legend(loc='upper right')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Normalized Counts')
     
    """
    plt.figure()
    plt.pcolormesh(a,v,cdf)
    plt.colorbar(label=r'Cumulative f(v)')
    plt.title('CDF')
    plt.xlabel('Angular Separation (deg)')
    plt.ylabel('Velocity (km/s)')
     
    plt.figure()
    plt.pcolormesh(a,i,icdf)
    plt.colorbar(label=r'Velocity (km/s)')
    plt.title('iCDF')
    plt.xlabel('Angular Separation (deg)')
    plt.ylabel('Cumulative f(v)')
    """
     
    plt.ion()

if __name__ == "__main__":
    pass
