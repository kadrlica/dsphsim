#!/usr/bin/env python
"""
Simple tool to simulate stellar populations.
"""
__author__ = "Alex Drlica-Wagner"
__email__ = "kadrlica@fnal.gov"
__version__ = "0.1.0"

import os,sys
import numpy as np
from multiprocessing import Pool

import scipy.stats as stats

from dwarf import Dwarf
from instruments import factory as instrumentFactory

np.random.seed(0)

def randerr(size=1,func='normal',**kwargs):
    """ Return a sample from a random variate. """
    kwargs.update(size=size)
    funclower = func.lower()
    if funclower in ['normal','gaussian','gauss']:
        rvs = stats.norm.rvs
    elif funclower in ['uniform']:
        rvs = stats.uniform(-1,2).rvs
    elif funclower in ['lorentzian','cauchy']:
        rvs = stats.cauchy.rvs
    elif funclower in ['delta']:
        rvs = stats.randint(1,2).rvs
    else:
        raise Exception('Unrecognized type: %s'%func)
    return rvs(**kwargs) 


class Simulator(object):
    
    def run(self, num=1, exptime=10000):
        self.create_dwarf()
        self.create_instrument()
        if not hasattr(exptime,'__iter__'): exptime = [exptime]

        out = []
        for e in exptime:
            for i in num:
                data = simulate(dwarf,instrument,exptime)
                out.append(data)
        return out
    
    @staticmethod
    def simulate(dwarf,instrument,exp=10000):
        """ Simulate observation """

        # Set the second band to 'i' (matches CaT lines)
        dwarf.band_1 = 'g'; dwarf.band_2 = 'i'
        mag_1,mag_2,ra,dec = dwarf.simulate()        
        snr = instrument.mag2snr(mag_2,exp)
     
        #olderr = np.seterr(all='ignore')
        sel = (mag_1 > 16) & (snr > 5)
        #np.seterr(**olderr)
       
        nstar = sel.sum()
        mag = mag_1[sel]
        color = (mag_1-mag_2)[sel]
        snr = snr[sel]

        # The true velocity, u, of each star is the sum of the mean velocity and
        # a component from the intrinsic velocity dispersion
        vtrue = dwarf.vmean + dwarf.vdisp*randerr(nstar,'normal')

        # There are two components of the measurement uncertainty on
        # the velocity of each star
        vstaterr = instrument.snr2err(snr)
        vsyserr = instrument.vsys

        # The measured velocity is the true velocity plus a component from the
        # instrumental measurement error
        vstat = vstaterr*randerr(nstar,'normal')
        vsys = vsyserr*randerr(nstar,'normal')

        vmeas = vtrue + vstat + vsys

        # Now assign the measurement error to the statistical error
        vmeaserr = vstaterr

        # The error that is commonly used is the sum of the measurement error
        # and the systematice error estimate in quadrature
        verr = np.sqrt(vstaterr**2 + vsyserr**2)
     
        names = ['RA','DEC','MAG_%s'%dwarf.band_1.upper(),'MAG_%s'%dwarf.band_2.upper(),
                 'SNR','VTRUE','VSTAT','VSYS','VMEAS','VMEASERR','VERR']
        data = [ra[sel],dec[sel],mag_1[sel],mag_2[sel],snr,vtrue,vstat,vsys,vmeas,vmeaserr,verr]
        return np.rec.fromarrays(data,names=names)
    
if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('outfile',nargs='?',
                        help="Optional output file")
    group = parser.add_argument_group('Physical')
    parser.add_argument('--stellar_mass',type=float,default=2000.,
                        help='Stellar mass for simulated satellite (Msun)')
    parser.add_argument('--vmean',type=float,default=60.,
                        help='Mean systemic velocity (km/s)')
    parser.add_argument('--vdisp',type=float,default=3.3,
                        help='Velocity dispersion (km/s)')
    
    group = parser.add_argument_group('Isochrone')
    group.add_argument('--isochrone',type=str,default='Bressan2012',
                        help='Isochrone type.')
    group.add_argument('--distance_modulus',type=float,
                        help='Distance modulus.')
    group.add_argument('--age',type=float,default=13.0,
                       help='Age of stellar population (Gyr).')
    group.add_argument('--metallicity',type=float,default=1e-3,
                       help='Metallicity of stellar population.')
    
    group = parser.add_argument_group('Kernel')
    group.add_argument('--kernel',type=str,default='EllipticalPlummer',
                       help='Kernel type.')
    group.add_argument('--ra',type=float,default=54.0,
                       help='Centroid right acension (deg).')
    group.add_argument('--dec',type=float,default=-54.0,
                       help='Centroid declination (deg).')
    group.add_argument('--extension',type=float,default=0.1,
                       help='Extension (deg).')
    group.add_argument('--ellipticity',type=float,default=0.0,
                       help='Spatial extension (deg).')
    group.add_argument('--position_angle',type=float,default=0.0,
                       help='Spatial extension (deg).')

    group = parser.add_argument_group('Instrument')
    group.add_argument('--instrument',default='gmacs',choices=['gmacs'],
                       help='Instrument')
    egroup = group.add_mutually_exclusive_group()
    egroup.add_argument('--exptime',default=3600.,type=float,
                        help='Exposure time (s)')
    egroup.add_argument('--maglim',default=None,type=float,
                        help='Limiting magnitude (S/N = 5)')
    group.add_argument('--vsys',default=None,type=float,
                       help='Systematic velocity error (km/s)')
    args = parser.parse_args()
    kwargs = vars(args)

    exptime = mag2exp(args.maglim) if args.maglim else args.exptime

    dwarf = Dwarf(vmean=args.vmean,vdisp=args.vdisp)
    isochrone=Dwarf.createIsochrone(name=args.isochrone, age=args.age,
                                    metallicity=args.metallicity,
                                    distance_modulus=args.distance_modulus)
    dwarf.set_isochrone(isochrone)

    kernel=Dwarf.createKernel(name=args.kernel,extension=args.extension,
                              ellipticity=args.ellipticity,
                              position_angle=args.position_angle,
                              lon=args.ra,lat=args.dec)
    dwarf.set_kernel(kernel)
    dwarf.richness = args.stellar_mass/dwarf.isochrone.stellar_mass()


    instr= instrumentFactory(args.instrument)
    if args.vsys is not None: instrument.vsys = args.vsys

    # Run the simulation
    data = Simulator.simulate(dwarf,instr,exptime)

    # Output
    if args.outfile:
        out = open(args.outfile,'w')
    else:
        out = sys.stdout
    out.write('#'+' '.join(['%-9s'%n for n in data.dtype.names])+'\n')
    np.savetxt(out,data,fmt='%-9.5f')
