#!/usr/bin/env python
"""
Simple tool to simulate stellar populations.
"""
__author__ = "Alex Drlica-Wagner"
__email__ = "kadrlica@fnal.gov"
__version__ = "0.1.0"

import os,sys
import numpy as np

import scipy.stats as stats

from dwarf import Dwarf
from instruments import factory as instrumentFactory


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
        mag_1,mag_2,ra,dec,velocity = dwarf.simulate()        
        snr = instrument.mag2snr(mag_2,exp)

        #olderr = np.seterr(all='ignore')
        # Allow user to change these thresholds?
        sel = (mag_1 > 16) & (snr > 5)
        nstar = sel.sum()
        #np.seterr(**olderr)

        ra = ra[sel]
        dec = dec[sel]
        mag_1 = mag_1[sel]
        mag_2 = mag_2[sel]
        sep = dwarf.kernel.angsep(ra,dec)
        vel = velocity[sel]
        snr = snr[sel]

        rproj = dwarf.distance * np.tan(np.radians(sep))
        #mag = mag_1
        #color = (mag_1-mag_2)

        # The true velocity, u, of each star is the sum of the mean velocity and
        # a component from the intrinsic velocity dispersion
        # ADW: The mean velocity is applied in the velocity sampling
        #vtrue = dwarf.vmean + dwarf.vdisp*randerr(nstar,'normal')
        #vtrue = dwarf.vmean + vel
        vtrue = vel

        # There are two components of the measurement uncertainty on
        # the velocity of each star
        vstaterr = instrument.snr2err(snr)
        vsyserr = instrument.vsys * np.ones_like(snr)

        # The measured velocity is the true velocity plus a component from the
        # instrumental measurement error
        vstat = vstaterr*randerr(nstar,'normal')
        vsys = vsyserr*randerr(nstar,'normal')

        vmeas = vtrue + vstat + vsys

        # Now assign the measurement error to the statistical error
        vmeaserr = vstaterr

        # The error that is commonly used is the sum of the measurement error
        # and the systematic error added in quadrature
        verr = np.sqrt(vstaterr**2 + vsyserr**2)

        # Do we also want to save vsyserr as VSYSERR?
        names = ['RA','DEC',
                 'MAG_%s'%dwarf.band_1.upper(),'MAG_%s'%dwarf.band_2.upper(),
                 'ANGSEP','RPROJ','SNR',
                 'VTRUE','VSTAT','VSYS',
                 'VMEAS','VMEASERR','VSYSERR','VERR']
        #data = [ra[sel],dec[sel],mag_1[sel],mag_2[sel],snr,vtrue,vstat,vsys,vmeas,vmeaserr,verr]
        data = [ra, dec,
                mag_1, mag_2,
                sep, rproj, snr,
                vtrue, vstat, vsys,
                vmeas, vmeaserr, vsyserr, verr]
        return np.rec.fromarrays(data,names=names)

    @classmethod
    def parser(cls):
        import argparse
        description = "Simulate the observable properties of a dwarf galaxy."
        formatter = argparse.ArgumentDefaultsHelpFormatter
        parser = argparse.ArgumentParser(description=description,
                                         formatter_class=formatter)
        parser.add_argument('outfile',nargs='?',
                            help="Optional output file")
        parser.add_argument('--seed',type=int,default=None,
                            help="Random seed")
        parser.add_argument('-v','--verbose',
                            help="Output verbosity")
         
        group = parser.add_argument_group('Physical')
        group.add_argument('--stellar_mass',type=float,default=2000.,
                            help='Stellar mass for simulated satellite (Msun)')

        group = parser.add_argument_group('Kinematic')
        group.add_argument('--kinematics',type=str,default='Gaussian',
                           help='Kinematic distribution')
        group.add_argument('--vmean',type=float,default=60.,
                            help='Mean systemic velocity (km/s)')
        # should be mutually exclusive with vmax and rs
        egroup = group.add_mutually_exclusive_group()
        egroup.add_argument('--vdisp',type=float,default=3.3,
                            help='Velocity dispersion (km/s)')
        egroup.add_argument('--vmax',type=float,default=10.0,
                            help='Maximum circular velocity (km/s)')
        egroup = group.add_mutually_exclusive_group()
        egroup.add_argument('--rvmax',type=float,default=None,
                           help='Radius of max circular velocity (kpc)')
        # ADW: it would be nice to remove this
        egroup.add_argument('--rs',type=float,default=None,
                           help='NFW scale radius for DM halo (kpc)')
         
        group = parser.add_argument_group('Isochrone')
        group.add_argument('--isochrone',type=str,default='Bressan2012',
                            help='Isochrone type')
        group.add_argument('--age',type=float,default=12.0,
                           help='Age of stellar population (Gyr)')
        group.add_argument('--metallicity',type=float,default=2e-4,
                           help='Metallicity of stellar population')
        group.add_argument('--distance_modulus',type=float,default=17.5,
                            help='Distance modulus')
         
        group = parser.add_argument_group('Kernel')
        group.add_argument('--kernel',type=str,default='EllipticalPlummer',
                           help='Kernel type')
        group.add_argument('--ra',type=float,default=54.0,
                           help='Centroid right acension (deg)')
        group.add_argument('--dec',type=float,default=-54.0,
                           help='Centroid declination (deg)')
        group.add_argument('--extension',type=float,default=0.1,
                           help='Extension (deg)')
        group.add_argument('--ellipticity',type=float,default=0.0,
                           help='Spatial extension (deg)')
        group.add_argument('--position_angle',type=float,default=0.0,
                           help='Spatial extension (deg)')
         
        group = parser.add_argument_group('Instrument')
        group.add_argument('--instrument',default='gmacs',
                           help='Spectroscopic instrument')
        egroup = group.add_mutually_exclusive_group()
        egroup.add_argument('--exptime',default=3600.,type=float,
                            help='Exposure time (s)')
        egroup.add_argument('--maglim',default=None,type=float,
                            help='Limiting magnitude (S/N = 5)')
        group.add_argument('--vsys',default=None,type=float,
                           help='Systematic velocity error (km/s)')
        return parser
    
if __name__ == "__main__":
    parser = Simulator.parser()
    args = parser.parse_args()
    kwargs = vars(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    exptime = mag2exp(args.maglim) if args.maglim else args.exptime


    dwarf = Dwarf()
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

    # Set the kinematic properties
    if args.rs is not None: args.rvmax = 2.163*args.rs
    kinematics=Dwarf.createKinematics(name=args.kinematics, 
                                      vdisp=args.vdisp, vmean=args.vmean,
                                      vmax=args.vmax, rvmax=args.rvmax)
    dwarf.set_kinematics(kinematics)

    instr= instrumentFactory(args.instrument)
    if args.vsys is not None: instr.vsys = args.vsys

    # Run the simulation
    data = Simulator.simulate(dwarf,instr,exptime)

    # Output
    if args.outfile:
        if os.path.exists(args.outfile): os.remove(args.outfile)
        out = open(args.outfile,'w',1)
    else:
        out = sys.stdout
    out.write('#'+' '.join(['%-9s'%n for n in data.dtype.names])+'\n')
    np.savetxt(out,data,fmt='%-9.5f')
    out.flush()
