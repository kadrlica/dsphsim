#!/usr/bin/env python
"""
Interface to instrument characteristics.
"""

import os,sys
from collections import OrderedDict as odict
import inspect
import copy

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# Minimum and maximum magnitudes for instruments
MAGMIN = 15
MAGMAX = 27
EXPMAX = 1e8

def factory(name, **kwargs):
    from ugali.utils.factory import factory
    return factory(name, module=__name__, **kwargs)

class Instrument(object):
    """ Base class for various spectroscopic instruments. """
    _datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
    _filename = 'inst.dat'
    _exptime0 = 36000.
    _defaults = odict([
        ('vsys' , 2.0),    # Systematic eror (km/s)
        ('fov',   50),     # Field of View (arcmin^2)
        ('nstar', 50),     # Number of stars per pointing
        ('overhead', 30),  # Overhead per exposure (s)
    ])
    MAGMIN=MAGMIN
    MAGMAX=MAGMAX
    EXPMAX=EXPMAX

    def __init__(self, **kwargs):
        self._setup(**kwargs)

    def _setup(self,**kwargs):
        defaults = self.defaults()
        for k,v in kwargs.items():
            if k not in defaults:
                msg = "Keyword argument not found in defaults"
                raise KeyError(msg)

        for k,v in defaults.items():
            kwargs.setdefault(k,v)
        
        self.__dict__.update(kwargs)

        #self.MAGMIN = MAGMIN
        #self.MAGMAX = MAGMAX

    @classmethod
    def defaults(cls):
        return copy.deepcopy(cls._defaults)

    @classmethod
    def mag2snr(cls, mag, exp=1000.):
        """Convert stellar magnitude into SNR for given exposure time."""
        # magnitude to snr file
        datafile = os.path.join(cls._datadir,cls._filename)
        # ADW: We may not want to read the file every time?
        _mag,_snr = np.genfromtxt(datafile,usecols=[0,1]).T
        exp0 = cls._exptime0
        interp = interp1d(_mag,_snr,bounds_error=False)
        return interp(mag) * np.sqrt(exp/exp0)

    @classmethod
    def maglim2exp(cls, maglim, snr=5):
        """Convert 5-sigma limiting magnitude into exposure time"""
        f = lambda e,m: cls.mag2snr(m,e) - snr
        _mag = np.linspace(cls.MAGMIN,cls.MAGMAX,25)
        _exp = [brentq(f,0,cls.EXPMAX,args=(_m)) for _m in _mag]
        interp = interp1d(_mag,_exp,bounds_error=False)
        return interp(maglim)

    @classmethod
    def exp2maglim(cls, exp, snr=5):
        """Convert exposure time into 5-sigma limiting magnitude"""
        _mag = np.linspace(MAGMIN,MAGMAX,25)
        _exp = cls.maglim2exp(_mag)
        interp = interp1d(_exp,_mag,bounds_error=False)
        return interp(exp)


class GMACS(Instrument):
    """Giant Magellan Telescope Multi-object Astronomical and
    Cosmological Spectrograph (GMACS)
    http://instrumentation.tamu.edu/gmacs.html

    Assume a systematic floor of 2.0 km/s
    """

    _filename = 'gmacs_i.dat'
    _defaults = odict([
        ('vsys',2.0),
        ('fov', 50),
        ('nstar', 50),
        ('overhead', 80),
   ])

    @classmethod
    def snr2err(cls, snr):
        """ copying from DEIMOS
         assuming that DEIMOS and GMACS will have the same spectral
         resolution and thus same snr vs v_error relation"""
        a,b = -1.2, 1.5
        return 10**(a*np.log10(snr) + b)

class IMACS(Instrument):
    """Inamori-Magellan Areal Camera & Spectrograph (IMACS)
    http://www.lco.cl/telescopes-information/magellan/instruments/imacs/imacs-specs
    http://instrumentation.obs.carnegiescience.edu/ccd/imacs/ccd.html

    Assume a systematic floor of 1.5 km/s
    """

    _filename = 'imacs_i.dat'
    _defaults = odict([
        ('vsys',1.5),
        ('fov', 239),
        ('nstar', 50),
        ('overhead', 80),
    ])

    @classmethod
    def snr2err(cls, snr):
        """ ***COPIED FROM GMACS***
        """
        a,b = -1.2, 1.5
        return 10**(a*np.log10(snr) + b)

class DEIMOS(Instrument):
    """DEep Imaging Multi-Object Spectrograph (DEIMOS)
    http://www2.keck.hawaii.edu/inst/deimos/specs.html
    https://www2.keck.hawaii.edu/inst/deimos/primer.html
    """

    _filename = 'deimos_i.dat'
    _defaults = odict([
        ('vsys',  2.0),
        ('fov',   83.5),
        ('nstar', 40),
        ('overhead', 80),
    ])

    @classmethod
    def snr2err(cls, snr):
        """ This function converts from signal-to-noise ration (SNR)
        to statistical velocity uncertainty using a functional fit to
        data in Figure 1 of Simon & Geha 2007. """
        a,b = -1.2, 1.5
        return 10**(a*np.log10(snr) + b)

class M2FS(Instrument):
    """Michigan/Magellan Fiber System (M2FS)
    https://www.cfa.harvard.edu/~kenyon/TAC/M2FS_2013B-2.pdf
    """
    _filename = 'm2fs.dat'
    _exptime0 = 7200.
    _defaults = odict([
        ('vsys',  0.9),
        ('fov',   706.9),
        ('nstar', 256),
        ('overhead', 30),
    ])
    EXPMAX=1e11
    
    @classmethod
    def snr2err(cls, snr):
        # This is for M2FS based on Simon et al. 2015
        # derived in a similar way as for DEIMOS
        a,b = -1.04, 0.7745
        return 10**(a*np.log10(snr) + b)

class GIRAFFE(Instrument):
    """GIRAFFE on VLT

    Assume a systematic floor of 0.5km/s (need to verify)
    """

    _filename = 'giraffe_i.dat'
    _defaults = odict([
        ('vsys',0.5),
        ('fov',0.136),
        ('nstar',132),
        ('overhead', 30),
    ])

    @classmethod
    def snr2err(cls, snr):
        # copying from DEIMOS, Need to be changed later
        a,b = -1.2, 1.5
        return 10**(a*np.log10(snr) + b)

class AAOmega(Instrument):
    """AAOmega/2dF on the AAT
    http://ftp.aao.gov.au/cgi-bin/aaomega_sn.cgi
    """
    _filename = 'aaomega_i.dat'
    _defaults = odict([
        ('vsys',0.9),
        ('fov', 3600.0),
        ('nstar',400),
        ('overhead', 30),
    ])

    @classmethod
    def snr2err(cls, snr):
        # This is for M2FS based on Simon et al. 2015
        # derived in a similar way as for DEIMOS
        # ***NEED TO UPDATE THIS FOR AAOMEGA***
        a,b = -1.04, 0.7745
        return 10**(a*np.log10(snr) + b)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()
