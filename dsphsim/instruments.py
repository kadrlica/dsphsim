#!/usr/bin/env python
import os,sys
from collections import OrderedDict as odict
import inspect

import numpy as np
from scipy.interpolate import interp1d


def factory(name, **kwargs):
    """
    Factory for creating spatial kernels. Arguments
    are passed directly to the constructor of the chosen
    kernel.
    """
    fn = lambda member: inspect.isclass(member) and member.__module__==__name__
    classes = odict(inspect.getmembers(sys.modules[__name__], fn))
    members = odict([(k.lower(),v) for k,v in classes.items()])
    
    namelower = name.lower()
    if namelower not in members.keys():
        msg = "%s not found in kernels:\n %s"%(name,classes.keys())
        #logger.error(msg)
        print msg
        msg = "Unrecognized kernel: %s"%name
        raise Exception(msg)
 
    return members[namelower](**kwargs)


class Instrument(object):
    """ Base class for various spectroscopic instruments. """

    defaults = (
        ('vsys' , 2.0, 'Systematic eror (km/s)'),
        )

    def __init__(self, **kwargs):
        self._setup(**kwargs)

    def _setup(self,**kwargs):
        defaults = self.default_dict()
        for k,v in kwargs:
            if k not in defaults:
                msg = "Keyword argument not found in defaults"
                raise KeyError(msg)

        for k,v in defaults.items():
            kwargs.setdefault(k,v)
        
        self.__dict__.update(kwargs)

    @classmethod
    def default_dict(cls):
        return odict([(d[0],d[1]) for d in cls.defaults])

    @classmethod
    def maglim2exp(cls, maglim):
        """ Convert exposure into 5sigma mag limit """
        f = lambda e,m: cls.mag2snr(m,e) - 5
        return brentq(f,0,1e5,args=(maglim))

    @classmethod
    def exp2maglim(cls, exp):
        f = lambda m,e: cls.mag2snr(m,e) - 5
        return brentq(f,16,27,args=(e))

class GMACS(Instrument):
    """ GMACS, assuming the systematic floor is 2.0 km/s """

    defaults = (
        ('vsys',2.0, 'Systematic error (km/s)'),
        )

    @classmethod
    def mag2snr(cls, mag, exp=1000.):
        basedir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(basedir,'data','gmacs_i.dat')
        # g-band magnitude to snr file
        _mag,_snr = np.genfromtxt(datafile,usecols=[0,1]).T
        exp0 = 36000.
        interp = interp1d(_mag,_snr,bounds_error=False)
        return interp(mag) * np.sqrt(exp/exp0)

    @classmethod
    def snr2err(cls, snr):
        """ copying from DEIMOS
         assuming that DEIMOS and GMACS will have the same spectral
         resolution and thus same snr vs v_error relation"""
        a,b = -1.2, 1.5
        return 10**(a*np.log10(snr) + b)

class DEIMOS(Instrument):
    """ DEIMOS """

    defaults = (
        ('vsys',2.2, 'Systematic error (km/s)'),
        )

    @classmethod
    def mag2snr(cls, mag, exp=1000.):
        basedir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(basedir,'data','deimos_i.dat')
        # g-band magnitude to snr file
        _mag,_snr = np.genfromtxt(datafile,usecols=[0,1]).T
        exp0 = 36000.
        interp = interp1d(_mag,_snr,bounds_error=False)
        return interp(mag) * np.sqrt(exp/exp0)

    @classmethod
    def snr2err(cls, snr):
        """ This function converts from signal-to-noise ration (SNR)
        to statistical velocity uncertainty using a functional fit to
        data in Figure 1 of Simon & Geha 2007. """
        a,b = -1.2, 1.5
        return 10**(a*np.log10(snr) + b)

class M2FS(Instrument):
    """ M2FS """

    defaults = (
        ('vsys',0.9, 'Systematic error (km/s)'),
        )

    @classmethod
    def mag2snr(cls, mag, exp=1000.):
        basedir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(basedir,'data','m2fs.dat')
        # g-band magnitude to snr file
        _mag,_snr = np.genfromtxt(datafile,usecols=[0,1]).T
        exp0 = 7200.
        interp = interp1d(_mag,_snr,bounds_error=False)
        return interp(mag) * np.sqrt(exp/exp0)

    @classmethod
    def snr2err(cls, snr):
        # This is for M2FS based on Simon et al. 2015
        # derived in a similar way as for DEIMOS
        a,b = -1.04, 0.7745
        return 10**(a*np.log10(snr) + b)

class GIRAFFE(Instrument):
    """ GIRAFFE, assuming systematic floor is 0.5km/s, need to be verified later """

    defaults = (
        ('vsys',0.5, 'Systematic error (km/s)'),
        )

    @classmethod
    def mag2snr(cls, mag, exp=1000.):
        basedir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(basedir,'data','giraffe_i.dat')
        # g-band magnitude to snr file
        _mag,_snr = np.genfromtxt(datafile,usecols=[0,1]).T
        exp0 = 36000.
        interp = interp1d(_mag,_snr,bounds_error=False)
        # Somewhere around here would be a place to hack the SNR given
        # the finite FoV and coverage fraction of each instrument...
        nstar = len(np.asarray(mag))
        #...

        return interp(mag) * np.sqrt(exp/exp0)

    @classmethod
    def snr2err(cls, snr):
        # copying from DEIMOS, Need to be changed later
        a,b = -1.2, 1.5
        return 10**(a*np.log10(snr) + b)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()
