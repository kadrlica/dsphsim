#!/usr/bin/env python
import os,sys
from collections import OrderedDict as odict
import inspect
import copy

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

    _defaults = odict([
        ('vsys' , 2.0), # Systematic eror (km/s)
        ('fov',   50),  # Field of View (arcmin^2)
        ('nstar', 50),  # Number of stars per pointing
    ])

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
        return copy.deepcopy(cls._defaults)

    @classmethod
    def snr2err(cls, snr):
        """ This function converts from signal-to-noise ration (SNR)
        to statistical velocity uncertainty using a functional fit to
        data in Figure 1 of Simon & Geha 2007. """
        a,b = -1.2, 1.5
        return 10**(a*np.log10(snr) + b)
        
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
    """Giant Magellan Telescope Multi-object Astronomical and
    Cosmological Spectrograph (GMACS)
    http://instrumentation.tamu.edu/gmacs.html
    """
                      
    _defaults = odict([
        ('vsys',2.0),
        ('fov', 50),
        ('nstar', 50),
    ])

    @classmethod
    def mag2snr(cls, mag, exp=1000.):
        basedir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(basedir,'data','gmacs_i.dat')
        # g-band magnitude to snr file
        _mag,_snr = np.genfromtxt(datafile,usecols=[0,1]).T
        exp0 = 36000.
        interp = interp1d(_mag,_snr,bounds_error=False)
        return interp(mag) * np.sqrt(exp/exp0)
    

class DEIMOS(Instrument):
    """DEep Imaging Multi-Object Spectrograph (DEIMOS)
    http://www2.keck.hawaii.edu/inst/deimos/specs.html
    """

    _defaults = odict([
        ('vsys',2.0),
        ('fov',0.0232),
        ('nstar',40),
    ])

    @classmethod
    def mag2snr(cls, mag, exp=1000.):
        basedir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(basedir,'data','deimos_i.dat')
        # g-band magnitude to snr file
        _mag,_snr = np.genfromtxt(datafile,usecols=[0,1]).T
        exp0 = 36000.
        interp = interp1d(_mag,_snr,bounds_error=False)
        return interp(mag) * np.sqrt(exp/exp0)


class M2FS(Instrument):
    """Michigan/Magellan Fiber System (M2FS)
    """

    _defaults = odict([
        ('vsys',0.5),
        ('fov', 0.196),
        ('nstar',256),
    ])

class GIRAFFE(Instrument):
    """GIRAFFE """

    _defaults = odict([
        ('vsys',0.5),
        ('fov',0.136),
        ('nstar',132),
    ])

    @classmethod
    def mag2snr(cls, mag, exp=1000.):
        basedir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(basedir,'data','giraffe_i.dat')
        # g-band magnitude to snr file
        _mag,_snr = np.genfromtxt(datafile,usecols=[0,1]).T
        exp0 = 36000.
        interp = interp1d(_mag,_snr,bounds_error=False)
        return interp(mag) * np.sqrt(exp/exp0)


if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()
