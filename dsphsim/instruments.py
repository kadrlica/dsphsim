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
        ('vstat', 3.0, 'Statistical error (km/s)'),
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
    def snr2err(cls, snr):
        """ This function converts from signal-to-noise ration (SNR)
        to statistical velocity uncertainty using a functional fit to
        data in Figure 1 of Simon & Geha 2007. """
        a,b = -1.2, 1.5
        return 10**(a*np.log10(snr) + b)

    @classmethod
    def default_dict(cls):
        return odict([(d[0],d[1]) for d in cls.defaults])
        
    @classmethod
    def snr2err(cls, snr, err=None):
        # Default to returning constant 3 km/s error
        return self.vstat if err is None else err

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
    """ GMACS """

    defaults = (
        ('vsys',2.0, 'Systematic eror (km/s)'),
        )

    @classmethod
    def mag2snr(cls, mag, exp=1000.):
        basedir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(basedir,'..','data','gmacs.dat')
        # g-band magnitude to snr file
        _mag,_snr = np.genfromtxt(datafile,usecols=[0,1]).T
        exp0 = 36000.
        interp = interp1d(_mag,_snr,bounds_error=False)
        return interp(mag) * np.sqrt(exp/exp0)
    

class DEIMOS(Instrument):
    """ DEIMOS """

    defaults = (
        ('vsys',2.0, 'Systematic eror (km/s)'),
        )

class M2FS(Instrument):
    """ M2FS """

    defaults = (
        ('vsys',0.5, 'Systematic eror (km/s)'),
        )

class FLAMES(Instrument):
    """ FLAMES """

    defaults = (
        ('vsys',0.5, 'Systematic eror (km/s)'),
        )

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()
