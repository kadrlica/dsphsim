#!/usr/bin/env python
"""
Choose how to target a field and determine how SNR increases with
exposure time (and number of stars).

"""
import copy
import logging
from collections import OrderedDict as odict

import numpy as np

from dsphsim.instruments import factory as instrumentFactory

def factory(name, **kwargs):
    from ugali.utils.factory import factory
    return factory(name, module=__name__, **kwargs)

class Tactician(object):
    """ Base class for observation tactician. """
    _defaults = odict([
        ('obstime',3600),  # Observation time (s)
        ('snr_thresh',5),  # Signal-to-noise threshold
        ('maglim',None),
        ('nstars',None),
    ])

    def __init__(self, instrument, **kwargs):
        self.name = self.__class__.__name__
        if isinstance(instrument,basestring):
            self.instrument = instrumentFactory(instrument)
        else:
            self.instrument = instrument
        self._setup(**kwargs)

    def __str__(self,indent=0):
        ret = '{0:>{2}}{1}'.format('',self.name,indent)
        ret += '\n{0:>{2}}{1}'.format('','Parameters:',indent+2)
        params = self.defaults().keys()
        width = len(max(params,key=len))
        for key in params:
            value = getattr(self,key)
            par = '{0!s:{width}} : {1!r}'.format(key,value,width=width)
            ret += '\n{0:>{2}}{1}'.format('',par,indent+4)
        return ret
        

    def _setup(self,**kwargs):
        defaults = self.defaults()
        for k,v in kwargs.items():
            if k not in defaults:
                msg = "Keyword argument not found in defaults: %s"%k
                raise KeyError(msg)

        for k,v in defaults.items():
            kwargs.setdefault(k,v)
        
        self.__dict__.update(kwargs)
        
    @classmethod
    def defaults(cls):
        return copy.deepcopy(cls._defaults)

    def schedule(self, data):
        """
        Function to return snr for each star in 'data'
        """
        msg = "schedule must be implemented by subclass"
        raise Exception(msg)

class ObstimeTactician(Tactician): 
    """ Simple tactician that allocates the same observation time to every star.

    """
    def schedule(self, data, **kwargs):
        obstime = kwargs.get('obstime',self.obstime)
        snr_thresh = kwargs.get('snr_thresh',self.snr_thresh)
        snr = self.instrument.mag2snr(data['mag'],obstime)

        num = (np.nan_to_num(snr)>snr_thresh).sum()
        msg = "%s -- ObsTime: %.2f, NStars: %i"%(self.__class__.__name__,obstime,num)
        return snr

class MaglimTactician(Tactician):
    """Simple tactician that converts a magnitude limit to an exposure
    time and applies that exposure time to every star.

    """
    _defaults = odict(Tactician._defaults.items() +
                      [('maglim',24.)]
    )

    def schedule(self, data, **kwargs):
        maglim = kwargs.get('maglim',self.maglim)
        snr_thresh = kwargs.get('snr_thresh',self.snr_thresh)
        obstime = self.instrument.maglim2exp(maglim,snr_thresh)
        snr = self.instrument.mag2snr(data['mag'],obstime)

        num = (np.nan_to_num(snr)>snr_thresh).sum()
        msg = "%s -- Maglim: %.2f, ObsTime: %.2f, NStars: %i"%(self.__class__.__name__,maglim,obstime,num)
        logging.debug(msg)

        return snr

class NStarsTactician(Tactician): 
    """
    Calculate exposure time to reach a certain number of stars.

    WARNING: May not deal with saturated stars as expected...
    """
    _defaults = odict(Tactician._defaults.items() +
                      [('nstars',25)]
    )

    def schedule(self, data, **kwargs):
        nstars     = kwargs.get('nstars',self.nstars)
        snr_thresh = kwargs.get('snr_thresh',self.snr_thresh)

        mags = data['mag']

        sort_idx = np.argsort(mags)
        sort_mag = mags[sort_idx]

        min_exptime = self.instrument.maglim2exp(sort_mag,snr_thresh)
        self.obstime = min_exptime[nstars-1]

        snr = self.instrument.mag2snr(sort_mag,self.obstime)
        nexp = 1

        num = (np.nan_to_num(snr)>snr_thresh).sum()
        msg = "%s -- NExp: %i, ExpTime: %.2f, NStars: %i"%(self.__class__.__name__,nexp,self.obstime,num)
        logging.debug(msg)
        return snr[np.argsort(sort_idx)]


class EqualTactician(Tactician): 
    """ Equally distribute time accross all stars
    1) Take total "star time" (inst.multiplex * exptime)
    2) Divide by number of stars and calculate S/N per star
    3) Figure out how many stars pass S/N > 5
    4) Divide by new number of stars to get time per star
    5) Iterate until convergence
    """
    def schedule(self, data, **kwargs):
        obstime    = kwargs.get('obstime',self.obstime)
        snr_thresh = kwargs.get('snr_thresh',self.snr_thresh)

        mags = data['mag']
        total_time = float(self.instrument.nstar * obstime)
        eff_time = obstime 
        delta_time = np.inf

        i = 0
        while delta_time > 30:
            # SNR given previous effective time
            snr = self.instrument.mag2snr(mags,eff_time)
            # New effective time based only on stars with SNR > thresh
            new_eff_time = total_time/(snr>snr_thresh).sum()
            # Change between previous and current effective time
            delta_time = np.abs(eff_time - new_eff_time)
            eff_time = new_eff_time

            num = (np.nan_to_num(snr)>snr_thresh).sum()
            i += 1
            msg = "%s -- Niter: %i, NExp: %i, ExpTime: %.2f, NStars: %i"%(self.__class__.__name__,i,total_time/eff_time,eff_time,num)
            logging.debug(msg)

        return snr

class EqualTimeTactician(Tactician):
    """Integrate according to pseudo-exposures

    1) Sort the stars in order of brightness
    2) Only take exposures that would add new stars above threshould

    This algorithm does not deal with overheads, which will be
    constant per exposure.

    """
    _defaults = odict(Tactician._defaults.items() +
                      [('max_nexp',np.inf)]
    )

    def schedule(self, data, **kwargs):
        obstime    = kwargs.get('obstime',self.obstime)
        snr_thresh = kwargs.get('snr_thresh',self.snr_thresh)
        max_nexp   = kwargs.get('max_nexp',self.max_nexp)
        
        if max_nexp < 0: max_nexp = np.inf

        mags = data['mag']
        
        total_time = float(obstime)
        multiplex = self.instrument.nstar

        sort_idx = np.argsort(mags)
        sort_mag = mags[sort_idx]
        nstars = len(sort_mag)

        # Minimum exposure time to reach SNR threshold for each object
        min_exptime = self.instrument.maglim2exp(sort_mag)
        snr = np.zeros_like(sort_mag)

        # Number of exposures and time per exposure
        nexps = np.arange(int(len(sort_mag)/multiplex)) + 1
        exptime = total_time/nexps
        next_nexps = nexps+1
        next_exptime = total_time/(next_nexps)

        # Decide how many exposures to take
        sel = (min_exptime[::multiplex] <= next_exptime) & (next_nexps <= max_nexp) 
        nexp = next_nexps[np.where(sel)[0].max()] if sel.sum() else 1
        idx = nexp * multiplex

        eff_exptime = total_time/nexp
        snr[:idx] = self.instrument.mag2snr(sort_mag[:idx],eff_exptime)

        msg = "%s -- NExp: %i, ExpTime: %.2f, NStars: %i"%(self.__class__.__name__,nexp,eff_exptime,idx)
        logging.debug(msg)
        
        return snr[np.argsort(sort_idx)]

class DynamicTimeTactician(Tactician): 
    """Maximize the number of stars above the SNR threshold. Overheads
    are added as a constant term for each exposure.

    1) Calculate the amount of time necessary to reach S/N = 5 for each star
    2) Group stars by brightness
    3) Expose each group long enough to get faintest star at S/N = 5
    4) Work from brightest group to faintest until out of time
    """
    def schedule(self, data, **kwargs):
        obstime    = kwargs.get('obstime',self.obstime)
        snr_thresh = kwargs.get('snr_thresh',self.snr_thresh)

        mags = data['mag']

        nstar = self.instrument.nstar
        overhead = self.instrument.overhead

        sort_idx = np.argsort(mags)
        sort_mag = mags[sort_idx]
        num = len(sort_mag)

        # Exposure time to reach S/N limit for each star
        min_exptime = self.instrument.maglim2exp(sort_mag)
        snr = np.zeros_like(sort_mag)

        used_time = 0
        for i in range(int(num/nstar)+1):
            # Out of time
            if used_time >= obstime: break

            imin = i*nstar
            imax = (i+1)*nstar if (i+1)*nstar < num else num-1

            eff_time = min_exptime[imax] + overhead
            
            # Last exposure...
            if obstime < (used_time+eff_time):
                eff_time = obstime - used_time

            snr[imin:imax] = self.instrument.mag2snr(sort_mag[imin:imax],eff_time)
            used_time += eff_time
            nexp = i+1

            ndetect = (np.nan_to_num(snr)>0).sum()
            msg = "%s -- NExp: %i, ExpTime: %.2f, NStars: %i"%(self.name,nexp,eff_time,ndetect)
            logging.debug(msg)

        return snr[np.argsort(sort_idx)]


if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()
