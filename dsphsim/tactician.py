#!/usr/bin/env python
"""
Choose how to target a field.
"""

import numpy as np

class Targeter(object):
    def __init__(self,*args,**kwargs):
        pass

class Tactician(object):
    def __init__(self, instrument, exptime=1000.):
        self.instrument = instrument
        self.exptime = exptime
        
    def schedule(self, data, obstime=None):
        if obstime is None: obstime = self.obstime
        return self.instrument.mag2snr(data['mag'],obstime)


class EqualTactician(Tactician): 
    """ Equally distribute time accross all stars
    1) Take total "star time" (inst.multiplex * exptime)
    2) Divide by number of stars and calculate S/N per star
    3) Figure out how many stars pass S/N > 5
    4) Divide by new number of stars to get time per star
    5) Iterate until convergence
    """
    def schedule(self, data, obstime=None, snr_thresh=5):
        if obstime is None: obstime = self.obstime
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

            num = (snr>snr_thresh).sum()
            i += 1
        print "Niter: %i, NExp: %i, ExpTime: %.2f, NStars: %i"%(i,total_time/eff_time,eff_time,num)

        return snr

class EqualTimeTactician(Tactician):
    """Integrate according to pseudo-exposures

    1) Sort the stars in order of brightness
    2) Only take exposures that would add new stars above threshould

    This algorithm does not deal with overheads, which will be
    constant per exposure.

    """
    def schedule(self, data, obstime=None, snr_thresh=5, max_nexp=None):
        if obstime is None: obstime = self.obstime
        if max_nexp is None or max_exps < 0: max_nexp = np.inf

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

        print "SmartTactician -- NExp: %i, ExpTime: %.2f, NStars: %i"%(nexp,eff_exptime,idx)
        
        return snr[np.argsort(sort_idx)]

class DynamicTimeTactician(Tactician): 
    """Maximize the number of stars with S/N > 5

    1) Calculate the amount of time necessary to reach S/N = 5 for each star
    2) Group stars by brightness
    3) Expose each group long enough to get faintest star at S/N = 5
    4) Work from brightest group to faintest until out of time

    This algorithm does not deal with overheads, which will be constant per exposure.
    """
    def schedule(self, data, obstime=None, snr_thresh=5):
        if obstime is None: obstime = self.obstime
        mags = data['mag']

        nstar = self.instrument.nstar

        sort_idx = np.argsort(mags)
        sort_mag = mags[sort_idx]
        num = len(sort_mag)

        min_exptime = self.instrument.maglim2exp(sort_mag)
        snr = np.zeros_like(sort_mag)

        used_time = 0
        for i in range(int(num/nstar)+1):
            if used_time >= obstime: break

            imin = i*nstar
            imax = (i+1)*nstar if (i+1)*nstar < num else num-1

            eff_time = min_exptime[imax]

            # Last exposure...
            if obstime < (used_time+eff_time):
                eff_time = obstime - used_time

            snr[imin:imax] = self.instrument.mag2snr(sort_mag[imin:imax],eff_time)
            used_time += eff_time
            nexp = i+1

        print "MaxStars Tactician -- NExp: %i, ExpTime: %.2f, NStar: %i"%(nexp,eff_time,(snr>0).sum())
        return snr[np.argsort(sort_idx)]


class NumberStarsTactician(Tactician): 
    """
    """
    def schedule(self, data, nstars=25, snr_thresh=5):
        mags = data['mag']

        sort_idx = np.argsort(mags)
        sort_mag = mags[sort_idx]

        min_exptime = self.instrument.maglim2exp(sort_mag)
        self.obstime = min_exptime[nstars]
        snr = self.instrument.mag2snr(sort_mag,self.obstime)
        nexp = 1

        print "NumberStars Tactician -- NExp: %i, ExpTime: %.2f, NStar: %i"%(nexp,self.obstime,(snr>snr_thresh).sum())
        return snr[np.argsort(sort_idx)]
    

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()
