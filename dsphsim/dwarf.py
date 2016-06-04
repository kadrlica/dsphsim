#!/usr/bin/env python
import copy
from collections import OrderedDict as odict
import numpy as np

from dsphsim.velocity import GaussianVelocityDistribution
from dsphsim.velocity import PhysicalVelocityDistribution

from ugali.analysis.model import Model, Parameter
from ugali.analysis.source import Source
from ugali.analysis.isochrone import isochroneFactory
from ugali.analysis.kernel import kernelFactory

# This is the place to store all the DM profile information
class Physical(Model):
    _params = odict([
        ('vmean', Parameter(0.0) ),  # km/s
        ('vdisp', Parameter(3.3) ),  # km/s
        ('vmax' , Parameter(10.0) ), # km/s
        ('rs',    Parameter(0.3) ),  # NFW scale radius (kpc)
    ])


class Dwarf(Source):
    """ Class containing the physical characteristics of the dwarf
    galaxy. This can be used to sample the `true` physical
    properties of the stars in a dwarf galaxy. """
    _defaults = odict(
        Source._defaults.items() + 
        [
            ('physical' , dict()),
        ])
    
    def __init__(self,name=None, **kwargs):
        self.set_model('physical',self.createPhysical())
        super(Dwarf,self).__init__(name,**kwargs)

    def _create_vdist(self):
        # Physical plummer radius (kpc)
        if not np.any(np.isnan([self.vmax,self.rs])):
            rh = self.extension * np.sqrt(1-self.ellipticity)
            rpl = self.distance * np.tan(np.radians(rh)) 
            print 'distance:',self.distance
            print 'extension:',self.extension
            print 'rpl:',rpl
            kwargs = dict(rpl=rpl,rs=self.rs,vmax=self.vmax)
            self.vdist = PhysicalVelocityDistribution(**kwargs)
        else:
            kwargs = dict(vdisp=self.vdisp)
            self.vdist = GaussianVelocityDistribution(**kwargs)

    def simulate(self,hold=False):
        stellar_mass = self.richness * self.stellar_mass()
        mag_1, mag_2 = self.isochrone.simulate(stellar_mass,mass_steps=1e4)
        lon, lat     = self.kernel.simulate(len(mag_1))

        # Physical projected radius
        #if not hasattr(self,'vdist'): self._create_vdist()
        if not hold: self._create_vdist()
        angsep       = self.kernel.angsep(lon,lat)
        velocity     = self.vdist.sample_angsep(angsep,self.distance)

        # Don't forget to add the systemic velocity
        #velocity    += self.vmean

        return mag_1, mag_2, lon, lat, velocity

    def parse(self, args):
        pass

    @property
    def distance(self):
        return self.isochrone.distance

    @classmethod
    def createPhysical(cls,**kwargs):
        for k,v in copy.deepcopy(cls._defaults['physical']).items():
            kwargs.setdefault(k,v)
        return Physical(**kwargs)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()
