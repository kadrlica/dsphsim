#!/usr/bin/env python
"""
Encapsulate the properties of a dwarf galaxy.
"""

import copy
from collections import OrderedDict as odict
import numpy as np
import logging

from dsphsim.velocity import GaussianVelocity
from dsphsim.velocity import PhysicalVelocity
from dsphsim.velocity import velocityFactory

from ugali.analysis.model import Model, Parameter
from ugali.analysis.source import Source

class Dwarf(Source):
    """ Class containing the physical characteristics of the dwarf
    galaxy. This can be used to sample the `true` physical
    properties of the stars in a dwarf galaxy. """

    _defaults = odict(
        Source._defaults.items() + 
        [
            ('kinematics' , dict(name='GaussianVelocity')),
        ])
    
    def __init__(self,name=None, **kwargs):
        kw = dict()
        if kwargs.get('kinematics'):
            kw.update(name=kwargs.pop('kinematics'))
        self.set_model('kinematics',self.createKinematics(**kw))
        super(Dwarf,self).__init__(name,**kwargs)

    def simulate(self):
        stellar_mass = self.richness * self.isochrone.stellar_mass()
        mag_1, mag_2 = self.isochrone.simulate(stellar_mass,mass_steps=1e4)
        lon, lat     = self.kernel.simulate(len(mag_1))

        # Physical projected radius
        #if not hasattr(self,'vdist'): self._create_vdist()
        #if not hold: self._create_vdist()
        angsep       = self.kernel.angsep(lon,lat)

        # If the extension, ellipticity, distance, or kinematics
        # change then we need to regenerate the velocity distribution
        sync = self.get_sync('kernel') \
               or self.get_sync('isochrone') \
               or self.get_sync('kinematics')

        if sync: logging.debug("Syncing velocity distribution")
        # Doesn't need to be done for richness...
        # Move to velocity.py?
        if sync:
            # Physical plummer radius (kpc)
            rh = self.extension * np.sqrt(1-self.ellipticity)
            rpl = self.distance * np.tan(np.radians(rh)) 
            self.kinematics.rpl = rpl

        logging.debug('distance: %s'%self.distance)
        logging.debug('extension: %s'%self.extension)
        logging.debug('rpl: %s'%self.kinematics.rpl)

        vel = self.kinematics.sample_angsep(angsep,self.distance,sync=sync)

        self.reset_sync()
        return mag_1, mag_2, lon, lat, vel

    def parse(self, args):
        pass

    def set_kinematics(self,kinematics): 
        self.set_model('kinematics',kinematics)

        
    @property
    def kinematics(self):
        return self.models['kinematics']

    @classmethod
    def createKinematics(cls,**kwargs):
        for k,v in copy.deepcopy(cls._defaults['kinematics']).items():
            kwargs.setdefault(k,v)
        return velocityFactory(**kwargs)

    @property
    def distance(self):
        return self.isochrone.distance

    @property
    def absolute_magnitude(self):
        return self.isochrone.absolute_magnitude(self.richness)

    @property
    def stellar_luminosity(self):
        return self.richness * self.isochrone.stellarLuminosity()

    @property
    def stellar_mass(self):
        return self.richness * self.isochrone.stellarMass()

    @property
    def rhalf(self):
        rh = self.extension * np.sqrt(1-self.ellipticity)
        return self.distance * np.tan(np.radians(rh))


if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()
