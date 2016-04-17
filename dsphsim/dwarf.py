#!/usr/bin/env python
import copy
from collections import OrderedDict as odict

from ugali.analysis.model import Model, Parameter
from ugali.analysis.source import Source
from ugali.analysis.isochrone import isochroneFactory
from ugali.analysis.kernel import kernelFactory

# This is the place to store all the DM profile information
class Physical(Model):
    _params = odict([
            ('vmean', Parameter(60.0) ),
            ('vdisp', Parameter(3.3) ),
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

    def simulate(self):
        stellar_mass = self.richness * self.stellar_mass()
        mag_1, mag_2 = self.isochrone.simulate(stellar_mass,mass_steps=1e4)
        lon, lat     = self.kernel.simulate(len(mag_1))
        return mag_1, mag_2, lon, lat

    def parse(self, args):
        pass

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
