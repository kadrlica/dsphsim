#!/usr/bin/env python

from ugali.analysis.source import Source
from ugali.analysis.isochrone import isochroneFactory
from ugali.analysis.kernel import kernelFactory

class Dwarf(Source):
    """ Class containing the physical characteristics of the dwarf
    galaxy. This can be used to sample the `true` physical
    properties of the stars in a dwarf galaxy. """
    def __init__(self,vmean,vdisp,name=None,**kwargs):
        super(Dwarf,self).__init__(name,**kwargs)
        self.vmean = vmean
        self.vdisp = vdisp

    def simulate(self):
        stellar_mass = self.richness * self.stellar_mass()
        mag_1, mag_2 = self.isochrone.simulate(stellar_mass,mass_steps=1e4)
        lon, lat     = self.kernel.simulate(len(mag_1))
        return mag_1, mag_2, lon, lat

    def parse(self, args):
        pass

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()
