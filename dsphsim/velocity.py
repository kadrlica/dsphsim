#!/usr/bin/env python
"""
Module for deriving velocity distribution
"""
__author__ = "Alex Drlica-Wagner"

import copy
from collections import OrderedDict as odict

import numpy as np
from scipy.integrate import romberg, quad, simps, romb, trapz, cumtrapz
from scipy.interpolate import interp1d
from scipy.misc import derivative

import vegas

def loginterp1d(x,y,**kwargs):
    """
    Interpolate a 1-D function applying a logarithmic transform to y.

    See scipy.interpolate.interp1d for more details.
    """
    loginterp = interp1d(x,np.log(y),**kwargs)
    return lambda x: np.exp(loginterp(x))

def plot_values_residuals(x1,y1,x2,y2,**kwargs):
    fig = plt.figure()
    return draw_values_residuals(x1,y1,x2,y2,**kwargs)

def draw_values_residuals(x1,y1,x2,y2,**kwargs):
    ax = plt.gca()
    ax.set_yscale('log',nonposy='clip')

    divider = make_axes_locatable(ax)
    rax = divider.append_axes("bottom", 1.2, pad=0.05, sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    #ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    rax.yaxis.set_major_locator(MaxNLocator(4,prune='both'))
    
    ax.plot(x1,y1,'o-k')
    ax.plot(x2,y2,'o-r')

    #loginterp = interp1d(x1,np.log(y1),bounds_error=False)
    #interp = lambda x: np.exp(loginterp(x))
    interp = loginterp1d(x1,y1,bounds_error=False)

    rax.plot(x1,(y1-interp(x1))/y1,'o-k')
    rax.plot(x2,(y2-interp(x2))/y2,'o-r')

    return ax, rax

def nfw_potential(x):
    """ 
    Normalized gravitational potential for NFW profile. 

    Parameters:
    x : Radial distance normalized to NFW scale radius (r/rs)

    Returns:
    phi : Normalized gravitational potential
    """
    return -np.log(1. + x)/x

def deriv_rhostar(x,y):
    """ 
    Derivative of the stellar density for Plummer profile.

    Parameters:
    x : Radial distance normalized to NFW scale radius (r/rs)
        NOTE: this is *not* normalized to the Plummer radius
    y : Ratio between plummer and NFW scale radius

    Returns:
    drho : Derivative of the stellar density
    """
    #y = rpl/r0 
    return (-2.5/(1.0+ x**2/y**2)**3.5) * 2.0* x/y**2

def deriv_nfw_potential(x):
    """
    Derivative of the NFW potential with respect to x.

    Parameters:
    x : Radial distance normalized to scale radius (r/rs)

    Returns:
    dPhi : Derivative (with respect to x) of gravitational potential
    """
    return np.log(1.0 + x) / x**2 - 1.0 / (x * (1.0 + x) )

def deriv_rhostar_deriv_Psi(x,y):
    """
    Derivative of the Plummer density with respect to the NFW potential.

    Parameters:
    x : Radial distance normalized to NFW scale radius (r/rs)
    y : Ratio between plummer and NFW scale radius

    Returns: 
    dPhi : Derivative of the stellar density with respect to
           the gravitational potential

    """
    return deriv_rhostar(x,y)/deriv_nfw_potential(x)

nx=200;nes=200;nts=100;nvs=100
Gn=6.67e-11*1.99e30*1.e-9 # Newton's constant in units of ...???

class VelocityDistribution(object):
    """
    Class to calculate and apply a dwarf galaxy velocity dispersion.
    """

    _defaults = odict([
        ('rpl', 0.04), # Plummer radius (kpc)
        ('r0',0.2),    # NFW scale radius (kpc)
        ('vmax',10.0), # Maximum circular velocity (km/s)
    ])
    
    def __init__(self,args,**kwargs):
        self._setup(**kwargs)

    def _setup(self,**kwargs):
        for k,v in kwargs:
            if k not in self._defaults:
                msg = "Keyword argument not found in defaults"
                raise KeyError(msg)

        for k,v in self._defaults.items():
            kwargs.setdefault(k,v)
        
        self.__dict__.update(kwargs)

        # Central potential as in 1406.6079 in (km/s)^2 
        self.Phis = self.vmax**2/0.465**2 


    def integrate_energy(energy, interp, method='simps'):
        err = np.seterr(divide='ignore')
        method = method.lower()
        emin = np.min(energy)

        def fa(x,e):
            return np.nan_to_num(interp(x)/np.sqrt(e-x))
            
        if method in ['quad','romberg']:
            # Integrate from function object
            int_e = []
            for e in energy:
                a,b = emin, 0.99*e
                if method == 'quad':
                    int_eappend(quad(fa,a,b,args=(e),epsrel=1e-8)[0])
                elif method == 'romberg':
                    int_e.append(romberg(fa,a,b,args=(e,),divmax=15,tol=1e-5))
            int_e = int_e
        elif method in ['trapz','simps']:
            # Integrate from array
            nstep = 1000
            x_arr = emin + (0.99*energy-emin)*np.linspace(0,1,nstep)[:,np.newaxis]
            fa_array = fa(x_arr,energy)
            if method == 'trapz':
                int_e = trapz(fa_array,x_array,axis=0)
            if method == 'simps':
                int_e = simps(fa_array,x_array,axis=0)
        else:
            msg = "Unrecognized method: %s"%method
            raise Exception(msg)

        np.seterr(**err)
        return np.asarray(int_e)

    def interpolate_logfe(self):

        # calculate the derivative to the stellar density with respect
        # to the potential xmin, xmax is in units of the dark matter
        # scale radius
        xmin = 1.e-6
        xmax = 1.e2
        #xstep = np.log(xmax/xmin)/float(nx-1)
        rx = np.exp(np.linspace(np.log(xmin),np.log(xmax),nx))

        # Calculate the approximate derivative of the NFW potential for
        # the Eddington formula
        Phi = nfw_potential(rx)
        Psi = -Phi

        self.interp_psi = loginterp1d(rx,Psi,bounds_error=False)

        drhostardPsi = -deriv_rhostar_deriv_Psi(rx,self.rpl/self.r0)

        # get the arrays in increasing numerical order 
        Psi_reverse = Psi[::-1]
        drhostardPsi_reverse = drhostardPsi[::-1]
        rx_reverse = rx[::-1]

        # Define the range of energies to evaluate at 
        emin = np.min(Psi_reverse)
        emax = np.max(Psi_reverse)
        estep = np.log(emax/emin)/float(nes-1)
        #print emin, emax, estep

        energy = emin*np.exp(np.arange(1,nes)*estep)
        self.interp_e = loginterp1d(Psi_reverse,drhostardPsi_reverse,
                                    kind='linear',bounds_error=False)

        int_e = self.integrate_energy(energy,self.interp_e,method='simps')
        log_int_e = np.log(int_e)

        # Prepend a zero, something Fortran was doing...
        energy    = np.append([0],energy)
        int_e     = np.append([0],int_e)

        self.interp_inte = interp1d(energy,int_e,kind='linear',bounds_error=False)

        def fb(x, xmax = np.max(Psi_reverse)):
            return np.where(x >= xmax, 0, self.interp_inte(x))
         
        temp_int = derivative(fb,energy[:-2],1e-4)
        temp_int[0] = 0
         
        logfe_temp = np.array(temp_int)
        kcut = ~np.isnan(logfe_temp)
        energy_cut = self.Phis*energy[kcut]
        logfe_cut = np.array(logfe_temp)[kcut]
         
        # Interpolation for log(f) as a function of energy       
        self.interp_logfe = interp1d(energy_cut,logfe_cut,
                                     kind='linear',bounds_error=False)

        return interp_logfe
        
if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()
