#!/usr/bin/env python
"""
Module for deriving kinematic velocity distributions.

WARNING: This module works in physical space (kpc) *not* angular space.

Louie points us to this reference: 
https://arxiv.org/abs/1003.4268

"""

import sys
import copy
from collections import OrderedDict as odict
import logging

import numpy as np
import scipy.stats
from scipy.integrate import romberg, quad, simps, trapz
from scipy.interpolate import interp1d
from scipy.misc import derivative
import matplotlib.tri as mtri

import vegas

from ugali.analysis.model import Model,Parameter

# Constants and sampling parameters (left over from Fortran)
nx=200;nes=200;nts=100;nvs=100;nrs=25
# Newton's constant in (km^3 Msun^-1 s^-1)
Gn=6.67e-11*1.99e30*1.e-9 
# Vegas MCMC integration parameters
n_grid = 1000; n_final = 5000

def loginterp1d(x,y,**kwargs):
    """
    Interpolate a 1-D function applying a logarithmic transform to y-values.

    See scipy.interpolate.interp1d for more details.
    """
    err = np.seterr(divide='ignore')
    loginterp = interp1d(x,np.log(y),**kwargs)
    np.seterr(**err)
    return lambda x: np.exp(loginterp(x))

def triinterp2d(x,y,z,**kwargs):
    """
    Create a linear triangular interpolation using the mlab implementation.

    See `matplotlib.tri.LinearTriInterpolator`

    Returns a numpy masked array.
    """
    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    triang = mtri.Triangulation(x.flat, y.flat)
    return mtri.LinearTriInterpolator(triang, z.flat)

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
    scalar = np.isscalar(x)
    x = np.atleast_1d(np.clip(x, 1e-10, None))
    Phi = -np.log(1. + x)/x
    if scalar: return np.asscalar(Phi)
    else: return Phi

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
    scalar = np.isscalar(x)
    x = np.atleast_1d(np.where(x==0, 1e-30, x))
    dPhi = np.log(1.0 + x) / x**2 - 1.0 / (x * (1.0 + x) )
    if scalar:
        return np.asscalar(dPhi)
    else:
        return dPhi

def deriv_rhostar_deriv_Psi(x,y):
    """
    Derivative of the Plummer density with respect to the NFW potential.

    Parameters:
    x : Radial distance normalized to NFW scale radius (r/rs)
    y : Ratio between Plummer half-light radius and NFW scale radius

    Returns: 
    dPhi : Derivative of the stellar density with respect to
           the gravitational potential
    """
    return deriv_rhostar(x,y)/deriv_nfw_potential(x)

def rhostar(x,y):
    """
    Physical projected (2D) Plummer stellar density.
    
    Parameters:
    x : Radial distance normalized to NFW scale radius (r/rs)
    y : Ratio between Plummer half-light radius and NFW scale radius

    Returns: 
    rhostar : 2D projected stellar density
    """
    #y = 0.105d0/r0 
    rhostar = 1.0/(1.0+ x**2/y**2)**2.0 
    return rhostar

#class VelocityDistribution(object):
class VelocityDistribution(Model):
    """
    Base class for velocity distribution.
    """
    _params = odict([
        ('vmean', Parameter(0.0) ),  # km/s
        ('vdisp', Parameter(1.0) ),    # Velocity dispersion (km/s)
    ])


    def _sample(self, radius, sync=True):
        """
        Do the sampling (overloaded by subclass)
        """
        return self.vdisp * np.ones_like(radius)

    def sample(self, radius, sync=True):
        """
        Draw a random sample of velocities from the inverse CDF.

        Parameters:
        radius : 2D physical projected radius (kpc)
        
        Returns:
        velocity : Randomly sampled velocities for each star (km/s)
        """
        scalar = np.isscalar(radius)
        vel = self._sample(np.atleast_1d(radius),sync)
        # Don't forget the mean velocity
        vel += self.vmean
        if scalar: return np.asscalar(vel)
        return vel

    def sample_angsep(self, angsep, distance, sync=True):
        """
        Draw a random sample of velocities from the inverse CDF.

        Parameters:
        angsep   : Angular projected radius (deg)
        distance : Heliocentric distance to dwarf (kpc)

        Returns:
        velocity : Randomly sampled velocities for each star (km/s)
        """
        radius = distance * np.tan(np.radians(angsep))
        return self.sample(radius,sync)

class GaussianVelocity(VelocityDistribution):
    """
    Simple Gaussian velocity dispersion Norm(mu=0, sigma=vdisp).
    """
    _params = odict([
        ('vmean', Parameter(0.0) ),  # km/s
        ('vdisp', Parameter(3.3) ),  # Velocity dispersion (km/s)
    ])

    def _sample(self, radius, hold=False):
        return np.random.normal(0.0,self.vdisp,size=len(radius))

class PhysicalVelocity(VelocityDistribution):
    """
    Class to calculate and apply a dwarf galaxy velocity dispersion.
    """

    _params = odict([
        ('vmean', Parameter(0.0) ),   # Systemic velocity (km/s)
        ('vmax',  Parameter(10.0) ),  # Maximum circular velocity (km/s)
        ('rvmax', Parameter(0.4) ),   # Radius of maximum circular velocity (kpc)
        ('rpl',   Parameter(0.04) ),  # Projected Plummer radius (kpc)
    ])
    
    def _cache(self, name=None):
        velmin = getattr(self,'velmin',-1.5 * self.vmax)
        velmax = getattr(self,'velmax', 1.5 * self.vmax)
        #velmin = -1.5 * self.vmax 
        #velmax = 1.5 * self.vmax 
        self.velocities = np.linspace(velmin,velmax,nvs)

        xmin = getattr(self,'xmin',1e-4)
        xmax = getattr(self,'xmax',5)
        self.xvalues = np.logspace(np.log10(xmin),np.log10(xmax),nrs)

        self.build_interps()

    @classmethod
    def rs2rvmax(cls, rs): 
        return 2.163 * rs

    @classmethod
    def rvmax2rs(cls, rvmax): 
        return rvmax/2.163

    @classmethod
    def rhos2vmax(cls, rhos, rs): 
        G = 4.302e-6 # kpc Msun^-1 (km/s)^2
        return rs  * np.sqrt( rhos * (4*np.pi*G)/4.625 )

    @classmethod
    def vmax2rhos(cls, vmax, rvmax): 
        G = 4.302e-6 # kpc Msun^-1 (km/s)^2
        rs = cls.rvmax2rs(rvmax)
        return 4.625/(4*np.pi*G) * (vmax/rs)**2

    @property
    def Phis(self):
        # Central potential as in 1406.6079 in (km/s)^2 
        return self.vmax**2/0.465**2 

    @property
    def rs(self):
        """
        NFW scale radius (kpc) from Eqn 9 of:
        http://arxiv.org/abs/0805.4416
        """
        return self.rvmax/2.163

    @property
    def rhos(self):
        """
        NFW scale density (Msun/kpc^3) from Eqn 10 of:
        http://arxiv.org/abs/0805.4416
        """
        G = 4.302e-6 # kpc Msun^-1 (km/s)^2
        return 4.625/(4*np.pi*G) * (self.vmax/self.rs)**2

    def integrate_energy(self, energy, interp, method='simps'):
        """
        Eddington inversion formula. Corresponds to Eqn (5) of:
        https://arxiv.org/abs/1003.4268

        f(e) ~ \int^0_e d2rho/dpsi2 dpsi/sqrt(psi-e)
        
        Parameters:
        energy : Energies at which to evaluate f(e)
        interp : Interpolation of integrandd2rho/dpsi2
        method : Integration technique

        Returns:
        integral : array as a function of e
        """
        err = np.seterr(divide='ignore',invalid='ignore')
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
            x_array = emin + (0.99*energy-emin)*np.linspace(0,1,nstep)[:,np.newaxis]
            fa_array = fa(x_array,energy)
            if method == 'trapz':
                int_e = trapz(fa_array,x_array,axis=0)
            if method == 'simps':
                int_e = simps(fa_array,x_array,axis=0)
        else:
            msg = "Unrecognized method: %s"%method
            raise Exception(msg)

        np.seterr(**err)
        return np.asarray(int_e)

    def build_interps(self):
        # calculate the derivative to the stellar density with respect
        # to the potential xmin, xmax is in units of the dark matter
        # scale radius
        xmin = 1.e-6
        xmax = 1.e2
        #xstep = np.log(xmax/xmin)/float(nx-1)
        rx = np.exp(np.linspace(np.log(xmin),np.log(xmax),nx))
        rx = np.append(0,rx)

        # Calculate the approximate derivative of the NFW potential for
        # the Eddington formula
        Phi = nfw_potential(rx)
        Psi = -Phi

        self.interp_psi = loginterp1d(rx,Psi,bounds_error=False)

        drhostardPsi = -deriv_rhostar_deriv_Psi(rx,self.rpl/self.rs)

        # get the arrays in increasing numerical order 
        Psi_reverse = Psi[::-1]
        drhostardPsi_reverse = drhostardPsi[::-1]
        rx_reverse = rx[::-1]

        # Interpolate derivative as a function of potential 
        self.interp_e = loginterp1d(Psi_reverse,drhostardPsi_reverse,
                                    kind='linear',bounds_error=False)

        # Define the range of (binding) energies to evaluate at 
        emin = np.min(Psi_reverse)
        emax = np.max(Psi_reverse)
        #estep = np.log(emax/emin)/float(nes-1)
        #print emin, emax, estep

        # Prepend a zero, something Fortran was doing...
        #energy = emin*np.exp(np.arange(1,nes)*estep)
        energy = np.exp(np.linspace(np.log(emin),np.log(emax),nes))
        energy = np.append(0,energy)

        int_e = self.integrate_energy(energy,self.interp_e,method='simps')

        ## Prepend a zero, something Fortran was doing...
        #energy    = np.append([0],energy)
        #int_e     = np.append([0],int_e)

        self.interp_inte = interp1d(energy,int_e,kind='linear',bounds_error=False)
        #self.interp_inte = UnivariateSpline(energy,int_e,k=1,s=0)

        def fb(e, emax=emax):
            return np.where(e >= emax, 0, self.interp_inte(e))

        temp_int = derivative(fb,energy[:-2],1e-4)
        temp_int[0] = 0
         
        logfe_temp = np.array(temp_int)
        kcut       = ~np.isnan(logfe_temp)
        energy_cut = self.Phis*energy[kcut]
        logfe_cut  = np.array(logfe_temp)[kcut]
         
        # Interpolation for log(f) (?) as a function of energy
        self.energy = energy_cut
        self.interp_logfe = interp1d(energy_cut,logfe_cut,
                                     kind='linear',bounds_error=False)

        return emin,emax

    def velocity_distribution(self, radius, velocities=None):
        """
        Calculate the velocity distribution.
        
        Parameters:
        radius     : Projected radius (in units of the scale radius?)
        velocities : Array of velocity to evalute f(v)
        
        Returns:
        fv         : The velocity distribution pdf
        """

        if not np.isscalar(radius):
            msg = "'radius' must be scalar (at least for now)"
            raise Exception(msg)
        
        # Projected scale(?) radius
        Rp_in = radius

        # Calculate the max radius for this velocity. 
        # Assume something large small error incurred but come back and refine this 
        maxr = 10. # in units of the scale radius?

        # If star is outside the maximum radius
        if Rp_in > maxr: 
            # warning....
            return np.zeros_like(self.velocities)

        # Define the projected LOS velocity range
        if velocities is None: 
            velocities = self.velocities
      
        # Define the min of the variable t to evaluate at
        tmin = 1.e-4 
        tmax = np.sqrt(maxr-Rp_in)
        #tstep = np.log(tmax/tmin)/float(nts-1)
        
        # Energy range
        emin = np.min(self.energy)
        emax = np.max(self.energy)

        # Bounds of integration
        region = [[tmin,tmax],  # t1
                  [emin,emax],  # velocity
                  [0, 2*np.pi], # eta
        ]
        self.region = region

        @vegas.batchintegrand
        def dfunc(y):
            """
            Probability density function?

            Parameters:
            y : (array-like) t1,vt,eta
            
            Returns:
            fe : density of f(e)
            """
            y = np.atleast_2d(y)
            t1,vt,eta = y.T

            rspline = t1**2 + Rp_in
            Psi_spline = self.interp_psi(rspline)

            sel = (emin <= vt) & (vt <= (self.Phis*Psi_spline - 0.5*v_in**2))
         
            val = self.interp_logfe(vt) * (t1**2+Rp_in)/np.sqrt(2*Rp_in+t1**2)
            return np.where(sel,val,0) 

        # For debugging
        self.dfunc = dfunc
        # self.dfunc = lambda self,y: return dfunc(y)

        # Setup the integrator
        integrator = vegas.Integrator(region)

        fv_out = []
        for v_in in velocities:
            # burn-in
            #integrator(dfunc,nitn=5,neval=n_grid)
            integrator(dfunc,nitn=5,neval=n_grid)
            # evaluate
            result = integrator(dfunc,nitn=5,neval=n_final)
            fv_out.append(result.val)

        return np.asarray(fv_out)
        
    def interp_vdist(self, xproj, nsteps=nrs):
        """
        Interpolate the velocity distribution.

        Parameters:
        xproj : Projected radius in units of the scale radius
        nsteps: Number of radii for interpolated velocity distribution

        Returns:
        None : (sets interp_pdf,interp_cdf,interp_icdf)
        """
        xproj = np.atleast_1d(xproj)

        #epsilon = 1e-4
        ##xmin =max(np.min(xproj)-epsilon,0)
        ##xmax = np.max(xproj)+epsilon
        #xmin = epsilon
        #xmax = max(np.max(xproj)+epsilon,5)
        # 
        ##xsteps = np.linspace(xmin,xmax,nsteps)
        #xstep = np.logspace(np.log10(xmin),np.log10(xmax),nsteps)

        xsteps = self.xvalues
        vsteps = self.velocities

        y,x = np.meshgrid(vsteps,xsteps)

        z = []
        for i,_x in enumerate(xsteps):
            #print >> sys.stderr, '(%i) x = %.3f'%(i,_x)
            logging.debug('(%i) x = %.3f'%(i,_x))
            fv = self.velocity_distribution(_x,vsteps)
            z.append(fv)

        z = np.array(z)

        # The nan_to_num is for 0/0 division (not too safe)
        u = np.nan_to_num(z/np.sum(z,axis=1)[:,np.newaxis])
        w = np.cumsum(z,axis=1)
        w = np.nan_to_num(w/np.max(w,axis=1)[:,np.newaxis])
        # Clip tiny values to avoid "Triangulation is invalid" error
        w = np.where(w < 1e-13, 0, w)
        
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        self.pdf  = z  # PDF
        self.npdf = u  # Normalized PDF
        self.cdf  = w  # Normalized CDF

        # Velocity and velocity function
        self.v  = y
        self.fv = z  

        # Triangular mesh interpolation from matplotlib
        self.interp_pdf  = triinterp2d(x,y,z)
        self.interp_cdf  = triinterp2d(x,y,w)
        self.interp_icdf = triinterp2d(x,w,y)

    def _sample(self, radius, sync=True):
        """
        Sample from the inverse CDF.
        """
        xproj = np.atleast_1d(radius)/self.rs
        if sync: 
            self.interp_vdist(xproj)
        i = np.random.uniform(size=len(xproj))
        vel = self.interp_icdf(xproj,i).filled(np.nan)
        return vel

    def radial_vdisp(self):
        if not hasattr(self,'interp_pdf'):
            self.interp_vdist()

        xproj = self.x
        vel = self.y

        pdf = self.interp_pdf(xproj,vel)

        sigma_r2 = simps(vel**2 * pdf, vel, axis=1)/simps(pdf, vel, axis=1)
        return np.sqrt(sigma_r2)

    def avg_vdisp(self):
        if not hasattr(self,'interp_pdf'):
            self.interp_vdist()

        xproj = self.x[:,0]
        sigma_r = self.radial_vdisp()
        I = rhostar(xproj,y=self.rpl/self.rs)
        sigma2 = simps(I*sigma_r**2*xproj, xproj)/simps(I*xproj, xproj)

        return np.sqrt(sigma2)

Gaussian = GaussianVelocity
Physical = PhysicalVelocity

# ADW: It would be good to replace this with ugali.utils.factory;
# however, this needs to accept unused kwargs.
def factory(name, **kwargs):
    """
    """
    import inspect

    cls = name
    module = __name__
    fn = lambda member: inspect.isclass(member) and member.__module__==module
    classes = odict(inspect.getmembers(sys.modules[module], fn))
    members = odict([(k.lower(),v) for k,v in classes.items()])
    
    lower = cls.lower()
    if lower not in members.keys():
        msg = "Unrecognized class: %s.%s"%(module,cls)
        raise KeyError(msg)

    kw = {k:v for k,v in kwargs.items() if k in members[lower]._params.keys()}
    return members[lower](**kw)

velocityFactory = factory
    
if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

