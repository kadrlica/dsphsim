#!/usr/bin/env python
"""
Calculate the velocity dispersion from output data.
"""

import os
import numpy as np
import pylab as plt
import emcee
import corner
import scipy.stats

from ugali.analysis.mcmc import Samples

def lnprior(theta, vel):
    """
    Log-prior to set the bounds of the parameter space:
      sigma > 0 
      vmin < mu < vmax
    """
    sigma, mu = theta
    sigma2 = sigma**2

    if  sigma2 < 0:
        return -np.inf
    if not vel.min() < mu < vel.max():
        return -np.inf

    return 0

def lnlike(theta, vel, vel_err):
    """
    Log-likelihood function from Walker et al. (2007) Eq. :
    http://arxiv.org/abs/astro-ph/0511465
    """
    sigma, mu = theta
    sigma2 = sigma**2

    # break long equation into three parts
    a = -0.5 * np.sum(np.log(vel_err**2 + sigma2))
    b = -0.5 * np.sum((vel - mu)**2/(vel_err**2 + sigma2))
    # ADW: 'c' is a constant and can be discarded
    c = -1. * (vel.size)/2. * np.log(2*np.pi)

    return a + b + c

def lnprob(theta, vel, vel_err):
    lp = lnprior(theta, vel)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, vel, vel_err)

def mcmc(vel, vel_err, **kwargs):
    ndim = 2  # number of parameters in the model

    nwalkers = kwargs.get('nwalkers',20) # number of MCMC walkers
    nburn    = kwargs.get('nburn',50)    # "burn-in" period to let chains stabilize
    nsteps   = kwargs.get('nsteps',5000) # number of MCMC steps to take

    if not np.all(np.isfinite([vel,vel_err])):
        print "WARNING: Non-finite value found in data"
        sel = np.isfinite(vel) & np.isfinite(vel_err)
        vel,vel_err = vel[sel], vel_err[sel]

    #np.random.seed()
    m = np.random.normal(np.mean(vel), scale=1, size=(nwalkers))
    s = np.random.normal(np.std(vel), scale=1, size=(nwalkers))
    starting_guesses = np.vstack([s,m]).T

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[vel,vel_err], threads=1)
    sampler.run_mcmc(starting_guesses, nsteps)

    samples = sampler.chain.reshape(-1,ndim,order='F')
    samples = Samples(samples.T,names=['sigma','mu'])
    return samples

def plot(samples,nburn=50,clip=5):
    chain = samples.get(burn=nburn,clip=clip)
    figure = corner.corner(chain,labels=["$\sigma$ (km/s)", "$\mu$ (km/s)"])
    return figure

def parser():
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infile')
    parser.add_argument('-p','--plot',action='store_true')
    parser.add_argument('--nwalkers',type=int,default=20)
    parser.add_argument('--nburn',type=int,default=50)
    parser.add_argument('--nsteps',type=int,default=5000)
    parser.add_argument('--vel',default='VMEAS')
    parser.add_argument('--velerr',default='VERR')
    return parser

if __name__ == "__main__":
    p = parser()
    args = p.parse_args()

    data = np.genfromtxt(args.infile,names=True)

    kwargs = dict(nwalkers=args.nwalkers,nburn=args.nburn,nsteps=args.nsteps)
    vel = data[args.vel]
    if args.velerr is None or args.velerr.lower() == 'none':
        velerr = np.zeros_like(vel)
    else:
        velerr = data[args.velerr]

    samples = mcmc(vel,velerr,**kwargs)

    if args.plot:
        plt.ion()
        fig = plot(samples,nburn=args.nburn)
        outfile = os.path.splitext(args.infile)[0]+'.png'
        plt.savefig(outfile,bbox_inches='tight')
        
    mean,std = scipy.stats.norm.fit(vel)
    print '%-05s : %.2f'%('mean',mean)
    print '%-05s : %.2f'%('std',std)

    for name in ['mu','sigma']:
        peak,[low,high] = samples.peak_interval(name)
        print "%-05s : %.2f [%.2f,%.2f]"%(name,peak,low,high)
