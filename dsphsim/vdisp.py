#!/usr/bin/env python
"""
Calculate the velocity dispersion from output data.
"""
import os
import numpy as np
import pylab as plt
import emcee
import corner

from ugali.analysis.mcmc import Samples

def log_prior(theta, vel):
    sigma, mu = theta
    sigma2 = sigma**2

    if  sigma2 < 0:
        return -np.inf
    if not vel.min() < mu < vel.max():
        return -np.inf

    return 0 #1

def log_likelihood(theta, vel, vel_err):
    sigma, mu = theta
    sigma2 = sigma**2

    # break long equation into three parts
    a = -0.5 * np.sum(np.log(vel_err**2 + sigma2))
    b = -0.5 * np.sum((vel - mu)**2/(vel_err**2 + sigma2))
    # ADW: 'c' is a constant and can be discarded
    c = -1. * (vel.size)/2. * np.log(2*np.pi)

    return a + b + c

def log_posterior(theta, vel, vel_err):
    lp = log_prior(theta, vel)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, vel, vel_err)

def mcmc(vel, vel_err, **kwargs):
    ndim = 2  # number of parameters in the model

    nwalkers = kwargs.get('nwalkers',20) # number of MCMC walkers
    nburn    = kwargs.get('nburn',50)    # "burn-in" period to let chains stabilize
    nsteps   = kwargs.get('nsteps',5000) # number of MCMC steps to take

    #np.random.seed()
    m = np.random.normal(np.mean(vel), scale=1, size=(nwalkers))
    s = np.random.normal(np.std(vel), scale=1, size=(nwalkers))
    starting_guesses = np.vstack([s,m]).T

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[vel,vel_err], threads=1)
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
    parser.add_argument('--vmeas',default='VMEAS')
    parser.add_argument('--vmeaserr',default='VMEASERR')
    return parser

if __name__ == "__main__":
    p = parser()
    args = p.parse_args()

    data = np.genfromtxt(args.infile,names=True)

    kwargs = dict(nwalkers=args.nwalkers,nburn=args.nburn,nsteps=args.nsteps)
    samples = mcmc(data[args.vmeas],data[args.vmeaserr],**kwargs)

    if args.plot:
        plt.ion()
        fig = plot(samples,nburn=args.nburn)
        outfile = os.path.splitext(args.infile)[0]+'.png'
        plt.savefig(outfile,bbox_inches='tight')
        
    for name in samples.dtype.names:
        peak,[low,high] =samples.peak_interval(name)
        print "%s : %.2f [%.2f,%.2f]"%(name,peak,low,high)
