#!/usr/bin/env python
"""
Calculate the mean and dispersion of a data set with errors using MCMC sampling of a Gaussian likelihood function.
"""
import os
import warnings
from collections import OrderedDict as odict

import numpy as np
import emcee
import scipy.stats

try:
    from dsphsim import __version__,__author__,__email__
except ImportError:
    __version__ = '0.1.0'
    __author__ = 'Alex Drlica-Wagner'
    __email__ = 'kadrlica@fnal.gov'

PARAMS = odict([
    ('mu',r'$\mu$ (km/s)'),
    ('sigma',r'$\sigma$ (km/s)'),
])

###############################################
# These functions come from ugali.utils.stats #
###############################################

def interval(best,lo=np.nan,hi=np.nan):
    """
    Pythonized interval for easy output to yaml
    """
    return [float(best),[float(lo),float(hi)]]

def kde(data, samples=1000):
    """
    Identify peak using Gaussian kernel density estimator.
    """
    # Clip severe outliers to concentrate more KDE samples in the
    # parameter range of interest
    mad = np.median(np.fabs(np.median(data) - data))
    cut = (data > np.median(data) - 5. * mad) & (data < np.median(data) + 5. * mad)
    x = data[cut]
    kde = scipy.stats.gaussian_kde(x)
    # No penalty for using a finer sampling for KDE evaluation except
    # computation time
    values = np.linspace(np.min(x), np.max(x), samples) 
    kde_values = kde.evaluate(values)
    peak = values[np.argmax(kde_values)]
    return values[np.argmax(kde_values)], kde.evaluate(peak)

def kde_peak(data, samples=1000):
    """
    Identify peak using Gaussian kernel density estimator.
    """
    return kde(data,samples)[0]

def peak_interval(data, alpha=0.32, samples=1000):
    """
    Identify interval using Gaussian kernel density estimator.
    """
    peak = kde_peak(data,samples)
    x = np.sort(data.flat); n = len(x)
    # The number of entries in the interval
    window = int(np.rint((1.0-alpha)*n))
    # The start, stop, and width of all possible intervals
    starts = x[:n-window]; ends = x[window:]
    widths = ends - starts
    # Just the intervals containing the peak
    select = (peak >= starts) & (peak <= ends)
    widths = widths[select]
    if len(widths) == 0:
        raise ValueError('Too few elements for interval calculation')
    min_idx = np.argmin(widths)
    lo = x[min_idx]
    hi = x[min_idx+window]
    return interval(peak,lo,hi)

###############################################

def lnprior(theta, vel):
    """
    Log-prior to set the bounds of the parameter space:
      sigma > 0 
      vmin < mu < vmax
    """
    #sigma, mu = theta
    mu, sigma = theta

    if  sigma < 0:
        return -np.inf
    if not vel.min() < mu < vel.max():
        return -np.inf

    return 0

def lnlike(theta, vel, vel_err):
    """
    Log-likelihood function from Walker et al. (2007) Eq. :
    http://arxiv.org/abs/astro-ph/0511465
    """
    #sigma, mu = theta
    mu, sigma = theta
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
    nperr = np.seterr(invalid='ignore')

    ndim = len(PARAMS)  # number of parameters in the model

    nwalkers = kwargs.get('nwalkers',20) # number of MCMC walkers
    nburn    = kwargs.get('nburn',50)    # "burn-in" period to let chains stabilize
    nsteps   = kwargs.get('nsteps',5000) # number of MCMC steps to take
    nthreads = kwargs.get('nthreads',1)  # number of threads

    if not np.all(np.isfinite([vel,vel_err])):
        print "WARNING: Non-finite value found in data"
        sel = np.isfinite(vel) & np.isfinite(vel_err)
        vel,vel_err = vel[sel], vel_err[sel]

    mu = np.random.normal(np.mean(vel), scale=1, size=(nwalkers))
    sigma = np.random.normal(np.std(vel), scale=1, size=(nwalkers))
    starting_guesses = np.vstack([mu,sigma]).T

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                    args=[vel,vel_err], threads=nthreads)
    sampler.run_mcmc(starting_guesses, nsteps)
    samples = sampler.chain.reshape(-1,ndim,order='F')

    names = PARAMS.keys()
    try:
        from ugali.analysis.mcmc import Samples
        samples = Samples(samples.T,names=names)
    except ImportError:
        # ugali is not installed; use recarray
        samples = np.rec.fromrecords(samples,names=names)

    samples = burn(samples,nburn*nwalkers)
    samples = cull(samples)

    np.seterr(**nperr)
    return samples

def burn(samples,nburn):
    """
    Burn the first `nburn` steps for each walker.
    """
    sel = np.zeros(len(samples),dtype=bool)
    sel[slice(nburn,None)] = 1
    return samples[sel]

def cull(samples):
    """
    Remove samples with unphysical dispersion (sigma < 0).
    """
    sel = (samples['sigma'] >= 0)
    return samples[sel]

def clip(samples,nsigma=4):
    """
    Sigma clip outliers from both parameters.
    """
    sel = np.ones(len(samples),dtype=bool)
    for n in samples.dtype.names:
        clip,cmin,cmax = scipy.stats.sigmaclip(samples[n],nsigma,nsigma)
        sel &= ((samples[n]>=cmin)&(samples[n]<=cmax))
    return samples[sel]

def plot(samples,intervals=None,sigma_clip=None):
    """ Create the corner plot (with some tweaking). """
    import corner

    names = samples.dtype.names

    if sigma_clip:
        samples = clip(samples,sigma_clip).view((float,len(names)))
    else:
        samples = samples.view((float,len(names)))

    figure = corner.corner(samples,labels=PARAMS.values())
    axes = figure.get_axes()    

    if intervals:
        kwargs = dict(ls='--',lw=1.5,c='gray')
        for i,(lo,hi) in enumerate(intervals):
            ax = axes[0] if not i else axes[-1]
            ax.axvline(lo,**kwargs)
            ax.axvline(hi,**kwargs)

    return figure

def parser():
    import argparse
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    
    group = parser.add_argument_group("General Arguments")
    group.add_argument('infile', help="Input data file")
    group.add_argument('-p','--plot',action='store_true',
                       help='Make a corner plot')
    group.add_argument('--seed',default=None,type=int,
                       help="Random number seed")
    group.add_argument('-V','--version',action='version',
                       version='%(prog)s '+__version__,
                       help="Print version and exit")

    group = parser.add_argument_group('Data Arguments')
    group.add_argument('--vel',default='VMEAS',
                       help="Velocity column name")
    group.add_argument('--velerr',default='VERR',
                       help="Velocity error column name")
    group.add_argument('--flag',default=None,
                       help="Optional flag column name")
    group.add_argument('--flagval',default=1,
                       help="Flag selection value")
    
    group = parser.add_argument_group("MCMC Arguments")
    group.add_argument('--nthreads',type=int,default=1,
                       help="Number of threads")
    group.add_argument('--nwalkers',type=int,default=20,
                       help="Number of walkers")
    group.add_argument('--nburn',type=int,default=100,
                       help="Number of initial steps*walkers to burn")
    group.add_argument('--nsteps',type=int,default=5000,
                       help="Number of steps per walker")
    egroup = group.add_mutually_exclusive_group()
    egroup.add_argument('--alpha',default=0.32,type=float,
                       help="Baysian credible pvalue")
    egroup.add_argument('--interval',default=None,type=float,
                       help="Baysian credible interval")
    return parser

if __name__ == "__main__":
    p = parser()
    args = p.parse_args()

    if args.seed is not None: np.random.seed(args.seed)

    if args.interval is not None: 
        alpha = 1-args.interval
    else:            
        alpha = args.alpha

    data = np.genfromtxt(args.infile,names=True,dtype=None)
    if args.flag:
        flag = data[args.flag]
        flagval = np.array(args.flagval).astype(flag.dtype).item()
        data = data[(flag == args.flagval)]

    vel = data[args.vel]
    if args.velerr is None or args.velerr.lower() == 'none':
        velerr = np.zeros_like(vel)
    else:
        velerr = data[args.velerr]

    # Remove nan values (is this fair?)
    cut = (np.isnan(vel) | np.isnan(velerr))
    vel,velerr = vel[~cut],velerr[~cut]

    kwargs = dict(nwalkers=args.nwalkers,nburn=args.nburn,
                  nsteps=args.nsteps,nthreads=args.nthreads)
    samples = mcmc(vel,velerr,**kwargs)
    
    mean,std = scipy.stats.norm.fit(vel)
    print '%-05s : %.2f'%('mean',mean)
    print '%-05s : %.2f'%('std',std)

    intervals = []
    for i,name in enumerate(PARAMS):
        peak,[low,high] = peak_interval(samples[name],alpha=alpha)
        print "%-05s : %.2f [%.2f,%.2f]"%(name,peak,low,high)
        intervals.append([low,high])

    if args.plot:
        try:
            import pylab as plt
            fig = plot(samples,intervals,sigma_clip=4)
            plt.ion(); plt.show()

            outfile = os.path.splitext(args.infile)[0]+'.pdf'
            warnings.filterwarnings('ignore')
            plt.savefig(outfile)
            warnings.resetwarnings()
            
        except ImportError as e:
            msg = '\n '+e.message
            msg +='\n Failed to create plot.'
            warnings.warn(msg)

