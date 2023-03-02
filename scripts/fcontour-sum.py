#!/usr/bin/env python3
import sys
import pandas as pd, numpy as np

from matplotlib import pyplot as plt, ticker #, cm
#from matplotlib.colors import LogNorm
if True:    # change it to False if you do not have LaTeX packages for matplotlib
    from matplotlib import rc
    plt.style.use('bmh')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def fmt(x, pos):
    pos = pos   # avoid warning
    b = int('{:.1e}'.format(x).split('e')[1])
    return r'$10^{%d}$' % (b)


def fcontour(df_file, title='', label=r'$\rho(\omega = 0)$',
    xlabel=r'$\mu / U$', ylabel=r'$U$', savefig='fig.png',
    log=False, trunc=1e-8, color='jet', n_lv=10000, sep=','):
    # read DataFrame file
    df = pd.read_csv(df_file, sep=sep, header=0, index_col=0)

    # sorting
    df = df.reindex(sorted(df.columns), axis=1)     # sorting columns (T values)
    df.sort_index(inplace=True)     # sorting indexes (U values)

    # numpy
    x = df.columns.to_numpy(dtype='float64')
    y = df.index.to_numpy(dtype='float64')
    z = np.abs(df.to_numpy(dtype='float64'))

    # applying transformation to z, example: truncating
    z = np.where(z < trunc, trunc, z)
    #z = lambd(z)

    if log:
        lvls = np.logspace(np.log10(trunc), np.log10(z.max()), n_lv)
        loct = ticker.LogLocator(base=10)
        plot = plt.contourf(x, y, z, levels=lvls, cmap=color, locator=loct)
        plt.colorbar(plot, label=label, format=ticker.FuncFormatter(fmt), ticks=loct)
    else:
        lvls = np.linspace(z.min(), z.max(), n_lv)
        plot = plt.contourf(x, y, z, levels=lvls, cmap=color)
        plt.colorbar(plot, label=label, format='%.3f')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefig, dpi=300, format='png', bbox_inches='tight')
    plt.clf()

## filling
##fcontour(sys.argv[1], title=r'M-I path filling $n$ for $T = 0.004$, $D = 1$',
##        log=False, color='jet_r', label=r'$n$')
#fcontour(sys.argv[1], title=r'M-I path filling $n$ for $T = 0.004$, $D = 1$',
#        log=False, color='jet', label=r'$n$')

# w = 0, log
fcontour(sys.argv[1], savefig='sum-rho_w0-log.png', title=r'sum of $\rho(\omega=0)$ for $T = 0.004$, $D = 1$',
        log=True, color='jet', label=r'$\rho(\omega=0)$', trunc=1e-5)

# w = 0, lin
fcontour(sys.argv[1], savefig='sum-rho_w0-lin.png', title=r'sum of $\rho(\omega=0)$ for $T = 0.004$, $D = 1$',
        log=False, color='jet', label=r'$\rho(\omega=0)$')
