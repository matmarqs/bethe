#!/usr/bin/env python3
import sys
import pandas as pd, numpy as np

from matplotlib import pyplot as plt, ticker, cm
#from matplotlib.colors import LogNorm
if True:    # change it to False if you do not have LaTeX packages for matplotlib
    from matplotlib import rc
    plt.style.use('bmh')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

#def fmt(x, pos):
#    pos = pos   # avoid warning
#    a, b = '{:.1e}'.format(x).split('e')
#    b = int(b)
#    return r'${} \times 10^{{{}}}$'.format(a, b)
def fmt(x, pos):
    pos = pos   # avoid warning
    b = int('{:.1e}'.format(x).split('e')[1])
    return r'$10^{%d}$' % (b)

truncate = float(sys.argv[4])

# read DataFrame file
typ = sys.argv[1].split('-')[0]
if typ != 'metal' and typ != 'insul':
    print("the file must start with 'metal' or 'insul'")
    exit(1)

df_file = sys.argv[1]
df = pd.read_csv(df_file, sep=',', header=0, index_col=0, na_values='NaN')

# sorting
df = df.reindex(sorted(df.columns), axis=1)     # sorting columns (T values)
df.sort_index(inplace=True)     # sorting indexes (U values)

# numpy
x = df.index.to_numpy(dtype='float64')
y = df.columns.to_numpy(dtype='float64')
z = np.abs(np.transpose(df.to_numpy(dtype='float64')))

if len(sys.argv) > 2 and sys.argv[2] == 'r':
    thecolor = cm.get_cmap('jet_r')
else:
    thecolor = cm.get_cmap('jet')

if len(sys.argv) > 3:
    scale = sys.argv[3]
else:
    scale = 'lin'

path = 'metal-insulator' if typ == 'metal' else 'insulator-metal'

if scale == 'log':
    z = np.where(z < truncate, truncate, z)
    locator = ticker.LogLocator(base=10)
    #lvls = np.logspace(np.log10(z.min()), np.log10(z.max()), 1000)
    lvls = np.logspace(np.log10(truncate), np.log10(z.max()), 10000)
    plot = plt.contourf(x, y, z, levels=lvls, cmap=thecolor, locator=locator)
    clb  = plt.colorbar(plot, format=ticker.FuncFormatter(fmt), ticks=locator, label=r'$\rho(\omega = 0)$')
    plt.title(r'%s path $\rho(\omega = 0)$ for $\mu = U/4$, $D = 1$' % (path))
else:
    lvls = np.linspace(z.min(), z.max(), 10000)
    plot = plt.contourf(x, y, z, levels=lvls, cmap=thecolor)
    clb  = plt.colorbar(plot, format='%.4f', label=r'$n$')
    plt.title(r'%s path filling $n$ for $\mu = U/4$, $D = 1$' % (path))

plt.xlabel(r'$U$', fontsize=14)
plt.ylabel(r'$T$', fontsize=14)
plt.savefig('%s-fig_%.1e.png' % (typ, truncate), dpi=300, format='png', bbox_inches='tight')
