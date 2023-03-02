#!/usr/bin/env python3

import sys
import numpy as np
from matplotlib import pyplot as plt
if True:    # change it to False if you do not have LaTeX packages
    from matplotlib import rc
    plt.style.use('bmh')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


U = '2.0'
T_values = ['0.005', '0.020', '0.050', '0.100', '1.000']


def n_insul(mu, B, U):
    return 2 * (np.exp(B*mu) + np.exp(B*(2*mu-U))) / (1 + 2*np.exp(B*mu) + np.exp(B*(2*mu-U)))


def main():
    colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    i = 0
    for T in T_values:    # fillings files in sys.argv
        mu = np.linspace(-0.2, 1.2, 300)
        plt.plot(mu, n_insul(mu*float(U), 1/float(T), float(U)), label=r'$T = %s$' % (T))
        i += 1
    plt.xlabel(r'$\mu / U$', fontsize=20)
    plt.ylabel(r'$n(\mu)$', fontsize=20)
    plt.legend(fontsize=14)
    plt.title(r'theoretical $U/t \gg 1$ limit: $n(\mu)$ for $U = %s$' % (U))
    plt.savefig('theo_filling-U=%s.png' % (U),
                dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
