#!/usr/bin/env python3

import sys
import numpy as np
from matplotlib import pyplot as plt
if True:    # change it to False if you do not have LaTeX packages
    from matplotlib import rc
    plt.style.use('bmh')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
from scipy.integrate import simpson


def fermi(x):
    return np.where(x>200, 0, np.where(x<-200, 1, 1/(np.exp(x)+1)))


def filling(freq, specfun, T):
    return 2 * simpson(specfun * fermi(freq/T), freq)


def main():
    colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    W = '2'
    T = '0.05'
    n, i = 0, 1
    U_old = '3.0'
    for arg in sys.argv[1:]:    # fillings files in sys.argv
        arg0 = arg[9:-11]
        comma_split = arg0.split(',')
        U  = comma_split[0].split('=')[1]
        if U != U_old:
            i += 1
            U_old = U
        W  = comma_split[1].split('=')[1]
        T  = comma_split[2].split('=')[1]
        mu = comma_split[3].split('=')[1]
        data = np.loadtxt(arg, unpack=True)
        specfun = - data[2] / np.pi
        fill = filling(data[0], specfun, float(T))
        #plt.plot(data[0] + float(mu) * float(U) - float(U)/2, specfun, label=r'$n = %.3f$' % (fill), color=colors[i])
        plt.plot(data[0], specfun, label=r'$n = %.3f$' % (fill), color=colors[i])
        plt.xlim([-2.5 * float(W), 2.5 * float(W)])
        #plt.xlabel(r'$\omega\prime = \omega + \mu - U/2$', fontsize=20)
        #plt.ylabel(r'$A(\omega\prime)$', fontsize=20)
        plt.xlabel(r'$\omega$', fontsize=20)
        plt.ylabel(r'$A(\omega)$', fontsize=20)
        plt.legend(fontsize=12)
        plt.title(r'$W=%s$, $T=%s$, $U=%s$, $\mu=%s \, U$' % (W, T, U, mu))
        plt.savefig('new_figs_omegaprime/orig-specfun_U=%s_img-%d.png' % (U, n), dpi=300, format='png', bbox_inches='tight')
        plt.clf()
        n += 1


if __name__ == '__main__':
    main()
