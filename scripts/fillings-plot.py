#!/usr/bin/env python3

import sys
import numpy as np
from matplotlib import pyplot as plt
if True:    # change it to False if you do not have LaTeX packages
    from matplotlib import rc
    plt.style.use('bmh')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def plot_fillings(params, figs_dir, U_values, colors, mu_values, fillings):
    mu = [ float(m) for m in mu_values ]
    for i in range(len(U_values)):
        plt.plot(mu, fillings[i], label=r'$U=%s$'%(U_values[i]), color=colors[i])
    plt.xlabel(r'$\mu / U$', fontsize=20)
    plt.ylabel(r'$n$', fontsize=20)
    plt.legend(fontsize=14)
    plt.title(r'$n(\mu)$ para $W=%s, T=%s$' % (params['W'], params['T']))
    plt.savefig(figs_dir+'/filling_'+dt.now().strftime('%Y-%m-%d--%H:%M:%S')+'.png',
                dpi=300, format='png', bbox_inches='tight')
    plt.clf()


def take_zero(x):
    lenx = len(x)
    i = 0
    for i in range(lenx-1):
        if (x[i] * x[i+1] < 0):
            return i
    return -1


def n_metal(m, U, A0):
    return 1 + 1.0 * (m - U/2) * A0


def n_insul(mu, B, U):
    return 2 * (np.exp(B*mu) + np.exp(B*(2*mu-U))) / (1 + 2*np.exp(B*mu) + np.exp(B*(2*mu-U)))


def main():
    colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    W = sys.argv[1].split(',')[1].split('=')[1]
    T = sys.argv[1].split(',')[2].split('=')[1]
    i = 0
    for arg in sys.argv[1:]:    # fillings files in sys.argv
        arg0 = arg[:-5]
        U = arg0.split(',')[0].split('=')[1]
        W = str(round(float(arg0.split(',')[1].split('=')[1])))
        T = arg0.split(',')[2].split('=')[1]
        phase = arg0.split(',')[3]; phase = phase
        mu, n = np.loadtxt(arg, unpack=True)
        line, = plt.plot(mu, n, label=r'$U = %s$' % (U))
        m = np.linspace(mu[0], mu[-1], 200)
        #if phase == 'metal':
        #    freq, reG, imG = np.loadtxt('gf_bethe,U=%s,W=%s,T=%s,mu=0.500U,%s.out' % (U, W, T, phase),
        #                                unpack=True)
        #    reG = reG  # avoid warning
        #    specfun = - imG / np.pi
        #    i_z = take_zero(freq)
        #    A0 = (specfun[i_z] + specfun[i_z + 1]) / 2.0
        #    plt.plot(m, n_metal(m*float(U), float(U), A0) + 0.001, line.get_color(), linestyle=':')
        #elif phase == 'insulator':
        #    plt.plot(m, n_insul(m*float(U), 1/float(T), float(U)) + 0.001, line.get_color(), linestyle=':')
        if phase == 'insulator':
            plt.plot(m, n_insul(m*float(U), 1/float(T), float(U)) + 0.001, line.get_color(), linestyle=':')
        i += 1
    plt.xlabel(r'$\mu / U$', fontsize=20)
    plt.ylabel(r'$n$', fontsize=20)
    plt.legend(fontsize=14)
    plt.title(r'$n(\mu)$ para $W = %s, T = %s$' % (W, T))
    plt.savefig('filling-W=%s,T=%s.png' % (W, T),
                dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
