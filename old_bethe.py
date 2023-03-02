#!/usr/bin/env python3

from matplotlib import pyplot as plt
if True:    # change it to False if you do not have LaTeX packages
    from matplotlib import rc
    plt.style.use('bmh')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
import os, subprocess, sys
import numpy as np
from scipy.integrate import simpson as intg
from gaumesh import genmesh

U_values = ['1.2']
#U_values = ['1.70','1.71','1.72','1.73','1.74','1.75','1.76','1.77',
#            '1.78','1.79','1.80','1.81','1.82','1.83','1.84','1.85',
#            '1.86','1.87','1.88','1.89','1.90','2.00','2.10']
params = {
    'version' : 'latest',       # 'latest' or '2004'
    'lattice' : 'bethe',        # 'bethe'
    'mixing' : 'linear',        # 'broyden' or 'linear'
    #'mixing' : 'broyden',       # 'broyden' or 'linear'
    'phase' : 'metal',          # initial phase: 'metal' or 'insulator'
    #'phase' : 'insulator',      # initial phase: 'metal' or 'insulator'
    'T' : '0.01',               # temperature
    'mu' : '0.85',              # (in units of U), 0.5 for half-filling
    'W' : '2',                  # effective bandwidth
    'followPeak' : +1,          # handle divergences. +1 and -3 are the most robust
    'max_diff' : 1e-6,          # maximum diff for convergence
    'max_steps' : 300,          # maximum steps at each impurity solver interation
}

### adjustable inner parameters ###
alpha = 0.8     # mixing ratio
max_err = 1e-3  # minimum difference between consecutive self-energies
dx0  = [2e-5, 2e-4, 2e-2,  0.20]#,  2.00]#, 2e-4, 2e-2,  2e-4,  2e-2]#, 0.20,  2.00]
fwhm = [2e-3, 2e-2, 2.00,  20.0]#, 200.0]#, 2e-2,  2.0,  2e-2,   2.0]#, 20.0, 200.0]


class lattice:
    def __init__(self, name, W):
        self.name = name
        self.W = W
        self.t = W / 4.0
    def dos(self):
        t = self.t
        return lambda E: 1/(2*np.pi*t**2)*np.sqrt(4-(np.clip(E,-2*t,2*t)/t)**2)


# plot spectral function
def plot_specfun(params, figs_dir, file, fill):
    name, U, W, T, mu = params['lattice'], params['U'], params['W'], params['T'], params['mu']
    U_ = float(U); mu_ = float(params['mu']) * U_
    # loading data
    data = np.loadtxt(file, unpack=True)
    specfun = - data[2] / np.pi
    # plotting
    plt.plot(data[0], specfun, label=r'$n = ' + f'{round(fill, 3)}' + r'$')
    plt.xlim([xlim_a - mu_ + U_/2, xlim_b - mu_ + U_/2])   # we plot for x in [a, b] translated
    #plt.xlim([xlim_a, xlim_b])   # we plot for x in [a, b]
    #plt.ylim([ylim_a, ylim_b])
    plt.xlabel(r'$\omega + \mu - U/2$', fontsize=20)
    plt.ylabel(r'$A(\omega)$', fontsize=20)
    plt.legend(fontsize=12)
    plt.title(r'%s lattice, %s: $U=%s$, $W = %s$, $T=%s$, $\mu = %s U$' % (name, params['phase'], U, W, T, mu))
    plt.savefig(figs_dir+'/%s-U=%s,W=%s,T=%s,mu=%sU,%s.png' % (name, U, W, T, mu, params['phase']),
                dpi=300, format='png', bbox_inches='tight')
    plt.clf()


# Broyden first iteration, it returns (V2, dF=[], u=[], N=2)
def broyden_init(V1, F1):
    return V1 + alpha * F1, [], [], 2


# Convergence acceleration and stabilization for DMFT calculations
# Appendix B of Real-frequency impurity solver for DMFT based on cluster perturbation theory
def broyden(V1, V2, F1, F2, dF, u, N):
    w0 = 0.01
    dF_e = (F2 - F1) / np.linalg.norm(F2 - F1)
    dV = (V2 - V1) / np.linalg.norm(F2 - F1)
    dF.append(dF_e)
    u.append(alpha * dF_e + dV)
    u_arr = np.array(u, ndmin=2)
    c, A = [], []
    for k in range(N-1):
        c.append(np.dot(F2, np.conj(dF[k])))
        A_lin = []
        for n in range(N-1):
            A_lin.append(np.dot(dF[k], np.conj(dF[n])))
        A.append(A_lin)
    c, A = np.array(c, ndmin=2), np.array(A, ndmin=2)
    beta = np.linalg.inv(w0*w0 + A)
    #soma = 0
    ##print('c[0] =', c[0])
    ##print('beta[0][0] =', beta[0])
    ##print('u_arr[0] =', u_arr[0])
    #for k in range(N-1):
    #    for n in range(N-1):
    #        soma += c[0][k] * beta[k][n] * u_arr[n]
    V3 = V2 + alpha*F2 - c @ (beta @ u_arr)
    return V3.ravel(), N+1


# calculate new bath function and save it to bath_file
def save_bath(lat, freq, Gf, bath_file):
    t = lat.t
    delta = - np.imag(t*t*Gf) / np.pi
    np.savetxt(bath_file, np.transpose((freq, delta)), fmt='%15.8e')


def fermi(x):
    return 1.0 / (np.exp(x) + 1)


def filling(freq, Gf, T):
    A = - np.imag(Gf) / np.pi
    n = intg(A * fermi(freq/T), freq)
    return n


# initial guess for bath function
def initial_guess(dx0, fwhm, x0, lat, W, sig_file):
    freq = -mu + np.array(genmesh(dx0, fwhm, x0, xmin=-3.3*float(W), xmax=3.3*float(W)))
    freq2 = -mu + np.array(genmesh(dx0, fwhm, x0, xmin=-6.6*float(W), xmax=6.6*float(W)))
    t = lat.t
    Sig = U / 20.0 * (1 - 1j)
    if params['phase'] == 'metal':
        #Gf = np.array([intg(dos(freq + mu) / (w + mu - Sig - freq), freq) for w in freq])   # translation by +mu
        Gf = 2 / ((freq + mu - Sig) + np.sqrt((freq + mu - Sig)**2 - (2*t)**2))     # translation by +mu
    else:
        Gf = 1 / ((freq + mu) **2 - Sig*20) + 1 / ((freq + mu)**2 + Sig*20)     # translation by +mu
    np.savetxt(sig_file, np.transpose((freq2)), fmt='%15.8e')
    return freq, Gf


def copy_guess(gf_file):
    freq, reG, imG = np.loadtxt(gf_file, unpack=True)
    Gf = reG + 1j * imG
    return freq, Gf


# impurity solver (NCA)
def imp_solver(bath_file, sig_file):
    status = subprocess.call([nca, 'out=' + out_dir,
        'Ac='+bath_file, 'Sig='+sig_file, 'cix='+cix,
        f'U={U}', f'T={T}', f'Ed={-mu}',
        f'followPeak={followPeak}',
        f'max_diff={max_diff}',
        f'max_steps={max_steps}',
        'prt=0', 'prtSF=0'],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    gloc = np.loadtxt(out_dir+'/gloc.out',
           unpack=True, ndmin=2)   # ndmin=2 is to be a matrix
    Gf = gloc[1] + 1j * gloc[2]
    return Gf, status


def broyden_main(params):
    if not os.path.exists(out_dir):     # make sure the directory 'out' exists
        os.makedirs(out_dir)

    freq, Gf1 = initial_guess(dx0, fwhm, x0, lat, W, sig_f)
    save_bath(lat, freq, Gf1, delta_f)
    F1, status = imp_solver(delta_f, sig_f)
    F1 = F1 - Gf1
    Gf2, dF, u, n = broyden_init(Gf1, F1)

    status, diff = 0, 1000
    while diff > max_err:
        save_bath(lat, freq, Gf2, delta_f)
        print('entering impurity...')
        F2, status = imp_solver(delta_f, sig_f)
        F2 = F2 - Gf2
        print('impurity done')
        Gf_cp = Gf2[:]
        print('entering broyden')
        Gf2, n = broyden(Gf1, Gf_cp, F1, F2, dF, u, n)
        print('broyden done')
        Gf1, F1 = Gf_cp[:], F2[:]
        diff = np.max(np.abs(Gf2 - Gf1))
        print(f'iteration {n}: diff = {diff}')
        if status == 1:
            print('error: status = 1')
            break

    if os.path.isfile('./cores.dat'):   # remove file ./cores.dat if it exists
        os.remove('./cores.dat')

    if status == 0:
        if not os.path.exists(figs_dir):   # make sure the directory figs_dir exists
            os.makedirs(figs_dir)
        # save the Green's function in the end
        np.savetxt(gf_file,
                   np.transpose((freq, np.real(Gf2), np.imag(Gf2))),
                   fmt='%15.8e')
        fill = 1
        plot_specfun(params, figs_dir, gf_file, fill)
    else:
        print('error: nothing done because status = 1')


def linear_main(params):
    if not os.path.exists(out_dir):     # make sure the directory 'out' exists
        os.makedirs(out_dir)

    len_arg = len(sys.argv)
    if len_arg == 1:
        freq, Gf = initial_guess(dx0, fwhm, x0, lat, W, sig_f)
    elif len_arg == 2:
        freq, Gf = copy_guess(sys.argv[1])
    else:
        print('only accept 0 or 1 argument')
        exit(1)

    n, status, diff = 1, 0, 1000
    while diff > max_err:
        save_bath(lat, freq, Gf, delta_f)
        Gf_old = Gf[:]     # old
        Gf, status = imp_solver(delta_f, sig_f)
        Gf = alpha * Gf + (1 - alpha) * Gf_old
        diff = np.max(np.abs(Gf - Gf_old))
        print(f'iteration {n}: diff = {diff}')
        n += 1
        if status == 1:
            print('error: status = 1')
            break

    if os.path.isfile('./cores.dat'):   # remove file ./cores.dat if it exists
        os.remove('./cores.dat')

    if status == 0:
        if not os.path.exists(figs_dir):   # make sure the directory figs_dir exists
            os.makedirs(figs_dir)
        # save the Green's function in the end
        np.savetxt(gf_file,
                   np.transpose((freq, np.real(Gf), np.imag(Gf))),
                   fmt='%15.8e')
        fill = 2 * filling(freq, Gf, T)
        plot_specfun(params, figs_dir, gf_file, fill)
        print('filling =', fill)
    else:
        print('error: nothing done because status = 1')


if __name__ == '__main__':
    if params['mixing'] == 'broyden':
        main = broyden_main
    else:
        main = linear_main

    for U_str in U_values:
        ### non-adjustable global parameters ###
        params['U'] = U_str; U = float(params['U'])
        W = float(params['W'])
        D = W / 2.0
        lat = lattice(params['lattice'], W)
        T = float(params['T'])
        mu = float(params['mu']) * U
        #x0 = np.full(len(dx0), 0)
        x0 = np.full(len(dx0), -mu + U/2)
        followPeak = params['followPeak']
        max_diff = params['max_diff']
        max_steps = params['max_steps']
        xlim_a, xlim_b = -2.5*W, 2.5*W  # range for plotting

        ### files ###
        figs_dir = 'plots_dmft' # where plots will be saved
        gf_file = figs_dir+'/gf_%s,U=%s,W=%s,T=%s,mu=%sU,%s.out' % (params['lattice'],   # final output for local green's function
                params['U'], params['W'], params['T'], params['mu'], params['phase'])
        out_dir = 'out'     # output directory for impurity solver files
        delta_f = '%s/delta.loop' % (out_dir)    # delta file at each iteration
        sig_f = '%s/Sigma.000' % (out_dir)    # sig file at each iteration
        nca = './nca-%s' % (params['version'])  # impurity solver executable
        cix = 'cix-%s.dat' % (params['version'])    # cix file for NCA

        main(params)
