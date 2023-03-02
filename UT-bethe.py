#!/usr/bin/env python3

import pandas as pd
import numpy as np
#import sys

W = 2       # effective bandwidth, W = 2D = 4t (t is the hopping parameter)
mu = 0.5    # fixed mu, half-filling

dec_U = 4
dec_T = 4
dec_mu = 1
fmtU = f'%.{dec_U}f'
fmtT = f'%.{dec_T}f'
fmtMu = f'%.{dec_mu}f'

mysep = ','
float_fmt = '%15.8e'
index_lab = "      T/U      "
nan_value = "      NaN      "
#all_df = pd.read_csv("results/almost/df_0-W=2,T=0.004.csv", sep=mysep, header=0, index_col=0)

# the variable path will determine how we are going to walk on the U x T diagram
# path = 'metal' corresponds to the metal -> insul path (going to the right)
# path = 'insul' corresponds to the insul -> metal path (going to the left)
path = 'metal'
suffix = '2'

min_T = 0.0084
max_T = 0.01
dT = 0.0001
inv_dT = int(1/dT)
round_T = lambda x: round(inv_dT * x) / inv_dT
g = lambda T: 1.9     # max of U for each T
f = lambda T: 1.7     # min of U for each T
dU = 0.0025
signed_dU = dU if path == 'metal' else -dU
inv_dU = int(1/dU)
round_U = lambda x: round(inv_dU * x) / inv_dU

round100 = lambda x: round(100*x) / 100
round200 = lambda x: round(200*x) / 200

#U_range = np.linspace(1.70, 1.90, 81)
T_range = np.linspace(min_T, max_T, int((max_T - min_T) / dT) + 1)

diagram = {}

for T in T_range:   # f(T) <= U <= g(T) é a região de calculo
    L = []
    if path == 'insul':
        u = round_U(g(T)); fu = f(T)
        while u >= fu:
            L.append(round(u, dec_U))
            u = round_U(u + signed_dU)
        diagram[round(T, dec_T)] = L
    else:       # path == 'metal'
        u = round_U(f(T)); gu = g(T)
        while u <= gu:
            L.append(round(u, dec_U))
            u = round_U(u + signed_dU)
        diagram[round(T, dec_T)] = L

# inner parameters to control DMFT
fq = 3.3        # the range of frequencies is ( -fq * W, +fq * W )
alpha = 0.7     # ratio for the linear mixing
max_err = 3e-6  # max diff between consecutive Green's functions (spectral_norm), 1e-6 recommended on 'acceleration-zitko' article
# dx0 are the Gaussian meshes maximum resolution around zero frequency
dx0 = [2e-5, 2e-4, 2e-2,  0.20]#,  2.00]#, 2e-4, 2e-2,  2e-4,  2e-2]#, 0.20,  2.00]
thermalization_period = 20  # num of linear step iterations of DMFT in the beginning
max_it = 3000   # max number of dmft iterations, if exceed we break


### CODE ###

import warnings
warnings.filterwarnings('ignore')   # suppress warnings
from gaumesh import genmesh     # generate meshes around zero frequency
import os, subprocess, io
from shutil import copyfile
from matplotlib import pyplot as plt
if True:    # change it to False if you do not have LaTeX packages for matplotlib
    from matplotlib import rc
    plt.style.use('bmh')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
from scipy.integrate import simpson
from scipy.interpolate import interp1d


class Bethe:
    def __init__(self, sU, sMu, sT, sW):
        self.name = 'bethe'
        self.sU = sU    # string that has U float value
        self.sMu = sMu  # string that has mu/U float value
        self.sT = sT    # string that has T float value
        self.sW = sW    # string that has W float value
        self.U = float(self.sU)     # float U
        self.mu = float(self.sMu) * self.U  # float mu = sMu * U
        self.T = float(sT)      # temperature T float value
        self.B = 1 / self.T     # B = 1/T, (Boltzmann kB = 1)
        self.W = float(sW)      # effective bandwidth 'W' float value
        self.D = self.W / 2.0   # half-bandwidth
        self.t = self.W / 4.0   # hopping parameter
    def generate_dos(self):
        t = self.t
        return lambda E: 1/(2*np.pi*t**2)*np.sqrt(4-(np.clip(E,-2*t,2*t)/t)**2)


class NCA:
    def __init__(self, bethe, dx0):
        self.bethe = bethe
        self.dx0 = np.array(dx0)        # resolution for mesh near its center
        self.fwhm = 1e+2 * self.dx0     # full width at half maximum
        self.x0 = np.full(len(dx0), 0)  # we later translate it to (-mu + U/2), the approximate quasiparticle peak
        self.version = 'latest'
        self.exe = './nca-%s' % (self.version)
        self.outdir = 'out'
        self.cix_file = 'cix-%s.dat' % (self.version)    # cix file for NCA
        self.freq_file = '%s/freq.000' % (self.outdir)
        self.freq2_file = '%s/freq2.000' % (self.outdir)
        self.gloc_file = '%s/gloc.out' % (self.outdir)
        self.delta_file = '%s/Delta.000' % (self.outdir)
        self.sig_file = '%s/Sigma.000' % (self.outdir)
        self.followPeak = +1    # handle divergences. +1 and -3 are the most robust
        self.max_diff = 1e-6    # maximum diff for convergence
        self.max_steps = 300    # maximum steps at each impurity solver interation
    # solve it by calling the nca executable
    def imp_solver(self):
        status = subprocess.call([self.exe, 'out='+self.outdir,
            'Ac='+self.delta_file, 'Sig='+self.sig_file, 'cix='+self.cix_file,
            f'U={self.bethe.U}', f'T={self.bethe.T}', f'Ed={-self.bethe.mu}',
            f'followPeak={self.followPeak}',
            f'max_diff={self.max_diff}',
            f'max_steps={self.max_steps}',
            'prt=0', 'prtSF=0'],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        gloc = np.loadtxt(self.gloc_file, unpack=True, ndmin=2)   # ndmin=2 is to be a matrix
        Gf = gloc[1] + 1j * gloc[2]
        return Gf, status
    # save hybridization function to nca.delta_file
    def save_bath(self, freq, Gf):
        delta = - np.imag(self.bethe.t**2 * Gf) / np.pi
        np.savetxt(self.delta_file, np.transpose((freq, delta)), fmt=float_fmt)


class DMFT:
    def __init__(self, nca, alpha, max_err, freq_range, path, therm_period):
        self.nca = nca
        bethe = nca.bethe
        self.bethe = nca.bethe
        self.alpha = alpha
        self.max_err = max_err
        self.freq_range = freq_range
        self.path = path
        self.tp = therm_period
        self.figdir = 'results/U_vs_T-W=%s_mu=%sU-path_%s-%s' % (bethe.sW, bethe.sMu, self.path, suffix)    # directory to save plots
        self.gf_data = '%s/gf-data' % (self.figdir)
        self.plotdir = '%s/specfun-plots' % (self.figdir)
        self.gf_file = '%s/gf_%s-W=%s,U=%s,T=%s,mu=%sU.out' % (self.gf_data,
                        bethe.name, bethe.sW, bethe.sU, bethe.sT, bethe.sMu)
        self.specfun_png = '%s/%s-W=%s,U=%s,T=%s,mu=%sU.png' % (self.plotdir, bethe.name,
                         bethe.sW, bethe.sU, bethe.sT, bethe.sMu)
        self.gloc_start = '%s/gloc_start.in' % (nca.outdir)


    def generate_freqs(self):
        freq_range = self.freq_range
        bethe, nca = self.bethe, self.nca
        freq  = np.array(genmesh(nca.dx0, nca.fwhm, nca.x0,
                  xmin=nca.x0[0]-freq_range*bethe.W, xmax=nca.x0[0]+freq_range*bethe.W))
        freq2 = np.array(genmesh(nca.dx0, nca.fwhm, nca.x0,
                xmin=nca.x0[0]-2*freq_range*bethe.W, xmax=nca.x0[0]+2*freq_range*bethe.W))
        np.savetxt(nca.freq_file, freq, fmt=float_fmt)
        np.savetxt(nca.freq2_file, freq2, fmt=float_fmt)
        return freq


    # initial guess for the Green function
    def initial_guess(self):
        freq, freq2 = self.generate_freqs()
        bethe, nca = self.bethe, self.nca
        Sig = bethe.U / 20.0 * (1 - 1j)
        if self.path == 'metal':      # trying to guess a metallic Green's function
            Gf = 2 / ((freq + bethe.mu - Sig) + np.sqrt((freq + bethe.mu - Sig)**2 - (2*bethe.t)**2))
        else:       # trying to guess an insulator Green's function
            Gf = 1 / ((freq + bethe.mu)**2 - Sig*20) + 1 / ((freq + bethe.mu)**2 + Sig*20)
        np.savetxt(nca.sig_file, np.transpose((freq2)), fmt=float_fmt)
        return freq, Gf


    def copy_guess(self, copy_file):
        freq = np.loadtxt(self.nca.freq_file, unpack=True)
        freq2 = np.loadtxt(self.nca.freq2_file, unpack=True)
        np.savetxt(self.nca.sig_file, np.transpose((freq2)), fmt=float_fmt)
        Gdata = np.loadtxt(copy_file, unpack=True)
        Gf = Gdata[1] + 1j * Gdata[2]
        return freq, Gf


    def update_start(self):
        copyfile(self.nca.gloc_file, self.gloc_start)


    # if colour is not False, it is the color of the plot
    # returns (filling, A(0))
    def analyze(self, colour='no'):
        bethe = self.bethe
        # loading data
        data = np.loadtxt(self.gf_file, unpack=True)
        specfun = - data[2] / np.pi
        # calculating the filling
        fill = filling(data[0], specfun, bethe.T)
        # plot spectral function, only if colour != False
        if colour != 'no':
            plt.plot(data[0], specfun, label=r'$n = %.3f$' % (fill), color=colour)
            #c = - bethe.mu + bethe.U/2
            c = 0
            range = 2.5 * bethe.W
            plt.xlim([c - range, c + range])
            plt.xlabel(r'$\omega$', fontsize=20)
            plt.ylabel(r'$A(\omega)$', fontsize=20)
            plt.legend(fontsize=12)
            plt.title(r'%s: $W=%s$, $T=%s$, $U=%s$, $\mu=%s U$' % (bethe.name,
                        bethe.sW, bethe.sT, bethe.sU, bethe.sMu))
            plt.savefig(self.specfun_png, dpi=300, format='png', bbox_inches='tight')
            plt.clf()
        # getting A(0) by interpolation
        A = interp1d(data[0], specfun, kind='cubic')
        return fill, A(0)


    def run(self, copy_file='-'):
        # make sure the directories exist
        for dir in [self.nca.outdir, self.figdir, self.gf_data, self.plotdir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # make sure freq files exists
        if not (os.path.isfile(self.nca.freq_file) and os.path.isfile(self.nca.freq2_file)):
            self.generate_freqs()

        if copy_file == '-':    # '-' means do not copy and try initial guess
            freq, Gf = self.initial_guess()
        elif copy_file == ';':  # ';' means copy last gloc.out file generated by NCA
            freq, Gf = self.copy_guess(self.nca.gloc_file)
        elif copy_file == '^':  # '^' means copy path start file
            freq, Gf = self.copy_guess(self.gloc_start)
        else:   # copy_file is copied
            freq, Gf = self.copy_guess(copy_file)

        N, status, diff = 1, 0, 1000
        #dG, dF = [], []
        while diff > self.max_err:
            Gf_old = Gf[:]
            Gf, N, status = self.linear_step(N, self.alpha, freq, Gf_old)
            diff = spectral_norm(freq, Gf - Gf_old)
            print(f'iteration {N-1}: diff = {diff}')
            if N > max_it:     # max it
                print(f"max iterations {max_it} reached, we are going to break")
                status = 1
            if status == 1:
                print('error: status == 1.')
                break

        if os.path.isfile('./cores.dat'):   # remove file ./cores.dat if it exists
            os.remove('./cores.dat')

        if status == 0:
            if not os.path.exists(self.figdir):   # make sure the directory figs_dir exists
                os.makedirs(self.figdir)
            # save the Green's function in the end
            np.savetxt(self.gf_file, np.transpose((freq, np.real(Gf), np.imag(Gf))), fmt=float_fmt)
        else:
            print('error: nothing done because status == 1.')

        return status


    def func(self, freq, G):
        self.nca.save_bath(freq, G)
        G_new, status = self.nca.imp_solver()
        return G_new - G, status


    def linear_step(self, N, alpha, freq, G):
        F, status = self.func(freq, G)
        return G + alpha * F, N+1, status


    # c eh um array 1 x (N-1)
    # A e B sao matrizes (N-1) x (N-1)
    # dF e dG sao listas de np.array da forma (N-1) x L, U eh uma matriz (N-1) x (L = numero de frequencias)
    # portanto c @ (B @ U) eh do shape 1 x L
    def broyden_step(self, N, alpha, freq, G, dG, dF):
        N_true = N; N = N - self.tp
        F, status_1 = self.func(freq, G)
        if N == 1:
            soma = 0
        else:
            w0 = 0.01
            # np.dot(a, b) eh na verdade a_1 * b_1 + ... + a_n * b_n (nao eh o produto escalar complexo!)
            # por isso temos que fazer np.dot(np.conj(a), b)
            c = np.array([np.dot(np.conj(dF[k]), F) for k in range(N-1)], ndmin=2)
            U = np.array([alpha * dF[n] + dG[n] for n in range(N-1)], ndmin=2)
            A = np.array([[np.dot(np.conj(dF[n]), dF[k]) for n in range(N-1)] for k in range(N-1)], ndmin=2)
            B = np.linalg.inv(w0*w0*np.identity(N-1) + A)
            soma = c @ (B @ U)
        G_next = np.ravel(G + alpha * F - soma)     # 1d array
        F_next, status_2 = self.func(freq, G_next)
        dG.append((G_next - G) / (np.linalg.norm(F_next - F)))      # np.linalg.norm gets complex norm correct
        dF.append((F_next - F) / (np.linalg.norm(F_next - F)))      # np.linalg.norm(a) = np.sqrt(np.real(np.dot(np.conj(a), a)))
        return G_next, N_true+1, status_1 or status_2


def spectral_norm(freq, gf):
    gf = - np.imag(gf) / np.pi
    return simpson(np.abs(gf), freq)


def fermi(x):
    return np.where(x>200, 0, np.where(x<-200, 1, 1/(np.exp(x)+1)))


def filling(freq, specfun, T):
    return 2 * simpson(specfun * fermi(freq/T), freq)


def sprint(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def main():
    df_n = pd.DataFrame()   # dataframe for the filling n
    df_0 = pd.DataFrame()   # dataframe for A(0)

    #oneshot = True  # we start with an initial guess, after that we copy previous guesses
    num = 1; total = sum([len(diagram[i]) for i in diagram])
    oneshot = True
    for T in diagram:
        start = True
        for U in diagram[T]:
            print("U = %s, T = %s, mu = %s, progress = %d/%d" % (fmtU % U, str(T), fmtMu % mu, num, total))
            bethe = Bethe(fmtU % U, fmtMu % mu, fmtT % T, str(W))
            nca = NCA(bethe, dx0)
            dmft = DMFT(nca, alpha, max_err, fq, path, thermalization_period)

            if oneshot:
                oneshot = False; start = False
                copy_file = "results/U_vs_T-W=2_mu=0.5U-path_metal-1/gf-data/gf_bethe-W=2,U=%s,T=%s,mu=0.5U.out" % (fmtU % (diagram[T][0]), fmtT % (T))
            elif start:
                start = False
                copy_file = "%s/gf_bethe-W=%s,U=%s,T=%s,mu=%sU.out" % (dmft.gf_data, str(W), fmtU % U, fmtT % (T - dT), fmtMu % mu)
            else:
                copy_file = "%s/gf_bethe-W=%s,U=%s,T=%s,mu=%sU.out" % (dmft.gf_data, str(W), fmtU % (U - signed_dU), fmtT % T, fmtMu % mu)

            if copy_file != "-" and os.path.exists(copy_file):
                print(f'found copy_file = {copy_file}')

            status = dmft.run(copy_file)
            num += 1

            if status == 1:     # ERROR, exit? the program
                print('ERROR at DMFT.')
                print('but we are going to continue...')
                #exit(1)

            if status == 0:
                n, a0 = dmft.analyze(colour='#348ABD')
                print("OK: U = %s, T = %s, mu = %s, A0 = %.3e, n = %.4f" % (fmtU % U, fmtT % T, fmtMu % mu, a0, n))
            else:
                n, a0 = np.nan, np.nan
                print("NOT OK: U = %s, T = %s, mu = %s, A0 = nan, n = nan" % (fmtU % U, fmtT % T, fmtMu % mu))

            # saving filling dataframe to file
            df_n.loc[T, U] = n
            df_n.to_csv('%s/df_n-W=%s,mu=%s.csv' % (dmft.figdir, str(W), fmtMu % mu),
                        sep=',', index_label=index_lab, na_rep=nan_value, float_format=float_fmt)

            # saving zero frequency peak dataframe to file
            df_0.loc[T, U] = a0
            df_0.to_csv('%s/df_0-W=%s,mu=%s.csv' % (dmft.figdir, str(W), fmtMu % mu),
                        sep=',', index_label=index_lab, na_rep=nan_value, float_format=float_fmt)

if __name__ == '__main__':
    main()
