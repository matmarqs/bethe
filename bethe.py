#!/usr/bin/env python3

import numpy as np
import sys

W = 2       # effective bandwidth, W = 2D = 4t (t is the hopping parameter)
T = 0.004   # fixed temperature

# 'mus' and 'Us' are arrays of mu's and U's
# They need to be ORDERED. You can use np.linspace or np.logspace
mus = np.array([float(sys.argv[2])])
fmtMu = '%.3f'   # mu / U
#Us = np.linspace(1.6, 5.0, 201); fmtU = '%.2f'
Us = np.array([float(sys.argv[1])])
fmtU = '%.2f'

suffix = sys.argv[3]

# the variable path will determine how we are going to walk on the U x T diagram
# path can be 'metal' or 'insul'
# 'metal' corresponds to the metal -> insul path (going to the right)
# 'insul' corresponds to the insul -> metal path (going to the left)
path = 'metal'

# inner parameters to control DMFT
fq = 3.3        # the range of frequencies is ( -fq * W, +fq * W )
alpha = 0.8     # ratio for the linear mixing
max_err = 1e-3  # maximum difference between consecutive Green's functions
# dx0 are the Gaussian meshes maximum resolution around zero frequency
dx0 = [2e-5, 2e-4, 2e-2,  0.20]#,  2.00]#, 2e-4, 2e-2,  2e-4,  2e-2]#, 0.20,  2.00]

# aesthetics parameters
float_fmt = '%15.8e'
index_lab = "     U/mu      "
nan_value = "      NaN      "


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
import pandas as pd


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
    def __init__(self, nca, alpha, max_err, freq_range, path):
        self.nca = nca
        bethe = nca.bethe
        self.bethe = nca.bethe
        self.alpha = alpha
        self.max_err = max_err
        self.freq_range = freq_range
        self.path = path
        self.figdir = 'plots_dmft/mu_vs_U-W=%s_T=%s-path_%s-%s' % (bethe.sW, bethe.sT, self.path, suffix)    # directory to save plots
        self.gf_data = '%s/gf-data' % (self.figdir)
        self.plotdir = '%s/specfun-plots' % (self.figdir)
        self.gf_file = '%s/gf_%s-W=%s,T=%s,U=%s,mu=%sU.out' % (self.gf_data,
                        bethe.name, bethe.sW, bethe.sT, bethe.sU, bethe.sMu)
        self.specfun_png = '%s/%s-W=%s,T=%s,U=%s,mu=%sU.png' % (self.plotdir, bethe.name,
                         bethe.sW, bethe.sT, bethe.sU, bethe.sMu)
        self.gloc_start = '%s/gloc_start.in' % (nca.outdir)

    # initial guess for the Green function
    def initial_guess(self):
        freq_range = self.freq_range
        bethe, nca = self.bethe, self.nca
        freq  = np.array(genmesh(nca.dx0, nca.fwhm, nca.x0,
                  xmin=nca.x0[0]-freq_range*bethe.W, xmax=nca.x0[0]+freq_range*bethe.W))
        freq2 = np.array(genmesh(nca.dx0, nca.fwhm, nca.x0,
                xmin=nca.x0[0]-2*freq_range*bethe.W, xmax=nca.x0[0]+2*freq_range*bethe.W))
        np.savetxt(nca.freq_file, freq, fmt=float_fmt)
        np.savetxt(nca.freq2_file, freq2, fmt=float_fmt)
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

    def run(self, copy_file='-'):
        # make sure the directories exist
        for dir in [self.nca.outdir, self.figdir, self.gf_data, self.plotdir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        if copy_file == '-':    # '-' means do not copy and try initial guess
            freq, Gf = self.initial_guess()
        elif copy_file == ';':  # ';' means copy last gloc.out file generated by NCA
            freq, Gf = self.copy_guess(self.nca.gloc_file)
        elif copy_file == '^':  # '^' means copy path start file
            freq, Gf = self.copy_guess(self.gloc_start)
        else:   # copy_file is copied
            freq, Gf = self.copy_guess(copy_file)

        i, status, diff = 1, 0, 1000
        while diff > self.max_err:
            self.nca.save_bath(freq, Gf)
            Gf_old = Gf[:]     # old Green function
            Gf, status = self.nca.imp_solver()
            Gf = self.alpha * Gf + (1 - self.alpha) * Gf_old
            diff = np.max(np.abs(Gf - Gf_old))
            print(f'iteration {i}: diff = {diff}')
            i += 1
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

    # if colour is not False, it is the color of the plot
    # returns (filling, A(-mu + U/2), A(0))
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
        # getting A(-mu + U/2) and A(0) by interpolation
        A = interp1d(data[0], specfun, kind='cubic')
        return fill, A(0)

    def write_parameters(self, Us, mus):
        text = """W = %s
T = %s
fq = %s  # frequency range if ( -fq * W, +fq * W )
alpha = %s  # mixing ratio
max_err = %.3e  # convergence difference
dx0 = %s  # meshes maximum resolution around zero frequency
path = '%s'  # chosen path, %s

Us  =    # %d values
%s

mus =    # %d values
%s

# %d * %d = %d points""" % (self.bethe.sW, self.bethe.sT, self.freq_range, self.alpha, self.max_err,
                     sprint(self.nca.dx0, end=''), self.path,
                     'metal -> insulator' if path == 'metal' else 'insulator -> metal',
                     len(Us), sprint(Us, end=''),
                     len(mus), sprint(mus, end=''), len(Us), len(mus), len(Us) * len(mus))

        filepath = self.figdir + '/params.txt'
        with open(filepath, 'w') as text_file:
            text_file.write(text)


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
    if path == 'insul':
        U_path = np.flip(Us)
    elif path == 'metal':
        U_path = Us
    else:
        sys.stderr.write("ERROR: variable 'path' has to be 'metal' or 'insul'.\n")
        exit(1)

    df_n = pd.DataFrame()   # dataframe for the filling n
    df_0 = pd.DataFrame()   # dataframe for A(0)

    oneshot = True  # we start with an initial guess, after that we copy previous guesses
    num = 1; total = len(Us) * len(mus)
    for mu in mus:
        start = True
        for U in U_path:
            print("U = %s, T = %s, mu = %s, progress = %d/%d" % (fmtU % U, str(T), fmtMu % mu, num, total))
            bethe = Bethe(fmtU % U, fmtMu % mu, str(T), str(W))
            nca = NCA(bethe, dx0)
            dmft = DMFT(nca, alpha, max_err, fq, path)

            copy_file = '^' if oneshot else ('^' if start else '^')
            status = dmft.run(copy_file)
            num += 1

            if status == 1:     # ERROR, exit the program
                print('ERROR at DMFT.')
                exit(1)

            if oneshot:
                dmft.write_parameters(Us, mus)
            oneshot = False
            if start:
                dmft.update_start()
                start = False

            n, a0 = dmft.analyze(colour='#348ABD')
            print("OK: U = %s, mu = %s, A0 = %.3e, n = %.4f" % (fmtU % U, fmtMu % mu, a0, n))

            # saving filling dataframe to file
            df_n.loc[U, mu] = n
            df_n.to_csv('%s/df_n-W=%s,T=%s.csv' % (dmft.figdir, str(W), str(T)),
                        sep=',', index_label=index_lab, na_rep=nan_value, float_format=float_fmt)

            # saving zero frequency peak dataframe to file
            df_0.loc[U, mu] = a0
            df_0.to_csv('%s/df_0-W=%s,T=%s.csv' % (dmft.figdir, str(W), str(T)),
                        sep=',', index_label=index_lab, na_rep=nan_value, float_format=float_fmt)

    #color_plot(df_f)
    #color_plot(df_m)
    #color_plot(df_0)


if __name__ == '__main__':
    main()
