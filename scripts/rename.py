#!/usr/bin/env python3

import sys, os

def right_name(name):
    name = name.strip('.out')
    T = float(name.split(',')[1].split('=')[1])
    U = float(name.split(',')[2].split('=')[1])
    return 'gf_bethe-W=2,T=%.4f,U=%.3f,mu=0.25U.out' % (T, U)

for arg in sys.argv[1:]:
    os.rename(arg, right_name(arg))
