#!/bin/sh

# $1 is U, $2 is mu, $3 is suffix
U_VAL=$(python3 -c "print('%.2f' % ($1-0.05))")
cp "plots_dmft/mu_vs_U-W=2_T=0.004-path_metal-various/gf-data/gf_bethe-W=2,T=0.004,U=$U_VAL,mu=$2U.out" out/gloc.out
python3 T_fixed.py "$1" "$2" "$3"
