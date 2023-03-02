#!/bin/sh

#./T_fixed "2.25" "0.375" "U=2.25,mu=0.375"

sed 's/^\([0-9]\.[0-9]\+\) \+\([0-9]\.[0-9]\+\)$/\1 \2 U=\1,mu=\2/' missing.txt | xargs -n 3 ./run_vals
