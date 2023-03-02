#!/usr/bin/env python3

import sys, os
import pandas as pd

for arg in sys.argv[1:]:
    file_1 = arg
    file_2 = arg[:-4] + '-append.csv'
    if not os.path.exists(file_1):
        print("file 1 does not exist.")
        exit(1)
    if not os.path.exists(file_2):
        print("file 2 does not exist.")
        exit(1)

    df1 = pd.read_csv(file_1, sep=',', header=0, index_col=0)
    df2 = pd.read_csv(file_2, sep=',', header=0, index_col=0)

    df = pd.concat([df1, df2], axis=1)
    df = df.reindex(sorted(df.columns), axis=1)

    os.remove(file_1); os.remove(file_2);
    df.to_csv(file_1, sep='\t', na_rep='NaN')
