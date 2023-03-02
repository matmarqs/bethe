#!/usr/bin/env python3
import pandas as pd
import sys, os
#import numpy as np

dfs = []
mysep = ','
float_fmt = '%15.8e'
index_lab = "     U/mu      "
nan_value = "      NaN      "

for file in sys.argv[1:]:
    if not os.path.exists(file):
        print(f"file f{file} does not exist.")
        exit(1)

    df = pd.read_csv(file, sep=mysep, header=0, index_col=0, na_values=nan_value)
    dfs.append(df)

M_df = dfs[0]

for df in dfs[1:]:
    M_df += df

M_df = M_df / len(dfs)

M_df = M_df.reindex(sorted(M_df.columns), axis=1)     # sorting columns (T values)
M_df.sort_index(inplace=True);     # sorting indexes (U values)

M_df.to_csv('mean_df.csv', sep=mysep, index_label=index_lab, na_rep=nan_value, float_format=float_fmt)

# print dataframe
print(M_df)
