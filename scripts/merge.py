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

    df = pd.read_csv(file, sep=mysep, header=0, index_col=0)
    dfs.append(df)

M_df = pd.DataFrame()
dic = []

for df in dfs:
    for U, row in df.iterrows():
        for mu in df.columns:
            if row[mu] != nan_value:
                #print("U =", U, ", type =", type(U))
                #print("mu =", mu, ", type =", type(mu))
                #print("row[mu] =", row[mu], ", type =", type(row[mu]))
                M_df.loc[float(U), float(mu)] = float(row[mu])

M_df = M_df.reindex(sorted(M_df.columns), axis=1)     # sorting columns (T values)
M_df.sort_index(inplace=True);     # sorting indexes (U values)

M_df.to_csv('out-merged_df.csv', sep=mysep, index_label=index_lab, na_rep=nan_value, float_format=float_fmt)

# print dataframe
print(M_df)

# print number of missing values
print("number of missing values =", M_df.isna().sum().sum())
