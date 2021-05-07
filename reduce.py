import pandas as pd
import sys

subset = []
with open(sys.argv[1]) as inFile:
    for l in inFile:
        subset.append(l.rstrip())

df = pd.read_csv(sys.argv[2],sep='\t')
df = df.set_index('ID')
df = df.loc[subset]
df.to_csv('mammalian_1k_to_2k_reduced_80.csv',sep='\t')
print(df)
