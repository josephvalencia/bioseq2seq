import pandas as pd
import sys
from collections import Counter

df = pd.read_csv(sys.argv[1],"\t")

coding = df[df["Type"] == "<PC>"]["RNA"]
noncoding = df[df["Type"] == "<NC>"]["RNA"]

coding_counts = Counter()
noncoding_counts = Counter()

pc_total = 0
nc_total = 0

for r in coding:
    chars = [x for x in r]
    coding_counts.update(chars)
    pc_total += len(chars)

for a in noncoding:
    chars = [x for x in a]
    noncoding_counts.update(chars)
    nc_total += len(chars)

print(coding_counts)
print(pc_total)

print(noncoding_counts)
print(nc_total)

#print(aa_counts)
#print(aa_total)



