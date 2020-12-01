import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def fold_change(ribo,rna):
    eps = 0.01
    ribo_pseudo = ribo+eps
    rna_pseudo = rna+eps
    return np.log2(ribo_pseudo/rna_pseudo)

lncRNA = []
mRNA = []

with open(sys.argv[1]) as inFile:

    for l in inFile.readlines()[1:]:

        fields = l.split("\t")
        
        tscript_type = fields[3]
        te_value = float(fields[4])
        ribo = float(fields[5])
        rna_seq = float(fields[6])

        if rna_seq > 0:
            alt_te = fold_change(ribo,rna_seq) 

            if tscript_type == "protein_coding":
                mRNA.append(alt_te)
            elif tscript_type == "lncRNA":
                lncRNA.append(alt_te)

bin_size = 0.1
bins = np.arange(min(lncRNA+mRNA),max(lncRNA+mRNA)+bin_size,bin_size)

plt.hist(mRNA,label="mRNA",density=True,alpha=0.5,bins=bins)
plt.hist(lncRNA,label="lncRNA",density=True,alpha=0.5,bins=bins)
plt.legend()
plt.savefig("te_test_distribution.pdf")
plt.close()


