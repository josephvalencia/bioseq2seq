import pandas as pd
from scipy.stats import pearsonr,spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from utils import setup_fonts

setup_fonts()
plt.figure(figsize=(3.5,3.5))
ism = pd.read_csv('ISM_codon_deltas.csv')
mdig = pd.read_csv('MDIG_codon_deltas_test.csv')
total = ism.merge(mdig,on='long_mutation',suffixes=['_ism','_mdig'])

ism_deltas = total['Mean_ism']
mdig_deltas = total['Mean_mdig']
pearson = pearsonr(ism_deltas,mdig_deltas)[0]
spearman = spearmanr(ism_deltas,mdig_deltas)[0]

g = sns.regplot(data=total,x='Mean_mdig',y='Mean_ism')
plt.xlabel(r'Mean $\Delta$S-MDIG synonymous mutation',fontsize=8) 
plt.ylabel(r'Mean $\Delta$S ISM synonymous mutation',fontsize=8) 
full = f'r={pearson:.3f}\n'+r'$\rho$'+f'={spearman:.3f}'
print(full)
plt.text(0.05, 0.9,full,
        transform=g.transAxes)
#plt.text(0.05, 0.9,full,
#        transform=g.ax_joint.transAxes)
sns.despine()
plt.tight_layout()
plt.savefig('ism_mdig_codon_correlation.svg')
plt.close()
