import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def logit(x):
    
    if x == 1:
        return 20
    elif x ==0:
        return -20
    else:
        return np.log(x/(1-x))

def fold_change(ribo,rna):
    
    eps = 0.01
    ribo_pseudo = ribo+eps
    rna_pseudo = rna+eps
    return np.log2(ribo_pseudo/rna_pseudo)

score_file = sys.argv[1]
te_file = sys.argv[2]
mapping_file = sys.argv[3]

df_score = pd.read_csv(score_file,sep="\t")
df_score['coding_prob'] = [logit(np.exp(x)) for x in df_score['coding_prob'].values.tolist()]

df_te = pd.read_csv(te_file,sep="\t")
df_te.columns = ["ensembl_transcript_id_version" if x == "transcript_id" else x for x in df_te.columns]

limited = df_te[(df_te['transcript_type'] == "protein_coding" ) | (df_te['transcript_type'] == "lncRNA")].copy()
#limited = limited[limited['te'] != 0.0]
#limited['te_log10'] = [np.log10(x) for x in limited['te'].values.tolist()]

sns.scatterplot(data=limited,x="rna_rpkm",y="ribo_rpkm",hue="transcript,size=[0.5]*len(limited))

#sns.displot(data=limited,x="te",hue="coding",kind="kde")
plt.savefig("scatter.pdf")
plt.close()

#eps = 1e-32
#df_te['te'] = [-np.log(x+eps) for x in df_te['te'].values.tolist()]
#df_te = df_te[df_te['rna_rpkm'] != 0.0]
'''
ribo = df_te['ribo_rpkm'].values.tolist()
rna_seq = df_te['rna_rpkm'].values.tolist()

df_te['te_alt'] = [fold_change(x,y) for x,y in zip(ribo,rna_seq)]
df_mapping = pd.read_csv(mapping_file,sep="\t")
df_te_refseq = limited.merge(df_mapping,on="ensembl_transcript_id_version")
df_all = df_te_refseq.merge(df_score,on="refseq_transcript_id")

sns.pairplot(data=df_all,hue="transcript_type",vars=["te","coding_prob","beam_score"],diag_kind="kde")
plt.savefig("te_coding_correlations.pdf")
plt.suptitle("Pairwise Scatterplots (Human Validation Set")
plt.close()
'''
