import pandas as pd
import sys,re
import matplotlib.pyplot as plt
import seaborn as sns
from bioseq2seq.bin.batcher import train_test_val_split
import numpy as np
from Bio import motifs
from Bio.Seq import Seq

def find_motif_locations(fimo_output,by='start',stat='count'):

    plt.figure(figsize=(20,20))

    # ingest stored data
    df_val = pd.read_csv('../Fa/test.csv',sep="\t")
    df_val = df_val.set_index('ID')
    
    subset = []
    sub_file = "output/test/redundancy/test_reduced_80_ids.txt" 
    with open(sub_file) as inFile:
        for l in inFile:
            subset.append(l.rstrip())
    
    df_val = df_val.loc[subset]
    df_val['cds_start'] = [get_CDS_start(cds,seq)[0] for cds,seq in zip(df_val['CDS'].values.tolist(),df_val['RNA'].values.tolist())]
    df_val['cds_end'] = [get_CDS_start(cds,seq)[1] for cds,seq in zip(df_val['CDS'].values.tolist(),df_val['RNA'].values.tolist())]
    
    df_fimo = pd.read_csv(fimo_output,sep='\t')
    df_fimo = df_fimo[~df_fimo['motif_id'].str.startswith('#')]
    df = pd.merge(df_fimo,df_val,right_on='ID',left_on='sequence_name')
    
    if by == 'end':
        df['rel_start'] = df['start'] - df['cds_end'] -1
    else:
        df['rel_start'] = df['start'] - df['cds_start'] -1
    
    starts = df['rel_start'].values.tolist()
    df['frame'] = df['rel_start'] % 3

    df = df.drop(df.columns.difference(['Type','motif_id','rel_start','frame']),1)
    by_type = df.groupby(['motif_id','Type','frame']).count()
    print(by_type)
    
    if by == 'end':
        bins = np.arange(-1000,750,10)
    else:
        bins = np.arange(-750,1000,10)
   
    by_count = df.groupby('motif_id').count().sort_values(by='Type',ascending=False)
    hue_order = by_count.index.values.tolist()
    g = sns.displot(data=df,x='rel_start',hue='motif_id',col='Type',kind='hist',bins=bins,stat=stat,element='step',facet_kws={'col_order':['<PC>','<NC>']})
    g._legend.set_title('IG motif (STREME ID)')

    axes = g.axes.flatten()
    axes[0].set_title("")
    axes[0].set_xlabel("Motif start relative to {} of mRNA CDS".format(by))
    ylab = "Density" if stat == 'density' else 'Count'
    axes[0].set_ylabel(ylab)
    axes[1].set_xlabel("Motif start relative to {} of lncRNA longest ORF".format(by))
    #axes[1].spines['left'].set_visible(False)
    axes[1].set_title("")
    axes[1].set_ylabel("")
    
    name = fimo_output.split('.')[0] + '_{}_pos_{}_hist.svg'.format(by,stat)
    plt.savefig(name)
    plt.close()

def histogram_fn(data, **kws):

    ax = sns.histplot(data,x='rel_start',hue='motif_id',binwidth=10,common_bins=True,element='step')
    ax.set_xlabel("Motif start position relative to CDS")

def get_CDS_start(cds,rna):

    if cds != "-1": 
        splits = cds.split(":")
        clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
        splits = [clean(x) for x in splits]
        start,end = tuple([int(x) for x in splits])
    else:
        start,end = getLongestORF(rna)
    
    return start,end
        
def getLongestORF(mRNA):
    ORF_start = -1
    ORF_end = -1
    longestORF = 0
    for startMatch in re.finditer('ATG',mRNA):
        remaining = mRNA[startMatch.start():]
        if re.search('TGA|TAG|TAA',remaining):
            for stopMatch in re.finditer('TGA|TAG|TAA',remaining):
                ORF = remaining[0:stopMatch.end()]
                if len(ORF) % 3 == 0:
                    if len(ORF) > longestORF:
                        ORF_start = startMatch.start()
                        ORF_end = startMatch.start()+stopMatch.end()
                        longestORF = len(ORF)
                    break
    return ORF_start,ORF_end

def kozak_logo():

    # ingest stored data
    df_val = pd.read_csv('test.csv',sep="\t")
   
    pc = df_val[df_val['Type'] == '<PC>']
    storage = [(get_CDS_start(cds,seq),seq) for cds,seq in zip(pc['CDS'].values.tolist(),pc['RNA'].values.tolist())]

    count = 0
    instances = []

    for start,seq in storage:
        if start >= 10:
            kozak = seq[start-10:start+13]
            instances.append(kozak)

    m = motifs.create(instances)
    m.weblogo('kozak.svg',format='svg',size='large',yaxis=5)
    
if __name__ == "__main__":

    #kozak_logo()
    find_motif_locations(sys.argv[1],by=sys.argv[2],stat=sys.argv[3])
