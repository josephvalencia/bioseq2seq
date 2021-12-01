import os,sys,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from bioseq2seq.bin.batcher import train_test_val_split,filter_by_length
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict

def mammalian_partitions(combined_file,max_len):
    
    random_seed = 65
    data = pd.read_csv(combined_file,sep='\t')
    # filter to <1000 nt and 80/10/10 split
    train,test,val = train_test_val_split(data,max_len,random_seed,min_len=200,splits=[0.8,0.1,0.1])
    
    # save to CSV and FASTA formats 
    train.to_csv(f'mammalian_200-{max_len}_train.csv',sep='\t',index=False)
    to_fasta(train,f'mammalian_200-{max_len}_train_RNA.fa')
    balanced = balance_class_and_length(train,max_len)     
    balanced.to_csv(f'mammalian_200-{max_len}_train_balanced.csv',sep='\t',index=False)
    to_fasta(balanced,f'mammalian_200-{max_len}_train_RNA_balanced.fa')
    
    test.to_csv(f'mammalian_200-{max_len}_test.csv',sep='\t',index=False)
    to_fasta(test,f'mammalian_200-{max_len}_test_RNA.fa')
    reduce_redundancy(f'mammalian_200-{max_len}_train_RNA.fa',f'mammalian_200-{max_len}_test')

    val.to_csv(f'mammalian_200-{max_len}_val.csv',sep='\t',index=False)
    to_fasta(val,f'mammalian_200-{max_len}_val_RNA.fa')
    reduce_redundancy(f'mammalian_200-{max_len}_train_RNA.fa',f'mammalian_200-{max_len}_val')

def balance_class_and_length(df,max_len):

    random_seed = 65
    df['RNA_len'] = [len(x) for x in df['RNA'].tolist()]
    sns.histplot(data=df,hue='Type',x = 'RNA_len',binwidth=1)
    plt.savefig(f'train_200-{max_len}_RNA_lens_unbalanced.svg')
    plt.close()

    pc = df[df['Type'] == '<PC>']
    nc = df[df['Type'] == '<NC>']

    dataset_nc , seqs_nc = arrange(nc)
    dataset_pc, _ = arrange(pc)

    nc_keys = [len(seq) for name, seq in seqs_nc]
    samples_pc = match_length_distribution(nc_keys,dataset_pc)
    
    pc_ids = [x[0] for x in samples_pc]
    nc_ids = nc['ID'].tolist()
    
    balanced_ids = pc_ids+nc_ids
    df = df.set_index('ID')
    df = df.loc[balanced_ids]
    df = df.reset_index()
    df = df.sample(frac=1.0,random_state=random_seed)
    sns.histplot(data=df,hue='Type',x = 'RNA_len',binwidth=1)
    plt.savefig(f'train_200-{max_len}_RNA_lens_balanced.svg')
    plt.close()
    
    return df[df.columns.difference(['RNA_len'])]

def match_length_distribution(sample_keys,dataset_b):
    #list of keys
    samples_b = []
    for key in sample_keys:
        found_neighbor = False
        sign = 1
        dist = 1
        s_key = key
        while not found_neighbor:
            try:
                samples_b.append(dataset_b[key][0])
                found_neighbor = True
                del dataset_b[key][0]
                if len(dataset_b[key]) == 0:
                    del dataset_b[key]
            except KeyError:
                key += dist * sign
                dist += 1
                sign *= -1
    return samples_b

def arrange(df):
    seqs = []
    dataset = {}

    for n,r in zip(df['ID'].tolist(),df['RNA'].tolist()):
        key = len(r)
        if key not in dataset:
            dataset[key] = []
        dataset[key].append((n,r))
        seqs.append((n,r))
    
    return dataset,seqs


def longer_mammalian(combined_file):

    #data = pd.read_csv(combined_file,sep="\t")
    # 1000-2000 nt
    #longer = filter_by_length(data,2000,min_len=1000)
    #longer.to_csv('mammalian_1k-2k.csv',sep='\t',index=False)
    #to_fasta(longer,'mammalian_1k-2k_RNA.fa')
    reduce_redundancy('data/mammalian_1k_train_RNA.fa','data/mammalian_1k-2k')
    
def to_fasta(df,name):

    ids = df['ID'].tolist()
    sequences  = df['RNA'].tolist()
    records = [make_record(name,seq) for name,seq in zip(ids,sequences)]
    SeqIO.write(records, name, "fasta")

def make_record(id,rna):
    return SeqRecord(Seq(rna),id=id)

def short_zebrafish(zebrafile):

    #data = pd.read_csv(zebrafile,sep='\t')
    #shorter = filter_by_length(data,1000,min_len=0)
    #shorter.to_csv('zebrafish_1k.csv',sep='\t',index=False)
    #to_fasta(shorter,'zebrafish_1k_RNA.fa')
    reduce_redundancy('data/mammalian_1k_train_RNA.fa','data/zebrafish_1k')

def filter_by_length(df,max_len,min_len=0):
    '''Filter dataframe to RNA within (min_len,max_len)'''
    
    df['RNA_LEN'] = [len(x) for x in df['RNA'].values]
    df['Protein_LEN'] = [len(x) for x in df['Protein'].values]

    # filter by maximum length, and optionally minimum length
    df = df[df['RNA_LEN'] < max_len]
    if min_len > 0:
        df =  df[df['RNA_LEN'] > min_len]

    return df[['ID','RNA', 'CDS', 'Type','Protein']]

def train_test_val_split(translation_table,max_len,random_seed,min_len=0,splits=[0.8,0.1,0.1]):
    
    # keep entries with RNA length < max_len
    translation_table = filter_by_length(translation_table,max_len,min_len)
    translation_table = translation_table.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    N = translation_table.shape[0]

    # splits to cumulative percentages
    cumulative = [splits[0]]
    for i in range(1,len(splits) -1):
        cumulative.append(cumulative[i-1]+splits[i])
    split_points = [int(round(x*N)) for x in cumulative]

    # split dataframe at split points
    train,test,val = np.split(translation_table,split_points)
    return train,test,val

def parse_nonredundant_transcripts(filtered_fa):
    '''Ingest FASTA output from CD-HIT'''

    subset = []
    with open(filtered_fa) as inFile:
        for record in SeqIO.parse(inFile,'fasta'):
            subset.append(record.id)
    return subset

def reduce_redundancy(train_fa,eval_prefix):

    eval_fa = eval_prefix+'_RNA.fa'
    reduced_fa = eval_prefix+'_RNA_nonredundant_80.fa'
    eval_csv = eval_prefix+'.csv' 
    reduced_csv = eval_prefix+'_nonredundant_80.csv'

    #cmd = f'cd-hit-est-2d -i {train_fa} -i2 {eval_fa} -c 0.80 -n 5 -M 16000 -T 8 -o {reduced_fa}'
    cmd = f'cd-hit-est-2d -i {train_fa} -i2 {eval_fa} -c 0.80 -n 5 -T 32 -o {reduced_fa}'
    os.system(cmd) 
    filtered = parse_nonredundant_transcripts(reduced_fa)
    
    df = pd.read_csv(eval_csv,sep='\t').set_index('ID')
    df_reduced = df.loc[filtered]
    df_reduced = df_reduced.reset_index()
    df_reduced.to_csv(reduced_csv,sep='\t',index=False)

if __name__ == "__main__":
   
    mammalian_file = 'data/mammalian_refseq.csv' 
    zebrafish_file = 'data/zebrafish_refseq.csv' 
    
    mammalian_partitions(mammalian_file,int(sys.argv[1])) 
    #longer_mammalian(mammalian_file)
    #short_zebrafish(zebrafish_file)
