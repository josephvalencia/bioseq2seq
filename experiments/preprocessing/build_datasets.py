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
from ushuffle import shuffle, Shuffler

def mammalian_partitions(combined_file,max_len):
    
    random_seed = 65
    data = pd.read_csv(combined_file,sep='\t')
    
    # filter to 200 <=x<= max_len  nt and 80/10/10 split
    train,test,val = train_test_val_split(data,max_len,random_seed,min_len=200,splits=[0.8,0.1,0.1])
    prefix = f'mammalian_200-{max_len}'
    train.to_csv(f'{prefix}_train.csv',sep='\t',index=False)
   
    # balance mRNA and lncRNA
    balanced = balance_class_and_length(train,max_len)     
    balanced.to_csv(f'{prefix}_train_balanced.csv',sep='\t',index=False)
    to_fasta(balanced,f'{prefix}_train_RNA_balanced.fa',column='RNA')
    to_fasta(balanced,f'{prefix}_train_PROTEIN_balanced.fa',column='Protein')
    to_fasta(balanced,f'{prefix}_train_CLASS_balanced.fa',column='Type')
   
    # save unreduced then remove tscripts >80% alignment
    test.to_csv(f'{prefix}_test.csv',sep='\t',index=False)
    to_fasta(test,f'{prefix}_test_RNA.fa',column='RNA')
    reduce_redundancy(f'{prefix}_train_RNA_balanced.fa',f'{prefix}_test')

    # save unreduced then remove tscripts >80% alignment
    val.to_csv(f'{prefix}_val.csv',sep='\t',index=False)
    to_fasta(val,f'{prefix}_val_RNA.fa',column='RNA')
    reduce_redundancy(f'{prefix}_train_RNA_balanced.fa',f'{prefix}_val')

def balance_class_and_length(df,max_len):

    random_seed = 65
    df['RNA_len'] = [len(x) for x in df['RNA'].tolist()]
    sns.histplot(data=df,hue='Type',x = 'RNA_len',binwidth=1)
    plt.savefig(f'train_200-{max_len}_RNA_lens_unbalanced.svg')
    plt.close()
    
    print(df.groupby('Type').count())
    pc = df[df['Type'] == '<PC>']
    nc = df[df['Type'] == '<NC>']
    print(f'# pc unbalanced = {len(pc)}, # nc unbalanced = {len(nc)}')
    dataset_nc, seqs_nc = arrange(nc)
    dataset_pc, seqs_pc = arrange(pc)
    nc_keys = [len(seq) for name, seq in seqs_nc]
    samples_pc = match_length_distribution(nc_keys,dataset_pc)
    pc_ids = [x[0] for x in samples_pc]
    nc_ids = nc['ID'].tolist()
    N = len(pc_ids)
    print(f'len(pc_ids) = {N} , len(nc_ids) = {len(nc_ids)}, diff = {len(pc)-N}')
    
    balanced_ids = pc_ids+nc_ids
    df = df.set_index('ID')
    df = df.loc[balanced_ids]
    
    df = df.reset_index()
    df = df.sample(frac=1.0,random_state=random_seed)

    sns.histplot(data=df,hue='Type',x = 'RNA_len',binwidth=1)
    plt.savefig(f'train_200-{max_len}_RNA_lens_balanced.svg')
    plt.close()
    
    return df[df.columns.difference(['RNA_len'])]

def generate_shuffled_seqs(df):

    ids = df['ID'].tolist()
    cds = df['CDS'].tolist()
    rna = df['RNA'].tolist()
    protein = df['Protein'].tolist()
    types = df['Type'].tolist()

    ids = ['XR_shuffled_'+t for t in ids]
    cds = [-1 for c in cds]
    protein = ['[NONE]' for p in protein]
    rna = [str(shuffle(r.encode('utf-8'),2)) for r in rna]
    types = ['<NC>' for t in types]

    data = {'ID' : ids , 'RNA' : rna , 'CDS' : cds , 'Type' : types , 'Protein' : protein}
    return pd.DataFrame(data=data)


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
    
def to_fasta(df,name,column='RNA'):

    ids = df['ID'].tolist()
    sequences  = df[column].tolist()
    records = [make_record(name,seq) for name,seq in zip(ids,sequences)]
    print(f'Writing {name}')
    SeqIO.write(records, name, "fasta")


def make_record(id,rna):
    return SeqRecord(Seq(rna),id=id,description='')

def filter_by_length(df,max_len,min_len=0):
    '''Filter dataframe to RNA within (min_len,max_len)'''
    
    df['RNA_LEN'] = [len(x) for x in df['RNA'].values]
    df['Protein_LEN'] = [len(x) for x in df['Protein'].values]

    # filter by maximum length, and optionally minimum length
    df = df[df['RNA_LEN'] <= max_len]
    if min_len > 0:
        df =  df[df['RNA_LEN'] >= min_len]

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
    eval_csv = eval_prefix+'.csv' 
    
    reduced_csv = eval_prefix+'_nonredundant_80.csv'
    reduced_rna_fa = eval_prefix+'_RNA_nonredundant_80.fa'
    reduced_protein_fa = eval_prefix+'_PROTEIN_nonredundant_80.fa'
    reduced_class_fa = eval_prefix+'_CLASS_nonredundant_80.fa'

    cmd = f'cd-hit-est-2d -i {train_fa} -i2 {eval_fa} -c 0.80 -n 5 -M 16000 -T 8 -o {reduced_rna_fa}'
    os.system(cmd) 
    filtered = parse_nonredundant_transcripts(reduced_rna_fa)
    
    df = pd.read_csv(eval_csv,sep='\t').set_index('ID')
    df_reduced = df.loc[filtered]
    df_reduced = df_reduced.reset_index()
    df_reduced.to_csv(reduced_csv,sep='\t',index=False)
    
    to_fasta(df_reduced,reduced_protein_fa,column='Protein')
    to_fasta(df_reduced,reduced_class_fa,column='Type')

if __name__ == "__main__":
   
    mammalian_file = 'mammalian_refseq.csv' 
    mammalian_partitions(mammalian_file,int(sys.argv[1])) 
