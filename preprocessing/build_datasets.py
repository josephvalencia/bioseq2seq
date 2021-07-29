import os,sys,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from bioseq2seq.bin.batcher import train_test_val_split,filter_by_length
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def short_mammalian_partitions(combined_file):
    
    random_seed = 65
    data = pd.read_csv(combined_file,sep='\t')
    # filter to <1000 nt and 80/10/10 split
    train,test,val = train_test_val_split(data,1000,random_seed,splits=[0.8,0.1,0.1])
   
    # save to CSV and FASTA formats 
    train.to_csv('mammalian_1k_train.csv',sep='\t',index=False)
    to_fasta(train,'mammalian_1k_train_RNA.fa')
    
    test.to_csv('mammalian_1k_test.csv',sep='\t',index=False)
    to_fasta(test,'mammalian_1k_test_RNA.fa')
    #reduce_redundancy('mammalian_1k_train_RNA.fa','mammalian_1k_test')

    val.to_csv('mammalian_1k_val.csv',sep='\t',index=False)
    to_fasta(val,'mammalian_1k_val_RNA.fa')
    reduce_redundancy('mammalian_1k_train_RNA.fa','mammalian_1k_val')

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

    cmd = f'cd-hit-est-2d -i {train_fa} -i2 {eval_fa} -c 0.80 -n 5 -M 16000 -T 8 -o {reduced_fa}'
    os.system(cmd) 
    filtered = parse_nonredundant_transcripts(reduced_fa)
    
    df = pd.read_csv(eval_csv,sep='\t').set_index('ID')
    df_reduced = df.loc[filtered]
    df_reduced = df_reduced.reset_index()
    df_reduced.to_csv(reduced_csv,sep='\t',index=False)

if __name__ == "__main__":
   
    mammalian_file = 'data/mammalian_refseq.csv' 
    zebrafish_file = 'data/zebrafish_refseq.csv' 
    
    #short_mammalian_partitions(mammalian_file) 
    longer_mammalian(mammalian_file)
    short_zebrafish(zebrafish_file)
