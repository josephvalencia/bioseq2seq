from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import numpy as np
import sys

def sample_by_class(balanced_file):

    prefix = balanced_file.split('_balanced.csv')[0]
    df = pd.read_csv(balanced_file,sep='\t')
    df = df.set_index('ID')
    by_type = df.groupby('Type')['RNA'].count()
    print(by_type)
    
    pc = df[df['Type'] == '<PC>']
    nc = df[df['Type'] == '<NC>']
    to_fasta(pc,'RNA',f'{prefix}_PC')
    to_fasta(nc,'RNA',f'{prefix}_NC')

def sample(df,n_samples):
    id_list = np.random.choice(df.index.values,size=n_samples,replace=False)
    return df.loc[id_list]

def to_fasta(df,seq_type,name):

    ids = df.index.tolist()
    sequences  = df[seq_type].tolist()
    records = [make_record(name,seq) for name,seq in zip(ids,sequences)]
    SeqIO.write(records, name+"_"+seq_type+".fa", "fasta")

def make_record(id,rna):
    return SeqRecord(Seq(rna),id=id)

if __name__ == "__main__":

    sample_by_class(sys.argv[1])
