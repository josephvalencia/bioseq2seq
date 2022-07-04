from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import numpy as np

def sample_by_class():

    df = pd.read_csv("data/mammalian_1k_train.csv",sep='\t')
    df = df.set_index('ID')
    by_type = df.groupby('Type')['RNA'].count()
    print(by_type)
    pc = df[df['Type'] == '<PC>']
    nc = df[df['Type'] == '<NC>']
    pc_reduced = sample(pc,22000)
    nc_reduced = sample(nc,22000)
    combined = pd.concat([pc_reduced,nc_reduced])
    combined = combined.sample(frac=1)

    to_fasta(pc_reduced,'RNA','mammalian_rnasamba_train_PC')
    to_fasta(nc_reduced,'RNA','mammalian_rnasamba_train_NC')
    to_fasta(combined,'RNA','mammalian_rnasamba_train_ALL')


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

    sample_by_class()
