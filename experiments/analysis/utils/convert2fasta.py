import sys
import pandas as pd
#from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition,train_test_val_split , filter_by_length
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

'''
def convert(input):

    if input.endswith(".gz"):
        dataframe = pd.read_csv(input,sep="\t",compression = "gzip")
    else:
        dataframe = pd.read_csv(input,sep="\t")

    # obtain splits. Default 80/10/10. Filter below max_len_transcript
    df_train,df_test,df_val = train_test_val_split(dataframe,1000,65)
    # convert to torchtext.Dataset

    to_fasta(df_train,"RNA","train")
    to_fasta(df_test,"RNA","test")
    to_fasta(df_val,"RNA","dev")
'''

def to_fasta(df,seq_type,name):

    ids = df['ID'].tolist()
    sequences  = df[seq_type].tolist()
    records = [make_record(name,seq) for name,seq in zip(ids,sequences)]
    
    SeqIO.write(records, name+"_"+seq_type+".fa", "fasta")


def to_fasta_by_class(df,seq_type,name):

    for c in ['PC','NC']:
        pc_type = f'<{c}>'
        df_by_class = df[df['Type'] == pc_type]
        ids = df_by_class['ID'].tolist()
        sequences  = df_by_class[seq_type].tolist()
        records = [make_record(name,seq) for name,seq in zip(ids,sequences)]
        filename = f'{name}_{seq_type}_{c}.fa'
        SeqIO.write(records,filename, "fasta")

def make_record(id,rna):

    return SeqRecord(Seq(rna),id=id)

if __name__ == "__main__":

    #convert(sys.argv[1])
    df = pd.read_csv('mammalian_1k_train.csv',sep='\t')
    to_fasta_by_class(df,'RNA','train_200-2000')

