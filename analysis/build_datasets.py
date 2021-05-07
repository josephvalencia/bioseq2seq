import pandas as pd
import sys,re
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from bioseq2seq.bin.batcher import train_test_val_split,filter_by_length
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def make_partitions(combined_file):
    
    random_seed = 65
    data = pd.read_csv(combined_file,sep='\t')
    
    train,test,val = train_test_val_split(data,1000,random_seed,splits=[0.8,0.1,0.1])
    test.to_csv('test.csv',sep='\t')
    val.to_csv('val.csv',sep='\t')
    train.to_csv('train.csv',sep='\t')

def longer_mammalian(combined_file):

    random_seed = 65
    data = pd.read_csv(combined_file,sep="\t")
    longer = filter_by_length(data,2000,min_len=1000)
    longer.to_csv('mammalian_1k_to_2k.csv',sep='\t')
    to_fasta(longer,'RNA','mammalian_1k_to_2k')
    
def to_fasta(df,seq_type,name):

    ids = df['ID'].tolist()
    sequences  = df[seq_type].tolist()
    records = [make_record(name,seq) for name,seq in zip(ids,sequences)]
    SeqIO.write(records, name+"_"+seq_type+".fa", "fasta")

def make_record(id,rna):
    return SeqRecord(Seq(rna),id=id)

def zebrafish(zebrafile):

    data = pd.read_csv(zebrafile,sep='\t')
    shorter = filter_by_length(data,1000,min_len=0)
    shorter.to_csv('zebrafish_1k.csv',sep='\t')
    to_fasta(shorter,'RNA','zebrafish_1k_to_2k')

def filter_by_length(df,max_len,min_len=0):
    
    '''Filter dataframe to RNA within (min_len,max_len)'''
    
    df['RNA_LEN'] = [len(x) for x in df['RNA'].values]
    df['Protein_LEN'] = [len(x) for x in df['Protein'].values]

    percentiles = [0.1 * x for x in range(1,10)]
    df = df[df['RNA_LEN'] < max_len]

    if min_len > 0:
        df =  df[df['RNA_LEN'] > min_len]
 
    distribution_file = 'length_hist.svg'
    sns.histplot(data=df,x='RNA_LEN',hue='Type')
    plt.savefig(distribution_file)
    plt.close()
    
    by_type = df.groupby('Type').count()
    print(by_type)

    print("total number =",len(df)) 
    return df[['ID','RNA', 'CDS', 'Type','Protein']]

if __name__ == "__main__":

    #make_partitions(sys.argv[1]) 
    #longer_mammalian(sys.argv[1])
    zebrafish(sys.argv[1])
