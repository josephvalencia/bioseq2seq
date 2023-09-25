import sys
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def make_record(id,rna):
    return SeqRecord(Seq(rna),id=id,description='')


def convert(rna_filename,prot_filename):
    '''Write to csv for compatibility with eval_translation scripts'''
    storage = []
    proteins = { record.id : str(record.seq) for record in SeqIO.parse(prot_filename,'fasta')}
    for record in SeqIO.parse(rna_filename,'fasta'):
        if record.id in proteins: 
            entry = {'ID' : record.id, 'RNA' : str(record.seq), 'CDS' : '-1', 'Type' : '<PC>', 'Protein' : proteins[record.id]}
            storage.append(entry)

    df = pd.DataFrame(storage)
    df.to_csv('data/lnc_PEP.csv',sep='\t')

if __name__ == "__main__":

    convert(sys.argv[1],sys.argv[2])
