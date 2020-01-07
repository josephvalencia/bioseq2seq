from Bio import SeqIO
from Bio.Alphabet import IUPAC
import sys, os
import matplotlib.pyplot as plt
import pandas as pd

class TranslationTable:

    class Entry:

        def __init__(self,rna="",protein=""):

            self.rna = rna
            self.protein = protein

    def __init__(self):

        self.table = {}

    def add_RNA(self,seq_record):
 
        id_fields = seq_record.id.split("|")

        tscript_name = [x for x in id_fields if x.startswith("ENST")][0]

        if tscript_name not in self.table:

            new_entry = self.Entry()

            new_entry.rna = seq_record.seq

            self.table[tscript_name] = new_entry

    def add_protein(self,seq_record):

        id_fields = seq_record.id.split("|")

        tscript_name = [x for x in id_fields if x.startswith("ENST")][0]
        if tscript_name in self.table:
            entry = self.table[tscript_name]

            entry.protein = seq_record.seq

            self.table[tscript_name] = entry

    def add_entry(self,name,transcript,protein):

        entry = self.Entry()

        entry.rna = transcript
        entry.protein = protein

        self.table[name] = entry

    def linearize(self):

        return [ (k,v.rna,v.protein) for k,v in self.table.items()]

    def to_csv(self,translation_file):

        linear = self.linearize()

        df = pd.DataFrame(linear,columns = ['ID','RNA','Protein'])
        df = df.set_index('ID')

        df.to_csv(translation_file)

def dataset_from_fasta(mRNA,protein,translation_file):

    translation = TranslationTable()

    for seq_record in SeqIO.parse(mRNA,"fasta"):

        translation.add_RNA(seq_record)

    for seq_record in SeqIO.parse(protein,"fasta"):

        translation.add_protein(seq_record)

    translation.to_csv(translation_file)

if __name__ =="__main__":

    mRNA_file = sys.argv[1]
    protein_file = sys.argv[2]
    translation_file = sys.argv[3]
    dataset = dataset_from_fasta(mRNA_file,protein_file,translation_file)
