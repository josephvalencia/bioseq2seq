from Bio import SeqIO
import sys,re
from random import sample

def parse(fname,outname):

    if 'RNA' in fname:
        prot_file = fname.replace('RNA','PROTEIN')
        rna_file = fname
    elif 'PROTEIN' in fname:
        rna_file = fname.replace('PROTEIN','RNA')
        prot_file = fname

    rna_dict =  SeqIO.to_dict(SeqIO.parse(rna_file,'fasta'))
    prot_dict =  SeqIO.to_dict(SeqIO.parse(prot_file,'fasta'))
   
    is_mrna = lambda x : True if x.startswith('XM') or x.startswith('NM') else False
    coding = [x for x in rna_dict.keys() if is_mrna(x)]
    noncoding = [x for x in rna_dict.keys() if not is_mrna(x)]
    
    coding = sample(coding,10)
    noncoding = sample(noncoding,0)

    rna = [rna_dict[c] for c in coding+noncoding]
    protein = [prot_dict[c] for c in coding+noncoding]

    with open(f'{outname}_PROTEIN.fa','w') as outfile:
        SeqIO.write(protein,outfile,"fasta")
    with open(f'{outname}_RNA.fa','w') as outfile:
        SeqIO.write(rna,outfile,"fasta")

if __name__ == "__main__":

    parse(sys.argv[1],sys.argv[2])
