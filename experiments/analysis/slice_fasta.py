from Bio import SeqIO
import sys
from random import sample, shuffle

def parse(fname,outname,only_verified=True,n=None):

    if 'RNA' in fname:
        prot_file = fname.replace('RNA','PROTEIN')
        rna_file = fname
    elif 'PROTEIN' in fname:
        rna_file = fname.replace('PROTEIN','RNA')
        prot_file = fname

    rna_dict =  SeqIO.to_dict(SeqIO.parse(rna_file,'fasta'))
    prot_dict =  SeqIO.to_dict(SeqIO.parse(prot_file,'fasta'))

    is_verified = lambda x : True if x.startswith('N') else False
    is_mrna = lambda x : True if x.startswith('XM') or x.startswith('NM') else False
    coding = [x for x in rna_dict.keys() if is_mrna(x)]
    noncoding = [x for x in rna_dict.keys() if not is_mrna(x)]
    
    if only_verified:
        coding = [x for x in coding if is_verified(x)]
        noncoding = [x for x in noncoding if is_verified(x)]
    
    n = min(len(coding),len(noncoding)) if n is None else n
    print(f"sampling {n} each of mRNA/lncRNA")
    coding = sample(coding,n)
    noncoding = sample(noncoding,0)

    combined = [(rna_dict[c],prot_dict[c]) for c in coding+noncoding]
    shuffle(combined)
    rna = [x[0] for x in combined]
    protein = [x[1] for x in combined]
    
    with open(f'{outname}_PROTEIN.fa','w') as outfile:
        SeqIO.write(protein,outfile,"fasta")
    with open(f'{outname}_RNA.fa','w') as outfile:
        SeqIO.write(rna,outfile,"fasta")

if __name__ == "__main__":

    parse(sys.argv[1],sys.argv[2])
