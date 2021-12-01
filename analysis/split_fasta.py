from Bio import SeqIO
import sys


def split_fasta(fasta):

    coding = []
    noncoding = []

    for r in SeqIO.parse(fasta,'fasta'):
        tscript = r.id
        if tscript.startswith('XR') or tscript.startswith('NR'):
            noncoding.append(r)
        else:
            coding.append(r)

    prefix = fasta.split('.')[0]
    
    with open(prefix+'_NC.fa','w') as outFile:
        SeqIO.write(noncoding,outFile,"fasta")
    with open(prefix+'_PC.fa','w') as outFile:
        SeqIO.write(coding,outFile,"fasta")


if __name__ == "__main__":

    split_fasta(sys.argv[1])
