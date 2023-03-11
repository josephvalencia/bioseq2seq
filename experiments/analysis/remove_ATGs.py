import sys
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

def remove_ATGs_all_fasta(fasta_file,num_shuffles=1):

    storage = []
    for record in SeqIO.parse(fasta_file,'fasta'):
        for i in range(num_shuffles):
            src = str(record.seq)
            src = src.replace('ATG','ATT')
            tscript = record.id
            variant_name=f'{tscript}-dinuc_shuffled-{i+1}'
            record = SeqRecord(Seq(src),
                                id=variant_name)
            storage.append(record)
   
    fields = fasta_file.split('.')
    saved_name = ''.join(fields[:-1]) +'_no_ATGs.fa'
    with open(saved_name,'w') as outFile:
        SeqIO.write(storage, outFile, "fasta")

if __name__ == '__main__':

    remove_ATGs_all_fasta(sys.argv[1])
