from Bio import SeqIO
import sys

def split(prefix):
    
    rna_file = f'{prefix}_RNA.fa'
    protein_file = f'{prefix}_PROTEIN.fa'
    is_coding = lambda x: x.startswith('XM') or x.startswith('NM')
    
    pc_storage = []
    nc_storage = []
    for record in SeqIO.parse(rna_file,'fasta'):
        if is_coding(record.id):
            pc_storage.append(record)
        else:
            nc_storage.append(record)
    with open(f'{prefix}_coding_RNA.fa','w') as outfile:
        SeqIO.write(pc_storage,outfile,"fasta")
    with open(f'{prefix}_noncoding_RNA.fa','w') as outfile:
        SeqIO.write(nc_storage,outfile,"fasta")

    pc_storage = []
    nc_storage = []
    for record in SeqIO.parse(protein_file,'fasta'):
        if is_coding(record.id):
            pc_storage.append(record)
        else:
            nc_storage.append(record)
    with open(f'{prefix}_coding_PROTEIN.fa','w') as outfile:
        SeqIO.write(pc_storage,outfile,"fasta")
    with open(f'{prefix}_noncoding_PROTEIN.fa','w') as outfile:
        SeqIO.write(nc_storage,outfile,"fasta")

if __name__ == "__main__":

    split(sys.argv[1])
