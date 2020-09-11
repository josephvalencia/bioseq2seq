from Bio import SeqIO
from collections import Counter
import gzip

def countkmers(substrings_file):

    counter = Counter()

    if substrings_file.endswith(".gz"):
        inFile = gzip.open(substrings_file,'rt')
    else:
        inFile = open(substrings_file)

    for record in SeqIO.parse(inFile,"fasta"):
        substring = record.seq
        codons = [substring[i:i+3] for i in range(len(substring)-3)]
        counter.update(codons)
    inFile.close()

    counts = counter.most_common(64)
    top_file = substrings_file+".codon_counts"
    total_codons = sum(x[1] for x in counts)
    with open(top_file,'w') as outFile:
        for substr, count in counts:
            outFile.write("{},{}\n".format(substr,count/total_codons))


if __name__ == "__main__":

    #countkmers("../Fa/refseq/homo_sapiens/GCF_000001405.39_GRCh38.p13_rna.fna.gz")

    for h in range(8):
        head = "medium/layer3head"+str(h)+"_attn_topk_idx_motifs.fasta"
        countkmers(head)
    