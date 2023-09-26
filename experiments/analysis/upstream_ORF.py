import sys
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
from utils import get_CDS_loc, getLongestORF
import pandas as pd
import numpy as np
import random

def has_uORF(fasta_file,test_csv):

    test_df = pd.read_csv(test_csv,sep='\t').set_index('ID')
    is_coding = lambda x : x.startswith('XM') or x.startswith('NM')
    storage = []
    count = 0
    mrna = 0
    orf_lens = []
    five_primes = []
    remainders = []
    for record in SeqIO.parse(fasta_file,'fasta'):
        src = str(record.seq)
        tscript = record.id
        if is_coding(tscript):
            cds = test_df.loc[tscript,'CDS']
            start,end = get_CDS_loc(cds,src)
            fiveprime = src[:start]
            upstream_ORF = getLongestORF(fiveprime)
            length = upstream_ORF[1] - upstream_ORF[0]
            
            # save 5' UTRs with a uORF and downstream without 
            if upstream_ORF != (-1,-1) and length >=9:
                count+=1
                orf_lens.append(length)
                five_primes.append(fiveprime)
            else:
                remainders.append((src[start:],tscript))
            mrna+=1
    print(f'{count}/{mrna} = {100*count/mrna:.1f}% have a uORF')
    
    q = [10*x for x in range(11)] 
    percents = np.percentile(orf_lens,q).tolist() 
    print(list(zip(q,percents)))
    print(len(five_primes),len(remainders)) 

    synthetic = []
    for r,tscript in remainders:
        fives = random.sample(five_primes,5)
        for i,f in enumerate(fives):
            record = SeqRecord(Seq(f+r),
                                id=f'{tscript}-5-prime-{i}')
            synthetic.append(record)

    with open('data/swapped_uORFs.fa','w') as outFile:
        SeqIO.write(synthetic, outFile, "fasta")
    
    print('saved data/swapped_uORFs.fa')

if __name__ == '__main__':
    has_uORF(sys.argv[1],sys.argv[2])
