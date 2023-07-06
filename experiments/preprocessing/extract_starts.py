import pandas as pd
from Bio import SeqIO
import sys

def extract(csv):

    df = pd.read_csv(csv,sep='\t').set_index('ID')
    df['RNA_len'] = [len(x) for x in df['RNA'].tolist()]
    starts = [x.split(':')[0] for x in df['CDS'].tolist()]
    lens = df['RNA_len'].tolist()

    bad_count = 0
    fname = csv.replace('.csv','_starts.txt')
    with open(fname,'w') as outFile:
        for s,l in zip(starts,lens):
            if s != "-1":
                if s == "<0":
                    bad_count +=1
                    outFile.write(f'{l-1}\n')
                else:
                    outFile.write(f'{s}\n')
            else:
                outFile.write(f'{l-1}\n')

    print(f'Start locations saved at {fname}, {bad_count} bad inputs')

if __name__ == "__main__":

    extract(sys.argv[1])
