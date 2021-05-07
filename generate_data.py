import sys
import pandas as pd

def make_nonredundant(dataset,nonredundant):

    keep_list = []
    with open(nonredundant) as inFile:
        for l in inFile:
            keep_list.append(l.rstrip())

    df = pd.read_csv(dataset,sep='\t')
    df = df.set_index('ID')
    df = df.loc[keep_list]
    df.to_csv('test_nonredundant_80.csv',sep='\t',index=True)

if __name__ == "__main__":

    make_nonredundant(sys.argv[1],sys.argv[2])
