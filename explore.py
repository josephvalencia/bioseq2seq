import pandas as pd
import sys
import matplotlib.pyplot as plt

def lengths(db):

    df = pd.read_csv(db)
    df = df[['antisense ncRNA', 'Target RNA', 'Type', 'Species','Target Interaction Region Start',
        'Target Interaction Region End','antisense ncRNA sequence', 'target RNA sequence']]
    df = df.dropna()
    
    df['source length'] = df['antisense ncRNA sequence'].str.len()
    df['target length'] = df['target RNA sequence'].str.len()
    df['total length'] = df['source length'] + df['target length']
    df = df[df['total length'] < 500]
    print(df)
    df.hist(column='total length')
    plt.savefig('length_distribution.png')

if __name__ =="__main__":

    lengths(sys.argv[1])
