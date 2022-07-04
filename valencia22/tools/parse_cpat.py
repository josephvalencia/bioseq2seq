import pandas as pd


def parse_cpat():

    df = pd.read_csv('train_cpat.ORF_prob.best.tsv',sep='\t')
    original_columns = ['seq_ID','mRNA','ORF','Fickett','Hexamer']
    new_columns = ['ID','mRNA','ORF','Fickett','Hexamer']
    df = df[original_columns]
    df.columns = new_columns 
    is_coding = lambda x : 1 if x.startswith('NM') or x.startswith('XM') else 0
    df['Label'] = [is_coding(x) for x in df['ID'].tolist()]
    df.to_csv('mammalian.dta',sep='\t',index=False)

if __name__ == "__main__":
    parse_cpat()
