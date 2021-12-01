import pandas as pd
import sys,re
import seaborn as sns
import matplotlib.pyplot as plt

def parse(f):

    df = pd.read_csv(f,sep='\t')
    #df['confidence'] = ['curated' if x.startswith('N') else 'predicted' for x  in df['ID'].tolist() ]
    #print(df.groupby(['confidence','Type']).count())
    #quit()

    df['RNA_len'] = [len(x) for x in df['RNA'].tolist()]
    df = df[(df['RNA_len'] < 1200) & (df['RNA_len'] > 200)]
    print(df.groupby('Type').count())
    '''
    cds_lens = []
    for r,p,t in zip(df['RNA'].tolist(),df['Protein'].tolist(),df['Type'].tolist()):
        if t == "<PC>":
            cds_lens.append(len(p)*3)
        else:
            cds_lens.append(getLongestORFLength(r))
    df['CDS_len'] = cds_lens
    
    df['coverage'] = df['CDS_len'] / df['RNA_len'] 
    print('Past coverage calculation')

    lnc = df[df['Type'] == '<NC>']
    eighty = lnc['RNA_len'].quantile(0.80)
    
    percentiles = [df['RNA_len'].quantile(0.1*x) for x in range(1,11)]
    print("percentiles")
    for k,v in zip(range(1,11),percentiles):
        cum_count = df[df['RNA_len'] < v]
        print(k*0.1,v,len(cum_count))

    prospective = df[(df['RNA_len'] < 2000) & (df['RNA_len'] > 200)]
    print("#200-2000bp",len(prospective))

    print(f'Transcript length (80%) = {eighty}')
    
    by_type = df[['RNA_len','Type']].groupby('Type').median()
    print(by_type)
    
    g1 = sns.histplot(data=df,x='RNA_len',hue='Type',binrange=(200,eighty))
    plt.savefig('transcript_len_95.svg')
    plt.close()

    g2 = sns.histplot(data=df,x='RNA_len',hue='Type',binrange=(200,1200))
    plt.savefig('transcript_len_1200bp.svg')
    plt.close()
    
    g3 = sns.histplot(data=df,x='coverage',hue='Type')
    plt.savefig('coverage_ALL.svg')
    plt.close()

    ninetyfive = lnc['CDS_len'].quantile(0.95)
    print(f'CDS length (95%) = {ninetyfive}')
    g4 = sns.histplot(data=df,x='CDS_len',hue='Type',binrange=(0,ninetyfive))
    plt.savefig('CDS_len_95.svg')
    plt.close()
    '''

def getLongestORFLength(mRNA):
    
    ORF_start = -1
    ORF_end = -1
    longestORF = 0
    for startMatch in re.finditer('ATG',mRNA):
        remaining = mRNA[startMatch.start():]
        if re.search('TGA|TAG|TAA',remaining):
            for stopMatch in re.finditer('TGA|TAG|TAA',remaining):
                ORF = remaining[0:stopMatch.end()]
                if len(ORF) % 3 == 0:
                    if len(ORF) > longestORF:
                        ORF_start = startMatch.start()
                        ORF_end = startMatch.start()+stopMatch.end()
                        longestORF = len(ORF)
                    break
    
    return longestORF

if __name__ == "__main__":

    data = sys.argv[1]
    parse(data)
