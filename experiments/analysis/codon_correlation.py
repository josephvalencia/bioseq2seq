import pandas as pd
from scipy.stats import pearsonr,spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from utils import setup_fonts,parse_config,build_output_dir
import re

def parse_entry(entry):
    if any([c == ' ' for c in entry]):
        return None
    else:
        return entry[0], float(entry[1])

def build_codon_table(filename):
   
    raw_frequencies = {}
    search_string = re.search('(.*)/(.*)_codon_table.txt',filename)
    if search_string is not None:
        species = search_string.group(2)
    else:
        species = filename

    with open(filename,'r') as inFile:
        fields = inFile.read().rstrip().replace('\n','\t').split('\t')
    
    for i in range(0,len(fields),3):
        entry = fields[i:i+3]
        result = parse_entry(entry)
        if result:
            raw_frequencies[result[0]] = result[1]
    return species,raw_frequencies

def tai_codon_corr(df,metric):
    
    tai_df = pd.read_csv('human_TAI_Tuller.csv')
    tai_dict = {x : y for x,y in zip(tai_df['Codon'],tai_df['Human'])} 
    
    _,human_usage = build_codon_table('data/codon_usage_tables/homo_sapiens_codon_table.txt')
    tai_deltas = []
    usage_deltas = []
    for long_mutation,mean in zip(df['long_mutation'],df['Mean']):
        src,dest = tuple(long_mutation.split('>'))
        delta_tai = tai_dict[dest] - tai_dict[src]
        delta_usage = human_usage[dest] - human_usage[src]
        usage_deltas.append(delta_usage) 
        tai_deltas.append(delta_tai)
    df['delta_TAI'] = tai_deltas 
    df['delta_usage'] = usage_deltas 
     
    s_deltas = df['Mean']
    pearson = pearsonr(s_deltas,tai_deltas)[0]
    spearman = spearmanr(s_deltas,tai_deltas)[0]
    g = sns.jointplot(data=df,x='Mean',y='delta_TAI',kind='reg')
    plt.xlabel(r'$\Delta$S synonymous mutation') 
    plt.ylabel(r'$\Delta$tAI synonymous mutation') 
    full = f'Pearson={pearson:.3f}\nSpearman={spearman:.3f}'
    print(full)
    plt.text(0.05, 0.9,full,
            transform=g.ax_joint.transAxes)
    plt.savefig(f'{metric}_delta_tAI_regression.svg')
    plt.close()
    print(f'saved {metric}_delta_tAI_regression.svg')

def ism_mdig_codon_corr(ism,mdig):
    
    plt.figure(figsize=(3,3.5))
    total = ism.merge(mdig,on='long_mutation',suffixes=['_ism','_mdig'])
    ism_deltas = total['Mean_ism']
    mdig_deltas = total['Mean_mdig']
    pearson = pearsonr(ism_deltas,mdig_deltas)[0]
    spearman = spearmanr(ism_deltas,mdig_deltas)[0]

    g = sns.regplot(data=total,x='Mean_mdig',y='Mean_ism')
    plt.xlabel(r'Mean $\Delta$S-MDIG synonymous mutation',fontsize=8) 
    plt.ylabel(r'Mean $\Delta$S ISM synonymous mutation',fontsize=8) 
    full = f'r={pearson:.3f}\n'+r'$\rho$'+f'={spearman:.3f}'
    print(full) 
    plt.text(0.05, 0.9,full,
            transform=g.transAxes)
    sns.despine()
    plt.tight_layout()
    print('ism_mdig_codon_correlation.svg') 
    plt.savefig('ism_mdig_codon_correlation.svg')
    plt.close()

if __name__ == "__main__":
    
    args,unknown_args = parse_config()
    setup_fonts()
    output_dir = build_output_dir(args) 
    
    ism = pd.read_csv(f'{output_dir}/mut/ISM_codon_deltas.csv')
    mdig = pd.read_csv(f'{output_dir}/mut/MDIG_codon_deltas.csv')
    ism_mdig_codon_corr(ism,mdig)
    tai_codon_corr(ism,'ISM')
