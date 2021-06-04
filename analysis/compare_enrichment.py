import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def compare(centrality_file,frequency_file):

    centrality_df = pd.read_csv(centrality_file,sep='\t',names=['amino acid','codon','score'])
    frequency_df = pd.read_csv(frequency_file,sep='\t')
    combined = centrality_df.merge(frequency_df,on="codon")

    start_stop = ['TAA','TAG','TGA']
    combined = combined[~combined['codon'].isin(start_stop)]
    combined.reset_index(inplace=True)
    by_aa = combined.groupby('amino acid')
    x_var = 'score'
    y_var = 'enrichment'
    combined = combined.astype({x_var : float , y_var : float}) 
    min_x = np.min(combined[x_var].values)
    max_x = np.max(combined[x_var].values)
    min_y = np.min(combined[y_var].values)
    max_y = np.max(combined[y_var].values)

    tau_list = []
    pearson_list = []
    spearman_list = []

    for idx, (aa, df_aa) in enumerate(by_aa):
        print(df_aa)
        res = stats.kendalltau(df_aa[x_var].values,df_aa[y_var].values)
        print("Kendall tau = {}, p-val = {}".format(res.correlation,res.pvalue))
        tau_list.append(res.correlation)
        res = stats.linregress(df_aa[x_var].astype('float').values,df_aa[y_var].astype('float').values)
        print("R-squared = {}, p-val = {}".format(res.rvalue,res.pvalue))
        pearson_list.append(res.rvalue)
        rho,pvalue = stats.spearmanr(df_aa[x_var].values,df_aa[y_var].values)
        print("Spearman = {}, p-val = {}".format(rho,pvalue))
        spearman_list.append(rho)

    print('Mean Kendall tau = {}, mean Pearson R = {}, mean Spearman rho = {}'.format(np.mean(tau_list),np.mean(pearson_list),np.mean(spearman_list)))
    res = stats.kendalltau(combined[x_var].values,combined[y_var].values)
    print("Total Kendall tau = {}, p-val = {}".format(res.correlation,res.pvalue))
    res = stats.linregress(combined[x_var].astype('float').values,combined[y_var].astype('float').values)
    print("Total R-square  = {}, p-val = {}".format(res.rvalue**2,res.pvalue))
    rho,pvalue = stats.spearmanr(combined[x_var].values,combined[y_var].values)
    print("Spearman = {}, p-val = {}".format(rho,pvalue))

    plt.figure(figsize=(12,7))
    ax = sns.scatterplot(x=x_var,y=y_var,data=combined)
    sns.regplot(data=combined, x=x_var, y=y_var, scatter=False, ax=ax)

    textstr ='\n'.join(( r'$R^{2}=%.3f$' % (res.rvalue**2, ),r'$p=%.1E$' % (res.pvalue,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
    ax.text(0.075, 0.95, textstr, transform=ax.transAxes, fontsize=15,
    verticalalignment='top', bbox=props)

    x_spread = max_x - min_x
    y_spread = max_y - min_y
    plt.xlim(min_x-x_spread*0.05,max_x+x_spread*0.05)
    plt.ylim(min_y-y_spread*0.05,max_y+y_spread*0.05)
    
    for line in range(0,combined.shape[0]):
        ax.text(combined[x_var][line], combined[y_var][line]+0.0075*y_spread, \
                combined['codon'][line], horizontalalignment='center',size='medium', color='black')
   
    #plt.xlabel('MDIG synonymous codon graph sampled Katz centrality')
    plt.xlabel('MDIG synonymous codon graph mean weighted in-degree - outdegree')
    #plt.ylabel('PC enrichment (log odds-ratio) ')
    plt.ylabel('CDS enrichment (log odds-ratio) ')
    plt.tight_layout()
    #plt.savefig('codon_scatter_no_stop_wt_degree_PC.svg')
    plt.savefig('codon_scatter_no_stop_wt_degree_CDS.svg')
    #plt.savefig('codon_scatter_no_stop_sampled_katz.svg')
    plt.close()

if __name__ == "__main__":

    compare(sys.argv[1],sys.argv[2])
