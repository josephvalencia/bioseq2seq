import logomaker
from utils import parse_config,add_file_list,load_CDS
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot(consensus_df,name,axis):
    
    domain =list(range(consensus_df.shape[0]))
   
    crp_logo = logomaker.Logo(consensus_df,shade_below=.5,fade_below=.5,flip_below=True,ax=axis)
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    threes = [x for x in domain if x % 3 == 0]
    
    crp_logo.ax.axvspan(12, 15, color='red', alpha=0.3)
    crp_logo.ax.set_xticks(threes)
    crp_logo.ax.set_xticklabels(threes)
    crp_logo.ax.set_title(name)
    ''' 
    labels = ['<blank>','<unk>','A','C','G','T','N','R']
    for i in range(consensus_df.shape[1]):
        c = 'b' if labels[i] in ['A','G','C','T'] else 'k'
        plt.plot(domain,consensus_df[:,i],label=labels[i],alpha=0.8,linewidth=1)
    plt.legend()
    '''  
    '''
    ax = sns.heatmap(consensus_df,cmap='bwr',center=0,square=True,vmin=-0.15,vmax=0.1,robust=True,xticklabels=3)
    #ax = sns.heatmap(consensus_df,cmap='bwr',center=0,square=True,robust=True,xticklabels=3)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 18)
    ax.tick_params(axis='x',labelsize=28)
    ax.axhline(y=0, color='k',linewidth=2.5)
    ax.axhline(y=consensus.shape[0], color='k',linewidth=2.5)
    ax.axvline(x=0, color='k',linewidth=2.5)
    ax.axvline(x=consensus.shape[1], color='k',linewidth=2.5)
    ax.add_patch(Rectangle((b,0),3, 4, fill=False, edgecolor='yellow', lw=2.5))
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=24)
    plt_filename = f'{name}_logo.svg'
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()
    '''

def plot_examples(args):

    saved = np.load(args.best_BIO_grad_PC)

    test_file = args.test_csv
    test_cds = load_CDS(test_file)
    df_test = pd.read_csv(test_file,sep='\t')
    ncols = 2
    nrows = 2
    fig, axs = plt.subplots(nrows,ncols,sharey=False,figsize=(16,2))
    i = 0 
    
    for tscript,grad in saved.items():
        cds_loc = [int(x) for x in test_cds[tscript].split(':')] 
        s,e = tuple(cds_loc) 
         
        coding =  tscript.startswith('XM') or tscript.startswith('NM') 
        if coding and i < ncols*nrows and s >= 12 and e > s+60: 
            grad = grad[s-12:s+60,2:6].T
            labels = ['A','C','G','T']
            grad_df = pd.DataFrame(data=grad,index=labels,columns=list(range(grad.shape[1]))).T
            row = i // ncols
            col = i % ncols
            plot(grad_df,tscript,axis=axs[row,col])
            i+=1

    plt_filename = f'corrected_PC_logos.svg'
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

if __name__ == "__main__":

    args,unknown_args = parse_config()
    plot_examples(args)
