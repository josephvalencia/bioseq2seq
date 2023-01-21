import logomaker
from utils import parse_config, build_EDA_file_list, load_CDS, grad_simplex_correction
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

def plot(consensus_df,name,target_pos,attr,axis=None):
    
    domain = list(range(-18,60))
    crp_logo = logomaker.Logo(consensus_df,shade_below=.5,fade_below=.5,flip_below=True,ax=axis)
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    threes = [x for x in domain if x % 3 == 0]
  
    target_pos = int(target_pos)

    crp_logo.ax.axvspan(-0.5, 2.5, color='green', alpha=0.3)
    if target_pos > 2: 
        left = 3*(target_pos-2)-0.5
        right = left+2.5
        crp_logo.ax.axvspan(left,right, color='red', alpha=0.3)
    crp_logo.ax.set_xticks(threes)
    crp_logo.ax.set_xticklabels(threes)
    crp_logo.ax.set_title(name)
    
    
    plt_filename = f'{name}_{attr}_summary_logo.svg'
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

def plot_v2(summary_df,signed_df,name,class_type,target_pos,attr,axis=None):

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(7.5,2))
    domain = list(range(-18,60))
    crp_logo = logomaker.Logo(summary_df,shade_below=.5,fade_below=.5,flip_below=True,ax=ax1,figsize=[7.5,1.25])
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    threes = [x for x in domain if x % 3 == 0]
  
    target_pos = int(target_pos)

    crp_logo.ax.axvspan(-0.5, 2.5, color='green', alpha=0.3)
    if target_pos > 2: 
        left = 3*(target_pos-2)-0.5
        right = left+2.5
        crp_logo.ax.axvspan(left,right, color='red', alpha=0.3)
    
    crp_logo.ax.set_xticks([])
    crp_logo.ax.set_xticklabels([])
    crp_logo.ax.set_title(name)
   
    vlim = np.nanmax(np.abs(signed_df.to_numpy()))
    g = sns.heatmap(signed_df,
                    cmap='RdBu_r',
                    linewidths=0,
                    cbar=False,
                    #cbar_kws= {'orientation':'horizontal'},
                    center=0,
                    vmin=-vlim,
                    vmax=vlim,
                    xticklabels=3,
                    ax=ax2)
    g.yaxis.set_ticklabels('ACGT',rotation='horizontal')
    plt.xticks(rotation=0) 
    plt.xlabel('Position rel. to start codon')
    #cbar_kws = {'orientation' :'horizontal'})
    #ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 18)
    #ax.tick_params(axis='x',labelsize=28)
    #ax.axhline(y=0, color='k',linewidth=2.5)
    #ax.axhline(y=consensus.shape[0], color='k',linewidth=2.5)
    #ax.axvline(x=0, color='k',linewidth=2.5)
    #ax.axvline(x=consensus.shape[1], color='k',linewidth=2.5)
    #ax.set_xticks(threes)
    #ax.set_xticklabels(threes)
    #ax.add_patch(Rectangle((b,0),3, 4, fill=False, edgecolor='yellow', lw=2.5))
    #cax = plt.gcf().axes[-1]
    #cax.tick_params(labelsize=24)

    plt_filename = f'{name}_{class_type}.{target_pos}.{attr}.mutations_logo.svg'
    plt.tight_layout(h_pad=0.3)
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

def plot_line(domain,grads,name,class_type,target_pos,attr):
   
    plt.figure(figsize=(6.4,3.2))
    #normed_grads = np.linalg.norm(grads,ord=2,axis=1)
    labels = 'ACGT'
    for j in range(grads.shape[1]):
        plt.plot(domain,grads[:,j],linewidth=1,label=labels[j])
    
    plt_filename = f'{name}_{attr}_lineplot.svg'
    plt.legend()
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

def plot_examples(parent,attr_type,class_type,target_pos):

    savefile = "/home/bb/valejose/valejose/bioseq2seq/{}.{}.{}.{}.npz".format(parent,class_type,target_pos,attr_type)
    saved = np.load(savefile)

    test_file = '/home/bb/valejose/valejose/bioseq2seq/data/mammalian_200-1200_test_nonredundant_80.csv'
    onehot_file = '/home/bb/valejose/valejose/bioseq2seq/coding/coding_RNA.PC.1.onehot.npz'
    test_cds = load_CDS(test_file)
    df_test = pd.read_csv(test_file,sep='\t')
    onehot_seqs = np.load(onehot_file)
    ncols = 1
    nrows = 4
    i = 0 
    
    for tscript,attr in saved.items():
        cds_loc = [int(x) for x in test_cds[tscript].split(':')] 
        s,e = tuple(cds_loc) 
        domain = list(range(-18,60))
        labels = ['A','C','G','T']
        onehot = onehot_seqs[tscript][:,2:6]
        if attr_type == 'grad':
            attr = attr[:,2:6]
            attr -= onehot * attr
        print(attr.shape) 
        if s>=18 and s+60 < attr.shape[0]:
            cds_attr = attr[s-18:s+60,:] 
            mins = np.min(attr,axis=1,keepdims=True) 
            max_min = np.max(mins[mins < 0.0]) 
            summary = np.abs(mins)
            summary = np.where(summary < 0.1*np.max(summary),0.1*np.max(summary),summary)
            print(summary) 
            summary = onehot * summary
            summary = summary[s-18:s+60,:] 
            domain = list(range(-18,60))
            summary_df = pd.DataFrame(data=summary,index=domain,columns=labels)
            signed_df = pd.DataFrame(data=cds_attr,index=domain,columns=labels).T
            plot_v2(summary_df,signed_df,tscript,class_type,target_pos,attr_type)
            #plot(signed_df.T,tscript,target_pos,attr_type)
            #cds_attr = -attr[:,s-18:s+60]
            #cds_attr = -attr_simplex_correction(attr[:,s-18:s+60])
            #attr_df = pd.DataFrame(data=cds_attr,index=labels,columns=domain).T
            #plot(attr_df,tscript,target_pos,attr)
        row = i 
        domain = list(range(-s,attr.shape[0]-s))
        plot_line(domain,attr,tscript,class_type,target_pos,attr_type)

if __name__ == "__main__":

    plot_examples(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
