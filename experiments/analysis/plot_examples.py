import logomaker
from utils import parse_config,add_file_list,load_CDS
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot(consensus_df,name,target_pos,axis=None):
    
    #domain = list(range(-12,60))
    crp_logo = logomaker.Logo(consensus_df,shade_below=.5,fade_below=.5,flip_below=True,ax=axis)
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    #threes = [x for x in domain if x % 3 == 0]
  
    target_pos = int(target_pos)

    crp_logo.ax.axvspan(-0.5, 2.5, color='green', alpha=0.3)
    if target_pos > 2: 
        left = 3*(target_pos-2)-0.5
        right = left+2.5
        crp_logo.ax.axvspan(left,right, color='red', alpha=0.3)
    #crp_logo.ax.set_xticks(threes)
    #crp_logo.ax.set_xticklabels(threes)
    crp_logo.ax.set_title(name)
    
    plt_filename = f'{name}_logo.svg'
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()
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

def plot_line(domain,grads,tscript,parent,class_type,target_pos,attr):
   
    plt.figure(figsize=(6.4,3.2))
    normed_grads = np.linalg.norm(grads,ord=2,axis=0)
    plt.plot(domain,normed_grads,linewidth=1)
    plt_filename = "/home/bb/valejose/home/bioseq2seq/{}_{}.{}.{}_{}.svg".format(parent,class_type,target_pos,attr,tscript)
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

def plot_examples(parent,attr,class_type,target_pos):

    savefile = "/home/bb/valejose/home/bioseq2seq/{}/{}.{}.{}.npz".format(parent,class_type,target_pos,attr)
    print(savefile)
    saved = np.load(savefile)

    #test_file = args.test_csv
    test_file = '/home/bb/valejose/home/bioseq2seq/data/mammalian_200-1200_test_nonredundant_80.csv'
    test_cds = load_CDS(test_file)
    df_test = pd.read_csv(test_file,sep='\t')
    ncols = 1
    nrows = 4
    #fig, axs = plt.subplots(nrows,ncols,sharey=False,figsize=(12,6))
    i = 0 
    
    for tscript,grad in saved.items():
        cds_loc = [int(x) for x in test_cds[tscript].split(':')] 
        s,e = tuple(cds_loc) 
        #grad = grad[:,2:6].T
        labels = ['A','C','G','T']
        print(grad.shape) 
        if s>=12 and s+60 < grad.shape[1]:
            cds_grad = grad[:,s-12:s+60]
            domain = list(range(-12,60))
            grad_df = pd.DataFrame(data=cds_grad,index=labels,columns=domain).T
            plot(grad_df,tscript,target_pos)
        
        row = i 
        domain = list(range(-s,grad.shape[1]-s))
        plot_line(domain,grad,tscript,parent,class_type,target_pos,attr)
        i+=1
        if i >= 4:
            break
        ''' 
        if i < ncols*nrows and s >= 12 and e > s+60: 
            grad = grad[s-12:s+60,2:6].T
            labels = ['A','C','G','T']
            domain = list(range(-12,60))
            grad_df = pd.DataFrame(data=grad,index=labels,columns=domain).T
            row = i 
            plot(grad_df,tscript,target_pos,axis=axs[row])
            i+=1
    plt_filename = "/home/bb/valejose/home/bioseq2seq/{}_{}.{}.{}_logos.svg".format(parent,class_type,target_pos,attr)
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()
    '''

if __name__ == "__main__":

    plot_examples(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
