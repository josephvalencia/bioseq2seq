import logomaker
from utils import parse_config,add_file_list,load_CDS
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot(consensus_df,name,target_pos,attr,axis=None):
    
    domain = list(range(-12,60))
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
    
    plt_filename = f'{name}_{attr}_logo.svg'
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

def plot_line(domain,grads,tscript,parent,class_type,target_pos,attr):
   
    plt.figure(figsize=(6.4,3.2))
    normed_grads = np.linalg.norm(grads,ord=2,axis=1)
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
        print(grad.shape)
        grad = grad[:,2:6].T
        labels = ['A','C','G','T']
        if s>=12 and s+60 < grad.shape[1]:
            cds_grad = -grad[:,s-12:s+60]
            domain = list(range(-12,60))
            grad_df = pd.DataFrame(data=cds_grad,index=labels,columns=domain).T
            plot(grad_df,tscript,target_pos,attr)
        row = i 
        domain = list(range(-s,grad.shape[0]-s))
        #plot_line(domain,grad,tscript,parent,class_type,target_pos,attr)
        
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
