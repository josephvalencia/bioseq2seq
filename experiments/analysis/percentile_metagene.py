import logomaker
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import re,random
from utils import parse_config,build_EDA_file_list,load_CDS
import math

def update_percentile_scores(running_percentile,scores,n_bins):

    cuts = np.linspace(0,scores.shape[1]-1,n_bins+1)
    cuts = [math.ceil(c) for c in cuts]
    percentile_splits = np.split(scores,cuts[1:-1],axis=1)
    averaged = [x.mean(axis=1) for x in percentile_splits]
    return running_percentile + np.asarray(averaged).T

def is_valid(start,end,total):
    return start >= 50 and (end-start) >= 100 and (total-end) >= 50

def metagene(cds_storage,saved_file,n_bins,grad=False):

    valid_count = 0 
   
    # initialize sum for mRNAs
    five_prime = np.zeros((8,n_bins)) 
    cds = np.zeros((8,n_bins))
    three_prime = np.zeros((8,n_bins)) 
    # initialize sum for lncRNAs
    upstream = np.zeros((8,n_bins)) 
    longest_orf = np.zeros((8,n_bins))
    downstream = np.zeros((8,n_bins)) 
    
    saved = np.load(saved_file)
    for tscript, attr in saved.items():
        is_pc =  tscript.startswith('NM') or tscript.startswith('XM')
        if grad:
            attr = attr.reshape(1,-1)
        if tscript in cds_storage:
            # case 1 : protein coding  case 2 : non coding 
            cds_loc = cds_storage[tscript]
            if cds_loc != "-1" :
                splits = cds_loc.split(":")
                clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                splits = [clean(x) for x in splits]
                start,end = tuple([int(x) for x in splits])
                if is_valid(start,end,attr.shape[1]): 
                    valid_count +=1                         
                    # divide by functional area
                    if is_pc: 
                        five_prime = update_percentile_scores(five_prime,attr[:,:start],n_bins) 
                        cds = update_percentile_scores(cds,attr[:,start:end],n_bins)
                        three_prime = update_percentile_scores(three_prime,attr[:,end:],n_bins) 
                    else:
                        upstream = update_percentile_scores(upstream,attr[:,:start],n_bins) 
                        longest_orf = update_percentile_scores(longest_orf,attr[:,start:end],n_bins)
                        downstream = update_percentile_scores(downstream,attr[:,end:],n_bins) 

    print(f'valid_count = {valid_count}')
    total_pc = np.concatenate([five_prime,cds,three_prime],axis=1) / valid_count
    total_nc = np.concatenate([upstream,longest_orf,downstream],axis=1) / valid_count
    return total_pc, total_nc

def build_consensus_EDA(cds_storage,output_dir,prefix,attribution_dict):

    file_list =  attribution_dict['path_list']
    model = attribution_dict['model']
    attr_type = attribution_dict['attr']
    n_layers = attribution_dict['n_layers']
    labels = [f'layer_{i}' for i in range(n_layers)]
    consensus = []
    
    n_bins = 25
    for l in range(n_layers):
        layer = file_list[l] 
        total_pc,total_nc = metagene(cds_storage,layer,n_bins) 
        name = f'{prefix}_EDA_layer{l}'
        plot_line(total_pc,total_nc,output_dir,name,n_bins,attr_type='EDA')

def build_consensus_multi_IG(cds_storage,output_dir,prefix,grad_file,coding=True):

    '''
    file_list =  attribution_dict['path_list']
    model = attribution_dict['model']
    attr_type = attribution_dict['attr']
    labels = attribution_dict['bases']
    '''
    consensus,domain  = summarize_head(cds_storage,grad_file,grad=False,align_on="start",coding=coding) 
    consensus = consensus.reshape(1,consensus.shape[0],consensus.shape[1]) 
    
    suffix = "group=PC" if coding else "group=NC"
    name = f'{prefix}_{suffix}'
    model = 'bioseq2seq'
    attr_type = 'grad'
    compute_heatmap = True 
    labels = [str(x) for x in range(8)]
    run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,attr_type=attr_type,heatmap=compute_heatmap)

def plot_line(total_pc,total_nc,output_dir,name,n_bins,attr_type='EDA'):
    
    if attr_type == 'EDA':
        fig,axes = plt.subplots(2,4,figsize=(8,2.5))
        for i,ax in enumerate(axes.flat): 
            
            if i == 0:
                ax.plot(total_pc[i,:],linewidth=1,label='mRNA',color='tab:red')
                ax.plot(total_nc[i,:],linewidth=1,label='lncRNA',color='tab:blue')
            else:
                ax.plot(total_pc[i,:],linewidth=1,color='tab:red')
                ax.plot(total_nc[i,:],linewidth=1,color='tab:blue')

            ax.axvline(x=n_bins, color='gray', linestyle=':')     
            ax.axvline(x=2*n_bins, color='gray', linestyle=':')
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            ax.set_xlabel(f'head {i}')
            if i % 4 == 0:
                ax.set_ylabel(f'{attr_type} consensus')
    else: 
        plt.plot(total,linewidth=1)

    plt.figlegend(loc='center')
    plt.tight_layout()
    plt_filename = f'{output_dir}/{name}_metagene.svg'
    print(f'Saving {plt_filename}')
    plt.savefig(plt_filename)
    plt.close()

def run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,attr_type,heatmap=False):

    plot_line(domain,consensus,output_dir,name,model,attr_type,plot_type='line',labels=labels)

def build_all(args):

    test_file = args.test_csv
    train_file = args.train_csv
    val_file = args.val_csv
    test_cds = load_CDS(test_file)
    val_cds = load_CDS(val_file)
    df_test = pd.read_csv(test_file,sep='\t')
    
    # load attribution files from config
    best_BIO_EDA = build_EDA_file_list(args.best_BIO_EDA,args.best_BIO_DIR)
    best_EDC_EDA = build_EDA_file_list(args.best_EDC_EDA,args.best_EDC_DIR)

    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}/'
    if not os.path.isdir(output_dir):
        print("Building directory ...")
        os.mkdir(output_dir)
   
    # build EDA consensus, both coding and noncoding
    build_consensus_EDA(test_cds,output_dir,'best_seq2seq_test',best_BIO_EDA)
    build_consensus_EDA(test_cds,output_dir,'best_EDC_test',best_EDC_EDA)
    
if __name__ == "__main__":
    
    args,unknown_args = parse_config()
    build_all(args)
