import logomaker
import orjson
import os
import sys
import gzip
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import yaml
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Rectangle
from collections import Counter
from scipy import stats , signal
import re,random
from scipy.stats import pearsonr, kendalltau
from Bio.Seq import Seq
from Bio import motifs
from sklearn import preprocessing
from datetime import datetime

from utils import parse_config,add_file_list,load_CDS

def summarize_head(cds_storage,saved_file,grad=False,align_on="start",coding=True):

    samples = []
    sample_ids = []
    before_lengths = []
    after_lengths = []

    saved = np.load(saved_file)
    
    for tscript, attr in saved.items():
        is_pc = lambda x : x.startswith('NM_') or x.startswith('XM_')
        attr = attr.T
        if grad:
            attr = attr.reshape(1,-1)
        if tscript in cds_storage:
            # case 1 : protein coding  case 2 : non coding 
            if (coding and is_pc(tscript)) or (not coding and not is_pc(tscript)): 
                cds = cds_storage[tscript]
                if cds != "-1" :
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    splits = [clean(x) for x in splits]
                    start,end = tuple([int(x) for x in splits])
                    # align relative to start or stop codon 
                    if align_on == "start":
                        before_lengths.append(start)
                        after_lengths.append(len(attr) - start)
                    elif align_on == "end":
                        before_lengths.append(end)
                        after_lengths.append(len(attr) - end)
                    else:
                        raise ValueError("align_on must be 'start' or 'end'")
                    sample_ids.append(tscript)
                    samples.append(attr)

    percentiles = [10*x for x in range(11)]
    after_percentiles = np.percentile(after_lengths,percentiles)
    before_percentiles = np.percentile(before_lengths,percentiles)
    max_before = max(before_lengths)
    max_after = max(after_lengths)
    domain = np.arange(-max_before,1200).reshape(1,-1)
     
    if align_on == "start":
        samples = [align_on_start(id,attn,start,max_before) for id,attn,start in zip(sample_ids,samples,before_lengths)]
    else:
        samples = [align_on_end(attn,end,max_after) for attn,end in zip(samples,before_lengths)]
   
    samples = np.stack(samples,axis=2)
    first = samples[0,:,:]
    support = np.count_nonzero(~np.isnan(first),axis=1)
    sufficient = support >= 0.70*first.shape[1]
    samples = samples[:,sufficient,:]
    domain = domain[:,sufficient]
    consensus = np.nanmean(samples,axis=2)
    return consensus.transpose(0,1),domain.ravel()

def build_consensus_EDA(cds_storage,output_dir,prefix,attribution_dict,coding=True):

    file_list =  attribution_dict['path_list']
    model = attribution_dict['model']
    attr_type = attribution_dict['attr']
    labels = attribution_dict['layers']

    consensus = []
    for l in labels:
        layer = file_list[l] 
        summary,domain  = summarize_head(cds_storage,layer,align_on="start",coding=coding) 
        consensus.append(summary.reshape(1,summary.shape[0],summary.shape[1]))
    
    consensus = np.concatenate(consensus,axis=0)
    
    suffix = "PC" if coding else "NC"
    name = f'{prefix}_{suffix}'
    run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,attr_type='EDA')

def build_consensus_multi_IG(cds_storage,output_dir,prefix,grad_file,coding=True):

    '''
    file_list =  attribution_dict['path_list']
    model = attribution_dict['model']
    attr_type = attribution_dict['attr']
    labels = attribution_dict['bases']
    '''
    consensus,domain  = summarize_head(cds_storage,grad_file,grad=False,align_on="start",coding=coding) 
    consensus = consensus.reshape(1,consensus.shape[0],consensus.shape[1]) 
    print(domain.min(),domain.max())
    suffix = "group=PC" if coding else "group=NC"
    name = f'{prefix}_{suffix}'
    model = 'bioseq2seq'
    attr_type = 'grad'
    compute_heatmap = True 
    labels = [str(x) for x in range(8)]
    run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,attr_type=attr_type,heatmap=compute_heatmap)

def plot_line(domain,consensus,output_dir,name,model,attr_type,plot_type='line',plt_std_error=False,labels=None):
    
    plt.figure(figsize=(12,6))

    palette = sns.color_palette()
    print('in plot',consensus.shape,domain.shape) 
    n_layers,n_heads,n_positions = consensus.shape
    
    if attr_type == 'EDA':
        for layer in range(n_layers):
            for i in range(n_heads): 
                label = layer if i % 8 == 0 else None
                color = layer % len(palette)
                print(layer,i)
                plt.plot(domain,consensus[layer,i,:],color=palette[color],label=label,alpha=0.8,linewidth=1)
    else: 
        for layer in range(n_layers):
            for i in range(n_heads):
                plt.plot(domain,consensus[layer,i,:],color=palette[i],label=labels[i],alpha=0.8,linewidth=1)
    ax = plt.gca()
    legend_title = f'{model} {attr_type}'
    plt.legend(title=legend_title)
    
    # inset at 150-200
    inset_start = 50
    inset_stop = 100
    inset_domain = np.arange(inset_start,inset_stop)
    s = inset_start - domain.min()
    width = inset_stop - inset_start
    inset_range = consensus[:,:,s:s+width]
    axins = ax.inset_axes([0.4, 0.2, 0.5, 0.5])
    axins.axhline(y=0, color='gray', linestyle=':')     
    
    if attr_type == 'EDA':
        for layer in range(n_layers):
            for i in range(n_heads): 
                label = layer if i % 8 == 0 else None
                subslice = inset_range[layer,i,:]
                color = layer % len(palette)
                axins.plot(inset_domain,subslice,color=palette[color],label=label,alpha=0.4,linewidth=2.5)
    else:
        for layer in range(n_layers):
            for i in range(n_heads): 
                subslice = inset_range[layer,i,:]
                axins.plot(inset_domain,subslice,color=palette[i],label=labels[i],alpha=0.8,linewidth=2.5)
    
    ax.indicate_inset_zoom(axins, edgecolor="black")
   
    plt.axhline(y=0, color='gray', linestyle=':')     
    
    plt.xlabel("Position relative to CDS")
    plt.ylabel(f"Mean {attr_type} Score")
    plt.tight_layout(rect=[0,0.03, 1, 0.95])
    plt_filename = f'{output_dir}/{name}_{attr_type}_{plot_type}plot.svg'
    print(f'Saving {plt_filename}')
    plt.savefig(plt_filename)
    plt.close()

def align_on_start(id,attn,cds_start,max_start):
    
    max_len = 1200
    left_remainder = max_start - cds_start
    prefix = np.ones((attn.shape[0],left_remainder)) * np.nan
    right_remainder = max_len + max_start - (left_remainder+attn.shape[-1])
    suffix = np.ones((attn.shape[0],right_remainder)) * np.nan
    total = np.concatenate([prefix,attn,suffix],axis=1)
    return total

def align_on_end(attn,cds_end,max_end):

    max_len = 1200
    indices = list(range(len(attn)))
    indices = [x-cds_end for x in indices]

    left_remainder = max_len-cds_end
    prefix = [np.nan for x in range(left_remainder)]
    right_remainder = max_end - indices[-1]-1
    suffix = [np.nan for x in range(right_remainder)]
    total = prefix+attn+suffix
    return total

def plot_heatmap(consensus,domain,output_dir,name,model):

    cds_start = -domain[0]

    plt.figure(figsize=(24, 6))
    palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

    b = 12 if cds_start > 12 else cds_start 
     
    #consensus = consensus.T
    consensus = consensus[0,2:6,cds_start-b:cds_start+60]
    min_val = np.min(consensus)
    max_val = np.max(consensus) 

    print('CONSENSUS',consensus.shape) 
    domain = list(range(-b,60)) 
    consensus_df = pd.DataFrame(data=consensus,index=['A','C','G','T'],columns=domain).T
    
    #df_melted = consensus_df.T.melt(var_name='MDIG baseline')
    #sns.displot(df_melted,x='value',hue='MDIG baseline',common_bins=True,bins=np.arange(-0.05,0.025,0.001))
    #plt.savefig(hist_file)
    #plt.close()
    #quit()
    #ax = sns.heatmap(consensus_df,cmap='bwr',vmin=-.15,vmax=0.1,center=0,square=True,robust=True,xticklabels=3)
    
    crp_logo = logomaker.Logo(consensus_df,flip_below=True)
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    
    threes = [x for x in domain if x % 3 == 0]
    crp_logo.ax.set_xticks(threes)
    crp_logo.ax.set_xticklabels(threes)

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
    '''
    plt_filename = f'{output_dir}/{name}_grad_logo.svg'
    plt.savefig(plt_filename)
    plt.close()

def plot_power_spectrum(consensus,output_dir,name,model,attr_type,units='freq',labels=None):

    print(consensus.shape)
    palette = sns.color_palette()
    freq, ps = signal.welch(consensus,axis=2,scaling='density',average='median')
    plt.figure(figsize=(5,3))
    ax1 = plt.gca() 

    n_layers, n_heads, n_freq_bins = ps.shape
    print(f'n_layers = {n_layers}, n_freq_bins = {n_freq_bins} , n_heads = {n_heads}') 
    x_label = "Period (nt.)" if units == "period" else "Frequency (cycles/nt.)"

    if attr_type == 'EDA':
        for l in range(n_layers):
            for i in range(n_heads):
                label = l if i % 8 == 0 else None
                color = l % len(palette)
                ax1.plot(freq,ps[l,i,:],color=palette[color],label=label,alpha=0.6)
    else:
        for l in range(n_layers):
            for i in range(n_heads):
                label = labels[i] if labels is not None else None
                ax1.plot(freq,ps[l,i,:],color=palette[i],label=label,alpha=0.6)
   
    tick_labels = ["0",r'$\frac{1}{10}$']+[r"$\frac{1}{"+str(x)+r"}$" for x in range(5,1,-1)]
    tick_locs =[0,1.0/10]+ [1.0 / x for x in range(5,1,-1)]
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labels,fontsize=14)

    if attr_type == 'EDA':
        ax1.legend(title=f'{model} attention layer')
        ax1.set_ylabel("Attention Power Spectrum")
    else:
        ax1.legend(title=f'{model} {attr_type} baseline')
        ax1.set_ylabel(f"{attr_type} Power Spectrum")
    
    ax1.set_xlabel(x_label)
    ax1.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    plt.tight_layout()
    plt_filename = f'{output_dir}/{name}_{attr_type}_PSD.svg'
    print(f'Saving {plt_filename}')
    plt.savefig(plt_filename)
    plt.close()

def run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,attr_type,heatmap=False):

    plot_line(domain,consensus,output_dir,name,model,attr_type,plot_type='line',labels=labels)
    plot_power_spectrum(consensus,output_dir,name,model,attr_type=attr_type,labels=labels)
    
    # only defined for heatmap
    if heatmap:
        plot_heatmap(consensus,domain,output_dir,name,model)

def build_all(args):

    test_file = args.test_csv
    train_file = args.train_csv
    val_file = args.val_csv
    test_cds = load_CDS(test_file)
    val_cds = load_CDS(val_file)
    df_test = pd.read_csv(test_file,sep='\t')
    
    # load attribution files from config
    best_BIO_EDA = add_file_list(args.best_BIO_EDA,'layers')
    best_EDC_EDA = add_file_list(args.best_EDC_EDA,'layers')
    
    best_BIO_grad_PC = args.best_BIO_grad_PC
    best_EDC_grad_PC = args.best_EDC_grad_PC
    best_BIO_grad_NC = args.best_BIO_grad_NC
    best_EDC_grad_NC = args.best_EDC_grad_NC
    
    best_BIO_inputXgrad_PC = args.best_BIO_inputXgrad_PC
    best_EDC_inputXgrad_PC = args.best_EDC_inputXgrad_PC
    best_BIO_inputXgrad_NC = args.best_BIO_inputXgrad_NC
    best_EDC_inputXgrad_NC = args.best_EDC_inputXgrad_NC

    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}/'
    if not os.path.isdir(output_dir):
        print("Building directory ...")
        os.mkdir(output_dir)
   
    # build EDA consensus, both coding and noncoding
    #build_consensus_EDA(test_cds,output_dir,'best_seq2seq_test',best_BIO_EDA,coding=True)
    #build_consensus_EDA(test_cds,output_dir,'best_seq2seq_test',best_BIO_EDA,coding=False)
    #build_consensus_EDA(test_cds,output_dir,'best_EDC_test',best_EDC_EDA,coding=True)
    #build_consensus_EDA(test_cds,output_dir,'best_EDC_test',best_EDC_EDA,coding=False)
    
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test_tgt=PC',best_BIO_grad_PC,coding=True)
    '''
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test_tgt=PC_',best_BIO_grad_PC,coding=False)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test_PC_tgt=PC',best_EDC_grad_PC,coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test_tgt=PC',best_EDC_grad_PC,coding=False)

    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test_tgt=NC',best_BIO_grad_NC,coding=False)
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test_tgt=NC',best_BIO_grad_NC,coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test_tgt=NC',best_EDC_grad_NC,coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test_tgt=NC',best_EDC_grad_NC,coding=False)
    
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test',best_BIO_inputXgrad_PC,coding=False)
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test',best_BIO_inputXgrad_PC,coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test',best_BIO_inputXgrad_NC,coding=False)
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test',best_BIO_inputXgrad_NC,coding=True)
    
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test',best_EDC_inputXgrad_PC,coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test',best_EDC_inputXgrad_PC,coding=False)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test',best_EDC_inputXgrad_NC,coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test',best_EDC_inputXgrad_NC,coding=False)

    '''
if __name__ == "__main__":
    
    args,unknown_args = parse_config()
    build_all(args)
