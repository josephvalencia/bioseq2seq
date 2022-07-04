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

from analysis.utils import parse_config, add_file_list , load_CDS
#import configargparse
#import yaml

def summarize_head(cds_storage,saved_file,tgt_head,mode="IG",align_on="start",coding=True):

    samples = []
    sample_ids = []
    before_lengths = []
    after_lengths = []

    with open(saved_file) as inFile:
        for l in inFile:
            fields = orjson.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"
            id = fields[id_field]
            is_pc = lambda x : x.startswith('NM_') or x.startswith('XM_')
            if id in cds_storage:
                # case 1 : protein coding  case 2 : non coding 
                if (coding and is_pc(id)) or (not coding and not is_pc(id)): 
                    cds = cds_storage[id]
                    if cds != "-1" :
                        splits = cds.split(":")
                        clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                        splits = [clean(x) for x in splits]
                        start,end = tuple([int(x) for x in splits])
                        attn = [float(x) for x in fields[tgt_head]]
                        # IG has padding, strip it out
                        if mode == "IG":
                            src = fields['src'].split('<pad>')[0]
                            attn = attn[:len(src)]
                        # align relative to start or stop codon 
                        if align_on == "start":
                            before_lengths.append(start)
                            after_lengths.append(len(attn) - start)
                        elif align_on == "end":
                            before_lengths.append(end)
                            after_lengths.append(len(attn) - end)
                        else:
                            raise ValueError("align_on must be 'start' or 'end'")
                        sample_ids.append(id)
                        samples.append(attn)

    percentiles = [10*x for x in range(11)]
    after_percentiles = np.percentile(after_lengths,percentiles)
    before_percentiles = np.percentile(before_lengths,percentiles)
    max_before = max(before_lengths)
    max_after = max(after_lengths)
    domain = np.arange(-max_before,999).reshape(1,-1)
    
    if align_on == "start":
        samples = [align_on_start(attn,start,max_before) for attn,start in zip(samples,before_lengths)]
    else:
        samples = [align_on_end(attn,end,max_after) for attn,end in zip(samples,before_lengths)]
    
    samples = np.asarray(samples)
    support = np.count_nonzero(~np.isnan(samples),axis=0)
    sufficient = support >= 0.70*samples.shape[0]
    samples = samples[:,sufficient]
    domain = domain[:,sufficient]
    consensus = np.nanmean(samples,axis=0)
    return consensus,domain.ravel()

def build_consensus_EDA(cds_storage,output_dir,prefix,attribution_dict,coding=True):

    file_list =  attribution_dict['path_list']
    model = attribution_dict['model']
    attr_type = attribution_dict['attr']
    labels = attribution_dict['layers']

    n_heads = 8

    consensus = []
    for l in labels:
        layer = file_list[l] 
        for h in range(n_heads):
            tgt_head = "layer{}head{}".format(l,h)
            summary,domain  = summarize_head(cds_storage,layer,tgt_head,mode="attn",align_on ="start",coding=coding) 
            consensus.append(summary.reshape(-1,1))
    consensus = np.concatenate(consensus,axis=1)
    
    suffix = "PC" if coding else "NC"
    name = f'{prefix}_{suffix}'
    run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,attr_type='EDA')

def build_consensus_multi_IG(cds_storage,output_dir,prefix,attribution_dict,summary_method,coding=True):

    file_list =  attribution_dict['path_list']
    model = attribution_dict['model']
    attr_type = attribution_dict['attr']
    labels = attribution_dict['bases']

    consensus = []
    for layer in file_list:
        summary,domain  = summarize_head(cds_storage,layer,summary_method,align_on ="start",coding=coding) 
        consensus.append(summary.reshape(-1,1))
    consensus = np.concatenate(consensus,axis=1)
   
    suffix = "PC" if coding else "NC"
    name = f'{prefix}_{suffix}'
    compute_heatmap = True if attr_type == 'MDIG' else False
    run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,attr_type=attr_type,heatmap=compute_heatmap)

def build_example_multi_IG(name,ig_file_list,tgt,id_list):

    for tscript in id_list:
        total = []
        for layer in ig_file_list:
            summary  = example_attributions(layer,tgt,tscript) 
            total.append(summary.reshape(-1,1))
        
        total = np.concatenate(total,axis=1)
        savefile = 'examples/'+tscript+'_'+name+"_multi.npz"
        np.savez(savefile,total=total) 

def example_attributions(saved_file,tgt,transcript):

    with open(saved_file) as inFile:
        query = "{\"ID\": \""+transcript
        for l in inFile:
            if l.startswith(query):
                fields = orjson.loads(l)
                id_field = "ID"
                id = fields[id_field]
                array = [float(x) for x in fields[tgt]]
                return np.asarray(array) 
    return None

def plot_line(domain,consensus,output_dir,name,model,attr_type,plot_type='line',plt_std_error=False,labels=None):
    
    plt.figure(figsize=(12,6))

    palette = sns.color_palette()
    n_positions,n_heads = consensus.shape
    
    if attr_type == 'EDA':
        for i in range(n_heads): 
            layer = i // 8
            label = layer if i % 8 == 0 else None
            if plot_type == 'stem':
                markerline, stemlines, baseline  = plt.stem(domain,consensus[:,i],label=label,use_line_collection=True,basefmt=" ")
            elif plot_type == 'line':
                plt.plot(domain,consensus[:,i],color=palette[layer],label=label,alpha=0.8,linewidth=1)
                if plt_std_error:
                    plt.fill_between(domain,consensus-2*error,consensus+2*error,alpha=0.5,edgecolor='#CC4F1B',facecolor='#FF9848')
    else: 
        for i in range(n_heads):
            if plot_type == 'stem':
                    markerline, stemlines, baseline  = plt.stem(domain,consensus[:,i],label=labels[i],use_line_collection=True,basefmt=" ")
            elif plot_type == 'line':
                plt.plot(domain,consensus[:,i],color=palette[i],label=labels[i],alpha=0.8,linewidth=1)
                if plt_std_error:
                    plt.fill_between(domain,consensus-2*error,consensus+2*error,alpha=0.5,edgecolor='#CC4F1B',facecolor='#FF9848')
    
    ax = plt.gca()
    legend_title = f'{model} {attr_type}'
    plt.legend(title=legend_title)
    
    # inset at 150-200
    inset_start = 50
    inset_stop = 100
    inset_domain = np.arange(inset_start,inset_stop)
    s = inset_start - domain.min()
    width = inset_stop - inset_start
    inset_range = consensus[s:s+width,:]
    axins = ax.inset_axes([0.4, 0.2, 0.5, 0.5])
    axins.axhline(y=0, color='gray', linestyle=':')     
    
    if attr_type == 'EDA':
        for i in range(n_heads): 
            layer = i // 8
            label = layer if i % 8 == 0 else None
            axins.plot(inset_domain,inset_range[:,i],color=palette[layer],label=label,alpha=0.8,linewidth=2.5)
    else:
        for i in range(n_heads): 
            axins.plot(inset_domain,inset_range[:,i],color=palette[i],label=labels[i],alpha=0.8,linewidth=2.5)
    
    ax.indicate_inset_zoom(axins, edgecolor="black")
   
    plt.axhline(y=0, color='gray', linestyle=':')     
    plt.xlabel("Position relative to CDS")
    plt.ylabel(f"Mean {attr_type} Score")
    plt.tight_layout(rect=[0,0.03, 1, 0.95])
    plt_filename = f'{output_dir}/{name}_{attr_type}_{plot_type}plot.svg'
    plt.savefig(plt_filename)
    plt.close()

def mean_by_mod(attn,savefile):

    idx = np.arange(attn.shape[0])
    zero = idx % 3 == 0
    one = idx % 3 == 1
    two = idx % 3 == 2

    storage = []
    for frame,mask in enumerate([zero,one,two]):
       slice = attn[mask]
       slice = slice[~np.isnan(slice)]
       print("frame {} , sum {}".format(frame,slice.sum()))
       for val in slice.tolist():
           entry = {"frame" : frame,"val" : val}
           storage.append(entry)

    df = pd.DataFrame(storage)
    result = stats.f_oneway(df['val'][df['frame'] == 0],df['val'][df['frame'] == 1],df['val'][df['frame'] == 2])
    means = [np.nanmean(attn[mask]) for mask in [zero,one,two]]
    
    textstr = '\n'.join((
    r'$F-statistic=%.3f$' % (result.statistic, ),
    r'$p-val=%.3f$' % (result.pvalue, )))

    print(textstr)

    '''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax = sns.barplot(x="frame",y="val",data=df)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    plt.xlabel("Pos. rel. to start mod 3")
    plt.ylabel("Attention")
    plt.title("Attention by Frame")
    prefix = savefile.split(".")[0]
    outfile = prefix+"attn_frames.svg"
    plt.savefig(outfile)
    plt.close()
    '''


def align_on_start(attn,cds_start,max_start,):

    max_len = 999
    indices = list(range(len(attn)))
    indices = [x-cds_start for x in indices]

    left_remainder = max_start - cds_start
    prefix = [np.nan for x in range(left_remainder)]
    right_remainder = max_len - indices[-1] -1
    suffix = [np.nan for x in range(right_remainder)]
    
    #min_information = 1/len(attn)
    min_information = -np.log2(1.0/len(attn))
    #attn = [min_information / -np.log2(x) for x in attn]
    #attn = [x/min_information for x in attn]
    total = prefix+attn+suffix
    return total

def align_on_end(attn,cds_end,max_end):

    max_len = 999
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
    
    consensus = consensus.T
    consensus = consensus[:4,cds_start-b:cds_start+60]
    min_val = np.min(consensus)
    max_val = np.max(consensus) 

    domain = list(range(-b,60)) 
    consensus_df = pd.DataFrame(data=consensus,index=['A','C','G','T'],columns=domain).T
    
    #df_melted = consensus_df.T.melt(var_name='MDIG baseline')
    #sns.displot(df_melted,x='value',hue='MDIG baseline',common_bins=True,bins=np.arange(-0.05,0.025,0.001))
    #plt.savefig(hist_file)
    #plt.close()
    #quit()
    #ax = sns.heatmap(consensus_df,cmap='bwr',vmin=-.15,vmax=0.1,center=0,square=True,robust=True,xticklabels=3)
    
    crp_logo = logomaker.Logo(consensus_df,flip_below=False)
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
    plt_filename = f'{output_dir}/{name}_MDIG_logo.svg'
    plt.savefig(plt_filename)
    plt.close()

def plot_power_spectrum(consensus,output_dir,name,model,attr_type,units='freq',labels=None):

    palette = sns.color_palette()
    freq,ps = signal.welch(consensus.transpose(1,0),axis=0,scaling='density',average='median')
    plt.figure(figsize=(5,3))
    ax1 = plt.gca() 

    n_freq_bins, n_heads = ps.shape
    print(f'n_freq_bins = {n_freq_bins} , n_heads = {n_heads}') 
    x_label = "Period (nt.)" if units == "period" else "Frequency (cycles/nt.)"
    x_vals = 1.0 / freq if units =="period" else freq    

    if attr_type == 'EDA':
        for i in range(n_heads):
            layer = i // 8
            label = layer if i % 8 == 0 else None
            ax1.plot(x_vals,ps[:,i],color=palette[layer],label=label,alpha=0.6)
    else:
        for i in range(n_heads):
            label = labels[i] if labels is not None else None
            ax1.plot(x_vals,ps[:,i],color=palette[i],label=label,alpha=0.6)
   
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
    plt.savefig(plt_filename)
    plt.close()

def scale_min_max(consensus):
    
    mins = consensus.min(axis=0)
    maxes = consensus.max(axis=0)
    return  (consensus - mins) / (maxes - mins)

def sample_examples(df,cds_storage):

    # select and save example transcripts
    np.random.seed(65)
    df = df.set_index('ID')
    id_list = np.random.choice(df.index.values,size=35,replace=False)
    seqs = df.loc[id_list]['Protein'].values.tolist()
    rna = df.loc[id_list]['RNA'].values.tolist() 
    df = df.reset_index() 
    cds_list = [cds_storage[i] for i in id_list]
    starts = [x.split(':')[0] for x in cds_list]
    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
    starts = [int(clean(s)) for s in starts] 
    ends = [int(clean(x.split(':')[1])) for x in cds_list] 
    np.savez('example_ids.npz',ids=id_list,protein=seqs,rna=rna,starts=starts,ends=ends) 


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
    df_test = pd.read_csv(test_file,sep='\t')
    
    # load attribution files from config
    best_seq_EDA = add_file_list(args.best_seq_EDA,'layers')
    best_EDC_EDA = add_file_list(args.best_EDC_EDA,'layers')
    best_seq_IG = add_file_list(args.best_seq_IG,'bases')
    best_EDC_IG = add_file_list(args.best_EDC_IG,'bases')
    best_seq_MDIG = add_file_list(args.best_seq_MDIG,'bases')
    best_EDC_MDIG = add_file_list(args.best_EDC_MDIG,'bases')
    
    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}/'
    if not os.path.isdir(output_dir):
        print("Building directory ...")
        os.mkdir(output_dir)
    
    # build EDA consensus, both coding and noncoding
    build_consensus_EDA(test_cds,output_dir,'best_seq2seq_test',best_seq_EDA,coding=True)
    build_consensus_EDA(test_cds,output_dir,'best_seq2seq_test',best_seq_EDA,coding=False)
    build_consensus_EDA(test_cds,output_dir,'best_EDC_test',best_EDC_EDA,coding=True)
    build_consensus_EDA(test_cds,output_dir,'best_EDC_test',best_EDC_EDA,coding=False)
   
    # build IG consensus
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test',best_seq_IG,'summed_attr',coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test',best_seq_IG,'summed_attr',coding=False)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test',best_EDC_IG,'summed_attr',coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test',best_EDC_IG,'summed_attr',coding=False)

    # build MDIG consensus
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test',best_seq_MDIG,'summed_attr',coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_seq2seq_test',best_seq_MDIG,'summed_attr',coding=False)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test',best_EDC_MDIG,'summed_attr',coding=True)
    build_consensus_multi_IG(test_cds,output_dir,'best_EDC_test',best_EDC_MDIG,'summed_attr',coding=False)

'''
def parse_config():

    p = configargparse.ArgParser() 
    
    p.add('--c','--config',required=False,is_config_file=True,help='path to config file')
    p.add('--test_csv',help='test dataset (.csv)' )
    p.add('--val_csv',help='validation dataset (.csv)')
    p.add('--train_csv',help='train dataset (.csv)')
    p.add('--competitors_results',help='competitors results (.csv)')
    p.add('--bioseq2seq_results',help='bioseq2seq results (.csv)')
    p.add('--EDC_results',help='EDC results (.csv)')
    p.add('--best_seq_self_attn',help='best bioseq2seq self-attention (.self_attn)')
    p.add('--best_EDC_self_attn',help='best EDC self-attention (.self_attn)')
    p.add('--best_seq_EDA',help ='best bioseq2seq encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_EDC_EDA',help ='best EDC encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_seq_IG',help = 'best bioseq2seq Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_EDC_IG',help = 'best EDC Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_seq_MDIG',help='best bioseq2seq MDIG (.ig)',type=yaml.safe_load)
    p.add('--best_EDC_MDIG',help='best EDC MDIG (.ig)',type=yaml.safe_load)
    
    return p.parse_known_args()
'''

if __name__ == "__main__":
    
    args,unknown_args = parse_config()
    build_all(args)
