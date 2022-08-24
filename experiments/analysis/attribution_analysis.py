import sys,random
import json
import os,re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr,kendalltau,ttest_ind,entropy
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import jensenshannon
from collections import defaultdict
from IPython.display import Image

from Bio.Data import CodonTable
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

from utils import parse_config, add_file_list, getLongestORF, get_CDS_start

def get_top_k(array,k=1):

    k = k if k < len(array) else len(array)
    array = np.asarray(array)
    k_largest_inds = np.argpartition(array,-k)[-k:]
    k_largest_scores = array[k_largest_inds].tolist()
    k_largest_inds = k_largest_inds.tolist()
    return k_largest_scores,k_largest_inds

def get_min_k(array,k=1):

    k = k if k < len(array) else len(array)
    array = np.asarray(array)
    k_smallest_inds = np.argpartition(array,k)[:k]
    k_smallest_inds = k_smallest_inds.tolist()
    return k_smallest_inds

def top_indices(saved_file,positive_topk_file,negative_topk_file,groups,metrics,mode="attn"):
    
    df_storage = []
    negative_storage = []
    positive_storage = []
    val_list = []

    out_name = saved_file.split(".")[0]

    loaded = np.load(saved_file)

    for tscript,array in loaded.items():
        
        L = len(array)
        array = np.asarray(array) 
        name = tscript + "_" + mode
        
        # find various indices of interest
        max_idx = np.argmax(array).tolist()
        min_idx = np.argmin(array).tolist()
        smallest_magnitude_idx = np.argmin(np.abs(array)).tolist()
        largest_magnitude_idx = np.argmax(np.abs(array)).tolist()
        
        disallowed = [max_idx+i for i in range(-20,20)]
        other_idx = [x for x in range(L) if x not in disallowed]
        random_idx = random.sample(other_idx,1)[0]
        
        val_list.append(np.max(array))

        coding = True if (tscript.startswith('XM_') or tscript.startswith('NM_')) else False 
        both_storage = [positive_storage,negative_storage]

        for g,m,dataset in zip(groups,metrics,both_storage): 
            result = None
            # partition based on args
            if g == 'PC' and coding:
                if m == 'max':
                    result = (tscript,max_idx)
                elif m == 'min':
                    result = (tscript,min_idx)
                elif m == 'random':
                    result = (tscript,random_idx)
            elif g == 'NC' and not coding:
                if m == 'max':
                    result = (tscript,max_idx)
                elif m == 'min':
                    result = (tscript,min_idx)
                elif m == 'random':
                    result = (tscript,random_idx)
            if result is not None:
                dataset.append(result) 

    val_mean = np.mean(val_list)
    val_std = np.std(val_list)
    print(f'Max val: mean={val_mean}, std={val_std}')
    
    # save top indices
    with open(positive_topk_file,'w') as outFile:
        for tscript,idx in positive_storage:
            outFile.write("{},{}\n".format(tscript,idx))

    with open(negative_topk_file,'w') as outFile:
        for tscript,idx in negative_storage:
            outFile.write("{},{}\n".format(tscript,idx))

def top_k_to_substrings(top_k_csv,motif_fasta,df):
    
    storage = []
    sequences = []
    
    # ingest top k indexes from attribution/attention
    with open(top_k_csv) as inFile:
        for l in inFile:
            fields = l.rstrip().split(",")
            tscript = fields[0]
            seq = df.loc[tscript,'RNA']

            left_bound = 10
            right_bound = 10

            # get window around indexes
            for num,idx in enumerate(fields[1:]):
                idx = int(idx)

                # enforce uniform window
                if idx < left_bound:
                    idx = left_bound
                if idx > len(seq) - right_bound -1:
                    idx = len(seq) - right_bound -1 

                start = idx-left_bound if idx-left_bound >= 0 else 0
                end = idx+right_bound+1

                substr = seq[start:end]
                if len(substr) != 21:
                    print(tscript,start,end,substr)
                if len(substr) > 7:
                    description = "loc[{}:{}]".format(start+1,end+1)
                    record = SeqRecord(Seq(substr),
                                            id=tscript+"_"+str(num),
                                            description=description)
                    sequences.append(record)

    with open(motif_fasta,'w') as outFile:
        SeqIO.write(sequences, outFile, "fasta")

def run_attributions(saved_file,df,parent_dir,groups,metrics,mode="attn"):

    attr_name = os.path.split(saved_file)[1]
    attr_name = attr_name.split('.')[0]
    prefix = f'{parent_dir}/{attr_name}/' 
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    # results files
    positive_indices_file = prefix+"positive_topk_idx.txt"
    negative_indices_file = prefix+"negative_topk_idx.txt"
    positive_motifs_file = prefix+"positive_motifs.fa"
    negative_motifs_file = prefix +"negative_motifs.fa"
    hist_file = prefix+"pos_hist.svg"

    top_indices(saved_file,positive_indices_file,negative_indices_file,groups,metrics,mode=mode)
    top_k_to_substrings(positive_indices_file,positive_motifs_file,df)
    top_k_to_substrings(negative_indices_file,negative_motifs_file,df)
    get_positional_bias(positive_indices_file,negative_indices_file,df,hist_file)

def get_positional_bias(coding_indices_file,noncoding_indices_file,df_data,hist_file):

    storage = []

    pc = pd.read_csv(coding_indices_file,names=['ID','start'])
    nc = pd.read_csv(noncoding_indices_file,names=['ID','start'])
    df_attn = pd.concat([pc,nc])
    df_data['cds_start'] = [get_CDS_start(cds,seq) for cds,seq in zip(df_data['CDS'].values.tolist(),df_data['RNA'].values.tolist())]
    df = pd.merge(df_attn,df_data,on='ID')
    df['rel_start'] = df['start'] - df['cds_start']-1
    df = df.drop(df.columns.difference(['Type','rel_start']),1)

    bins = np.arange(-750,1200,10)
    g = sns.displot(data=df,x='rel_start',col='Type',kind='hist',stat='density',bins=bins,element='step')
    axes = g.axes.flatten()
    axes[0].set_title("")
    axes[0].set_xlabel("Position of max IG val rel. start")
    axes[0].set_ylabel("Density")
    axes[1].set_title("")
    axes[1].set_xlabel("Position of min IG val rel. to start longest ORF")
    axes[1].set_ylabel("")
    plt.savefig(hist_file)
    plt.close()

def attribution_loci_pipeline(): 

    args, unknown_args = parse_config()
    
    # ingest stored data
    test_file = args.test_csv
    train_file = args.train_csv
    val_file = args.val_csv
    df_test = pd.read_csv(test_file,sep="\t").set_index("ID")
   
    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}/'
    attr_dir  =  f'{output_dir}/attr/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # make subdir for attribution loci results  
    if not os.path.isdir(attr_dir):
        os.mkdir(attr_dir)

    # load attribution files from config
    best_BIO_EDA = add_file_list(args.best_BIO_EDA,'layers')
    best_EDC_EDA = add_file_list(args.best_EDC_EDA,'layers')
    best_BIO_grad = args.best_BIO_grad
    best_EDC_grad = args.best_EDC_grad
    
    groups = [['PC','NC'],['PC','PC'],['NC','NC']]
    metrics = [['max','max'],['max','min'],['max','random'],['min','random'],['random','random']]
    
    for g in groups:
        for m in metrics:
            # g[0] and g[1] are transcript type for pos and neg sets
            # m[0] and m[1] are loci of interest for pos and neg sets
            a = f'{g[0]}-{m[0]}'
            b = f'{g[1]}-{m[1]}'
            # ensure comparison groups are different 
            if a != b:
                trial_name = f'{a}_{b}'
                # build directories
                best_EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/'
                if not os.path.isdir(best_EDC_dir):
                    os.mkdir(best_EDC_dir)
                best_BIO_dir = f'{attr_dir}/best_seq2seq_{trial_name}/'
                if not os.path.isdir(best_BIO_dir):
                    os.mkdir(best_BIO_dir)
                
                # run all IG bases for both models 
                run_attributions(best_BIO_grad,df_test,best_BIO_dir,g,m,'inputXgrad')
                #run_attributions(best_EDC_grad,df_test,'summed_attr',best_EDC_dir,g,m,'IG')
                ''' 
                # run all EDA layers for both models
                for l,f in enumerate(best_seq_EDA['path_list']):
                    for h in range(8):
                        tgt_head = f'layer{l}head{h}'
                        run_attributions(f,df_test,tgt_head,best_seq_dir,g,m,'attn')
                for l,f in enumerate(best_EDC_EDA['path_list']):
                    for h in range(8):
                        tgt_head = f'layer{l}head{h}'
                        run_attributions(f,df_test,tgt_head,best_EDC_dir,g,m,'attn')
                '''
if __name__ == "__main__":
    
    attribution_loci_pipeline() 
