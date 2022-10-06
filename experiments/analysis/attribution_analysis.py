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
from scipy.signal import convolve
from scipy.spatial.distance import jensenshannon
from collections import defaultdict
from IPython.display import Image

from Bio.Data import CodonTable
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

from utils import parse_config, add_file_list, getLongestORF, get_CDS_start

def select_index(array,mode,excluded=None):

        # find various indices of interest
        if mode == 'max':
            return np.argmax(array).tolist()
        elif mode == 'min':
            return np.argmin(array).tolist()
        elif mode == 'rolling-abs':
            rolling_mean = convolve(np.abs(array),np.ones(21)) / 21
            max_rolling_abs = np.argmax(rolling_mean) + 10
            return max_rolling_abs.tolist()
        elif mode == 'random':
            if excluded is None:
                return random.randrange(array.shape[-1])
            else:
                excluded_idx = [excluded+x for x in range(-20,20)]
                other_idx = [x for x in range(array.shape[-1]) if x not in excluded_idx]
                return random.sample(other_idx,1)[0]
        else:
            raise ValueError("Not a strategy")

def top_indices(saved_file,positive_topk_file,negative_topk_file,groups,metrics,mode="attn",head_idx=0):
    
    df_storage = []
    negative_storage = []
    positive_storage = []

    out_name = saved_file.split(".")[0]
    loaded = np.load(saved_file)

    for tscript,array in loaded.items():
        
        L = array.shape[-1]
        array = np.asarray(array)
        if mode == 'attn':
            array = array[head_idx,:]
        name = tscript + "_" + mode
        coding = True if (tscript.startswith('XM_') or tscript.startswith('NM_')) else False 
        both_storage = [positive_storage,negative_storage]

        for i,(g,m,dataset) in enumerate(zip(groups,metrics,both_storage)): 
            result = None
            # partition based on args
            if (g == 'PC' and coding) or (g == 'NC' and not coding):
                if i==1 and m=='random' and groups[0] == groups[1]:
                    excluded = positive_storage[-1][1]
                    selected = select_index(array,m,excluded=excluded) 
                    result = (tscript,selected)
                else:
                    selected = select_index(array,m) 
                    result = (tscript,selected)
            if result is not None:
                dataset.append(result) 
    
    #positive_storage,negative_storage = **both_storage
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

def run_attributions(saved_file,df,parent_dir,groups,metrics,mode="attn",layer_idx=0,head_idx=0):

    if mode == "attn":
        attr_name = f'layer{layer_idx}head{head_idx}'
    else:
        attr_name = os.path.split(saved_file)[1]
        fields = attr_name.split('.')
        attr_name = f'{fields[2]}_{fields[1]}'

    prefix = f'{parent_dir}/{attr_name}/' 
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    print(prefix)
    # results files
    positive_indices_file = prefix+"positive_topk_idx.txt"
    negative_indices_file = prefix+"negative_topk_idx.txt"
    positive_motifs_file = prefix+"positive_motifs.fa"
    negative_motifs_file = prefix +"negative_motifs.fa"
    hist_file = prefix+"pos_hist.svg"
    top_indices(saved_file,positive_indices_file,negative_indices_file,groups,metrics,mode=mode,head_idx=head_idx)
    top_k_to_substrings(positive_indices_file,positive_motifs_file,df)
    top_k_to_substrings(negative_indices_file,negative_motifs_file,df)
    get_positional_bias(positive_indices_file,negative_indices_file,df,hist_file,groups,metrics)

def get_positional_bias(positive_indices_file_file,negative_indices_file_file,df_data,hist_file,groups,metrics):

    storage = []
    pos_name = f'{groups[0]}-{metrics[0]}' 
    neg_name = f'{groups[1]}-{metrics[1]}' 
    pos = pd.read_csv(positive_indices_file_file,names=['ID','start'])
    neg = pd.read_csv(negative_indices_file_file,names=['ID','start'])
    pos['class'] = len(pos)*[pos_name]
    neg['class'] = len(neg)*[neg_name]
    df_attr = pd.concat([pos,neg])
    df_data['cds_start'] = [get_CDS_start(cds,seq) for cds,seq in zip(df_data['CDS'].values.tolist(),df_data['RNA'].values.tolist())]
    df = pd.merge(df_attr,df_data,on='ID')
    df['rel_start'] = df['start'] - df['cds_start']-1
    df = df.drop(columns=df.columns.difference(['class','rel_start']))
    
    bins = np.arange(-750,1200,10)
    g = sns.displot(data=df,x='rel_start',col='class',kind='hist',hue_order=[pos_name,neg_name],stat='density',bins=bins,element='step')
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
    output_dir  =  f'results_{config_prefix}'
    attr_dir  =  f'{output_dir}/attr'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # make subdir for attribution loci results  
    if not os.path.isdir(attr_dir):
        os.mkdir(attr_dir)

    # load attribution files from config
    best_BIO_EDA = add_file_list(args.best_BIO_EDA,'layers')
    best_EDC_EDA = add_file_list(args.best_EDC_EDA,'layers')
    best_BIO_grad_PC = args.best_BIO_grad_PC
    best_EDC_grad_PC = args.best_EDC_grad_PC
    best_BIO_grad_NC = args.best_BIO_grad_NC
    best_EDC_grad_NC = args.best_EDC_grad_NC
    
    groups = [['PC','NC'],['PC','PC'],['NC','NC']]
    cross_metrics = [['max','max'],['max','min'],['min','max'],['rolling-abs','rolling-abs'],['random','random']]
    same_metrics = [['max','min'],['max','random'],['min','random'],['rolling-abs','random']]
    
    for i,g in enumerate(groups):
        metrics = same_metrics if i>0 else cross_metrics
        for m in metrics:
            # g[0] and g[1] are transcript type for pos and neg sets
            # m[0] and m[1] are loci of interest for pos and neg sets
            a = f'{g[0]}-{m[0]}'
            b = f'{g[1]}-{m[1]}'
            # ensure comparison groups are different 
            if a != b:
                trial_name = f'{a}_{b}'
                # build directories
                best_EDC_dir = f'{attr_dir}/best_EDC_{trial_name}'
                if not os.path.isdir(best_EDC_dir):
                    os.mkdir(best_EDC_dir)
                best_BIO_dir = f'{attr_dir}/best_seq2seq_{trial_name}'
                if not os.path.isdir(best_BIO_dir):
                    os.mkdir(best_BIO_dir)
                
                # run all IG bases for both models 
                #run_attributions(best_BIO_grad_PC,df_test,best_BIO_dir,g,m,'inputXgrad')
                run_attributions(best_EDC_grad_PC,df_test,best_EDC_dir,g,m,'inputXgrad')
                #run_attributions(best_BIO_grad_NC,df_test,best_BIO_dir,g,m,'inputXgrad')
                run_attributions(best_EDC_grad_NC,df_test,best_EDC_dir,g,m,'inputXgrad')
                
                # run all EDA layers for both models
                '''
                for l,f in enumerate(best_BIO_EDA['path_list']):
                    for h in range(8):
                        run_attributions(f,df_test,best_BIO_dir,g,m,'attn',layer_idx=l,head_idx=h)
                '''
                for l,f in enumerate(best_EDC_EDA['path_list']):
                    for h in range(8):
                        run_attributions(f,df_test,best_EDC_dir,g,m,'attn',layer_idx=l,head_idx=h)

if __name__ == "__main__":
    
    attribution_loci_pipeline() 
