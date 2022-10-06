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
                run_attributions(best_BIO_grad_PC,df_test,best_BIO_dir,g,m,'inputXgrad')
                run_attributions(best_EDC_grad_PC,df_test,best_EDC_dir,g,m,'inputXgrad')
                run_attributions(best_BIO_grad_NC,df_test,best_BIO_dir,g,m,'inputXgrad')
                run_attributions(best_EDC_grad_NC,df_test,best_EDC_dir,g,m,'inputXgrad')
                
                # run all EDA layers for both models
                for l,f in enumerate(best_BIO_EDA['path_list']):
                    for h in range(8):
                        run_attributions(f,df_test,best_BIO_dir,g,m,'attn',layer_idx=l,head_idx=h)
                for l,f in enumerate(best_EDC_EDA['path_list']):
                    for h in range(8):
                        run_attributions(f,df_test,best_EDC_dir,g,m,'attn',layer_idx=l,head_idx=h)

if __name__ == "__main__":
    
    attribution_loci_pipeline() 
