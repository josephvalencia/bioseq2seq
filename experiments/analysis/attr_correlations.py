import torch
import numpy as np
import sys
from utils import parse_config,build_EDA_file_list,load_CDS
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr,kendalltau,spearmanr, kstest, ttest_ind,ttest_rel
from math import copysign
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from os import listdir
from os.path import isfile, join, exists
import re
import glob

def calc_correlations(file_a,file_b,model_type,corr_mode,metric):

    storage = []
    file_a = np.load(file_a)
    file_b = np.load(file_b)
    
    for tscript,array_a in file_a.items():
        array_b = file_b[tscript] 
        corr,cos = similarity_scores(array_a,array_b,corr_mode) 
        entry = {'tscript' : tscript :, corr_mode : corr , 'cosine_sim' : cos, 'model' : model_type, 'Importance Metric' : metric, 'replicate' : replicate}
        storage.append(entry) 
    
    return storage

def similarity_scores(a,b,corr_mode):

    frob = np.linalg.norm(a-b,ord='fro') / np.linalg.norm(b,ord='fro')
    a_lof = a.min(axis=1) 
    b_lof = a.min(axis=1) 
    #print(f'% Frob distance = {frob}')  
    # median positionwise cosine similarity in the character dimension 
    cos_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(a),torch.from_numpy(b),dim=1).numpy()
    median_cos_sim = np.median(cos_sim)
    
    a = a.ravel()
    b = b.ravel()
    
    if corr_mode == 'pearson': 
        corr = pearsonr(a,b)
    elif corr_mode == 'spearman': 
        corr = spearmanr(a,b)
    elif corr_mode == 'kendall': 
        corr = kendalltau(a,b)
    else:
        raise ValueError("corr_mode must be one of [pearson,spearman,kendall]")
    
    return corr[0], median_cos_sim

def statistical_significance(df_a,df_b,metric):
    
    stat,pval = kstest(df_a[metric],df_b[metric])
    t_stat,pval_t = ttest_rel(df_a[metric],df_b[metric])
    mean_a = df_a[metric].mean() 
    mean_b = df_b[metric].mean() 
    print(f'{metric} {mean_a:.3f} vs. {mean_b:.3f} p-val (Paired T-test) = {pval_t:.2E}, p-val(KS) = {pval:.2E}')

def plot_metrics(df,savefile,corr_mode,height):

    sns.set_style("whitegrid")
    
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(12,height)) 
    g = sns.violinplot(data=df,y='Importance Metric',x=corr_mode,hue='model',ax=ax1,cut=0)
    #g.legend_.remove()
    closest_min = df[corr_mode].min()-0.1 
    ax1.set_xlim(closest_min,1.0)
    ax1.set_ylabel('')
    if corr_mode == 'kendall': 
        ax1.set_xlabel(r'Kendall $\tau$ ')
    elif corr_mode == 'spearman': 
        ax1.set_xlabel(r'Spearman $\rho$ ')
    else: 
        ax1.set_xlabel(r'Pearson $\rho$ ')
    
    g = sns.violinplot(data=df,y='Importance Metric',x='cosine_sim',hue='model',ax=ax2,cut=0,split=False)
    #g.legend_.remove()
    closest_min = df['cosine_sim'].min()-0.1 
    ax2.set_xlim(closest_min,1.0)
    ax2.set_ylabel('')
    ax2.set_xlabel(r'Median positionwise cosine sim.')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close() 


def calculate_metrics(ism_file,mdig_file_dict,saliency_file,onehot_file,ig_file,test_cds,model_type,replicate,corr_mode,mrna_zoom=False):

    storage = []
    is_coding = lambda x: x.startswith('XM') or x.startswith('NM') 
    
    for tscript,ism in ism_file.items():
        cds_loc = [int(x) for x in test_cds[tscript].split(':')] 
        s,e = tuple(cds_loc) 
        labels = ['A','C','G','T']
        metric = 'Importance Metric'    
        
        ism = ism_file[tscript]
        grad = saliency_file[tscript]
        onehot = onehot_file[tscript]
        ig = ig_file[tscript]

        if onehot.shape != grad.shape:
            shorter = min(onehot.shape[0],grad.shape[0])
            onehot = onehot[:shorter,:]
            grad = grad[:shorter,:]
       
        taylor = grad - onehot*grad 
        taylor = taylor[:,2:6]
        modified_IG = -(ig - onehot*ig)
        modified_IG = modified_IG[:,2:6]
        if mrna_zoom:
            window_start = -18
            window_end = 60
            if is_coding(tscript) and s>= max(0,-window_start) and s+window_end < onehot.shape[0]:
                domain = list(range(window_start,window_end))
                cds_ism = ism[s+window_start:s+window_end,:]
                cds_taylor = taylor[s+window_start:s+window_end,:]
                cds_ig = modified_IG[s+window_start:s+window_end,:]
                for alpha,mdig_file in mdig_file_dict.items():  
                    mdig = mdig_file[tscript] 
                    cds_mdig = mdig[s+window_start:s+window_end,:]
                    corr,cos = similarity_scores(cds_mdig,cds_ism,corr_mode) 
                    entry = {'tscript' : tscript, corr_mode : corr , 'cosine_sim' : cos, metric : f'MDIG-{alpha}', 'model' : model_type ,'replicate' : replicate}
                    storage.append(entry)
                corr,cos = similarity_scores(cds_taylor,cds_ism,corr_mode) 
                entry = {'tscript' : tscript, corr_mode : corr , 'cosine_sim' : cos, metric : 'Taylor', 'model' : model_type, 'replicate' : replicate}
                storage.append(entry)
                corr,cos = similarity_scores(cds_ig,cds_ism,corr_mode) 
                entry = {'tscript' : tscript, corr_mode : corr , 'cosine_sim' : cos, metric : 'IG', 'model' : model_type, 'replicate' : replicate}
                storage.append(entry)
        else: 
            for alpha,mdig_file in mdig_file_dict.items(): 
                mdig = mdig_file[tscript] 
                corr,cos = similarity_scores(mdig,ism,corr_mode) 
                entry = {'tscript' : tscript, corr_mode : corr , 'cosine_sim' : cos, metric : f'MDIG-{alpha}', 'model' : model_type, 'replicate' : replicate}
                storage.append(entry)
            corr,cos = similarity_scores(taylor,ism,corr_mode) 
            entry = {'tscript' : tscript, corr_mode : corr , 'cosine_sim' : cos, metric : 'Taylor', 'model' : model_type, 'replicate' : replicate}
            storage.append(entry)
            corr,cos = similarity_scores(modified_IG,ism,corr_mode) 
            entry = {'tscript' : tscript, corr_mode : corr , 'cosine_sim' : cos, metric : 'IG', 'model' : model_type, 'replicate' : replicate}
            storage.append(entry)
    
    return storage

def ism_agreement(prefix,models1,models2,corr_mode,parent='.'):
    ''' compare [MDIG,IG,Taylor] with ISM for a given model replicate '''
    
    sns.set_style("whitegrid")
    
    #test_file = '/home/bb/valejose/home/bioseq2seq/data/mammalian_200-1200_test_nonredundant_80.csv'
    test_file = '/home/bb/valejose/home/bioseq2seq/data/mammalian_200-1200_val_nonredundant_80.csv'
    test_cds = load_CDS(test_file)
    storage = [] 
    
    with open(models1) as inFile:
        model_list1 = [x.rstrip().replace('/','').replace('.pt','') for x in inFile.readlines()]
    with open(models2) as inFile:
        model_list2 = [x.rstrip().replace('/','').replace('.pt','') for x in inFile.readlines()]
    
    mdig_alphas = [0.10,0.25,0.50,0.75,1.00] 
    for m in model_list1:
        ism_file = np.load(f'{parent}/{m}/{prefix}.ISM.npz') 
        onehot_file = np.load(f'{parent}/{m}/{prefix}.onehot.npz') 
        saliency_file = np.load(f'{parent}/{m}/{prefix}.grad.npz') 
        ig_file = np.load(f'{parent}/{m}/{prefix}.IG.npz') 
        mdig_file_dict =  {a : np.load(f'{parent}/{m}/{prefix}.MDIG.max_{a:.2f}.npz') for a in mdig_alphas} 
        results = calculate_metrics(ism_file,mdig_file_dict,saliency_file,onehot_file,ig_file,test_cds,'bioseq2seq',m,corr_mode,mrna_zoom=True) 
        storage.extend(results) 
    for m in model_list2:
        ism_file = np.load(f'{parent}/{m}/{prefix}.ISM.npz') 
        onehot_file = np.load(f'{parent}/{m}/{prefix}.onehot.npz') 
        saliency_file = np.load(f'{parent}/{m}/{prefix}.grad.npz') 
        ig_file = np.load(f'{parent}/{m}/{prefix}.IG.npz') 
        mdig_file_dict =  {a : np.load(f'{parent}/{m}/{prefix}.MDIG.max_{a:.2f}.npz') for a in mdig_alphas} 
        results = calculate_metrics(ism_file,mdig_file_dict,saliency_file,onehot_file,ig_file,test_cds,'EDC',m,corr_mode,mrna_zoom=True) 
        storage.extend(results) 
    
    df = pd.DataFrame(storage)
    print('ISM AGREEMENT') 
    print(df)
    summarize(df,corr_mode)
    plot_metrics(df,'full_ISM_agreement.svg',corr_mode,height=6.0)
    return df

def self_agreement(prefix,models1,models2,corr_mode,parent='.'):
  
    ''' compare [MDIG,ISM,grad] across model replicates'''
     
    storage = [] 
    with open(models1) as inFile:
        model_list1 = [x.rstrip().replace('/','').replace('.pt','') for x in inFile.readlines()]
    with open(models2) as inFile:
        model_list2 = [x.rstrip().replace('/','').replace('.pt','') for x in inFile.readlines()]
    
    group1_combs = list(combinations(model_list1,2))
    group2_combs = list(combinations(model_list2,2))

    # pairwise comparisons all bioseq2seq replicates
    for metric in ['MDIG.max_0.50','MDIG.max_0.25','ISM','grad','IG']:
        for a,b in group1_combs:
            data_a = f'{parent}{a}/{prefix}.{metric}.npz' 
            data_b = f'{parent}{b}/{prefix}.{metric}.npz' 
            if exists(data_a) and exists(data_b):
                print(f'comparing {metric} in {a} and {b}') 
                results = calc_correlations(data_a,data_b,'bioseq2seq',corr_mode,metric.replace('.max_','-')) 
                storage.extend(results)

    for metric in ['MDIG.max_0.50','MDIG.max_0.25','ISM','grad','IG']:
        # pairwise comparisons all EDC replicates
        for a,b in group2_combs:
            data_a = f'{parent}{a}/{prefix}.{metric}.npz' 
            data_b = f'{parent}{b}/{prefix}.{metric}.npz' 
            if exists(data_a) and exists(data_b): 
                print(f'comparing {metric} in {a} and {b}') 
                results = calc_correlations(data_a,data_b,'EDC',corr_mode,metric.replace('.max_','-')) 
                storage.extend(results)

    df = pd.DataFrame(storage)
    print('SELF AGREEMENT')
    print(df)
    summarize(df,corr_mode) 
    plot_metrics(df,'full_self_agreement.svg',corr_mode,height=4.5) 
    return df

def summarize(df,corr_mode):
    
    by_type_mean = df[['Importance Metric','model',corr_mode]].groupby(['Importance Metric','model']).median()
    by_type_std = df[['Importance Metric','model',corr_mode]].groupby(['Importance Metric','model']).std()
    for_latex_mean = by_type_mean.applymap('{:.3f}'.format)
    for_latex_std = by_type_std.applymap('{:.3f}'.format)
    for_latex = for_latex_mean.add(' $\pm$ ').add(for_latex_std)
    col_format = 'c'*len(for_latex.columns)
    table = for_latex.style.to_latex(column_format=f'|{col_format}|',hrules=True)
    print(table)

if __name__ == "__main__":
    
    bio_top1 = "experiments/scripts/top1_bioseq2seq_models.txt"
    EDC_top1 = "experiments/scripts/top1_EDC-large_models.txt"
    ism_df = ism_agreement('verified_val_RNA.PC.1',bio_top1,EDC_top1,'pearson',parent="experiments/output/")
    self_df = self_agreement('verified_val_RNA.PC.1',bio_top1,EDC_top1,'pearson',parent="experiments/output/")
