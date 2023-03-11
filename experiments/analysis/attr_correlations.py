import torch
import numpy as np
import sys,os
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
from utils import parse_config, setup_fonts

def calc_correlations(file_a,file_b,model_type,corr_mode,metric):

    storage = []
    onhot_file_a = None 
    onhot_file_b = None 
    if metric == 'grad':
        onehot_file_a = np.load(file_a.replace('grad','onehot')) 
        onehot_file_b = np.load(file_b.replace('grad','onehot')) 
        metric = 'Taylor' 
     
    file_a = np.load(file_a)
    file_b = np.load(file_b)
    count = 0 
    for tscript,array_a in file_a.items():
        array_b = file_b[tscript] 
        if metric == 'grad':
            onehot_a = onehot_file_a[tscript]
            onehot_b = onehot_file_b[tscript]
            taylor_a = array_a - onehot_a*array_a 
            taylor_b = array_b - onehot_b*array_b 
            array_a = taylor_a[:,2:6]
            array_b = taylor_b[:,2:6]
        corr,cos,mae,sign_match_pct = similarity_scores(array_a,array_b,corr_mode) 
        entry = {'tscript' : tscript , corr_mode : corr , 'cosine_sim' : cos, 'MAE' : mae, 'sign_match_pct' : sign_match_pct, 'Model' : model_type, 'Mutation method' : metric}
        ''' 
        if metric == 'ISM' and model_type == 'EDC':
            a = array_a.ravel()
            b = array_b.ravel()
            both_nonzero = (a != 0.0) & (b != 0.0)
            a = a[both_nonzero]
            original_size = a.size 
            b = b[both_nonzero]
            pearson = pearsonr(a,b)[0]
            spearman = spearmanr(a,b)[0]
            g = sns.jointplot(x=a,y=b,kind='reg')
            full = f'Full\nPearson={pearson:.3f}\nSpearman={spearman:.3f}\nn={a.size}/{original_size} = {a.size/original_size:.3f}'
            non_outliers = both_not_outlier(a,b)
            a = a[non_outliers]
            b = b[non_outliers]
            pearson_partial = pearsonr(a,b)[0]
            spearman_partial = spearmanr(a,b)[0]
            partial = f'Inside (Q1-1.5IQR,Q3+1.5IQR)\nPearson={pearson_partial:.3f}\nSpearman={spearman_partial:.3f}\nn={a.size}/{original_size} = {a.size/original_size:.3f}'
            plt.text(0.05, 0.9,full,
                    transform=g.ax_joint.transAxes)
            plt.text(0.50, 0.9,partial,
                    transform=g.ax_joint.transAxes)
            plt.savefig(f'sanity_check_scatter_{model_type}_{tscript}.svg')
            plt.xlabel('Replicate 1') 
            plt.ylabel('Replicate 2') 
            plt.title(tscript)
            plt.close()
            count +=1
        
        if count == 10:
            quit()
        '''
        storage.append(entry) 
    
    return storage

def both_not_outlier(a,b):

    a_Q1 = np.quantile(a,0.25)
    a_Q3 = np.quantile(a,0.75)
    b_Q1 = np.quantile(b,0.25)
    b_Q3 = np.quantile(b,0.75)
    a_IQR = a_Q3 - a_Q1
    b_IQR = b_Q3 - b_Q1
    a_not_outlier = (a > a_Q1 - 1.5*a_IQR) & (a < a_Q3 + 1.5*a_IQR)
    b_not_outlier = (b > b_Q1 - 1.5*b_IQR) & (b < b_Q3 + 1.5*b_IQR)
    return  a_not_outlier & b_not_outlier

def similarity_scores(a,b,corr_mode):

    frob = np.linalg.norm(a-b,ord='fro') / np.linalg.norm(b,ord='fro')
    a_lof = a.min(axis=1) 
    b_lof = a.min(axis=1) 
    # median positionwise cosine similarity in the character dimension 
    cos_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(a),torch.from_numpy(b),dim=1).numpy()
    median_cos_sim = np.median(cos_sim)
    
    a = a.ravel()
    b = b.ravel()
    
    # exclude the zeros for the endogeneous characters 
    both_nonzero = (a != 0.0) & (b != 0.0)
    a = a[both_nonzero]
    b = b[both_nonzero]
   
    mae = (np.abs(a-b) / np.abs(b) ).mean()  #/ (np.linalg.norm(b,ord=1) / b.size) 
    #rmse = np.sqrt((a-b)**2).mean()) 
    
    sign_match_pct = np.count_nonzero(np.sign(a) == np.sign(b)) / b.size

    if corr_mode == 'pearson': 
        corr = pearsonr(a,b)
    elif corr_mode == 'spearman':
        corr = spearmanr(a,b)
    elif corr_mode == 'kendall': 
        corr = kendalltau(a,b)
    else:
        raise ValueError("corr_mode must be one of [pearson,spearman,kendall]")
    
    return corr[0], median_cos_sim, mae, sign_match_pct

def statistical_significance(df_a,df_b,metric):
    
    stat,pval = kstest(df_a[metric],df_b[metric])
    t_stat,pval_t = ttest_rel(df_a[metric],df_b[metric])
    mean_a = df_a[metric].mean() 
    mean_b = df_b[metric].mean() 
    print(f'{metric} {mean_a:.3f} vs. {mean_b:.3f} p-val (Paired T-test) = {pval_t:.2E}, p-val(KS) = {pval:.2E}')

def plot_metrics(df,savefile,corr_mode,height,xlabel=None,y='Mutation method',order=None,show_legend=False):
    
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(7.5,height)) 
    g = sns.violinplot(data=df,y=y,x=corr_mode,hue='Model',ax=ax1,cut=0,order=order,hue_order=['bioseq2seq','EDC'])
    if show_legend: 
        #sns.move_legend(g,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=2)
        sns.move_legend(g,loc="upper right",bbox_to_anchor=(0.0,1.0),ncol=1)
    else:
        g.legend_.remove()
    sns.despine() 
    
    closest_min = df[corr_mode].min()-0.1 
    ax1.set_xlim(closest_min,1.0)
    
    is_multiple = len(df[y].unique()) > 1

    if is_multiple:
        ax1.set_ylabel('Method')
    else:
        ax1.set_ylabel('')
        ax1.set_yticks([])

    if corr_mode == 'spearman': 
        if xlabel is not None:
            ax1.set_xlabel(xlabel + '\n'+r'(Spearman $\rho$)',multialignment='center')
    elif corr_mode == 'kendall': 
        if xlabel is not None:
            ax1.set_xlabel(xlabel + '\n'+r'(Kendall $\tau$)',multialignment='center')
    else: 
        if xlabel is not None:
            ax1.set_xlabel(xlabel+'\n'+'(Pearson r)',multialignment='center')
    g = sns.violinplot(data=df,y=y,x='cosine_sim',hue='Model',ax=ax2,cut=0,split=False,order=order,hue_order=['bioseq2seq','EDC'])
    sns.despine() 
    g.legend_.remove()
    closest_min = df['cosine_sim'].min()-0.1 
    ax2.set_xlim(closest_min,1.0)
    
    if is_multiple:
        ax2.set_ylabel('Method')
    else:
        ax2.set_ylabel('')
        ax2.set_yticks([])
    ax2.set_xlabel(xlabel+'\n'+r'(Median position-wise cosine sim.)',multialignment='center')
    
    plt.tight_layout()
    print(corr_mode,savefile) 
    plt.savefig(savefile)
    plt.close() 

def calculate_metrics(ism_file,metric_file_dict,onehot_file,test_cds,model_type,replicate,corr_mode,mrna_zoom=False):

    storage = []
    is_coding = lambda x: x.startswith('XM') or x.startswith('NM') 
    
    for tscript,ism in ism_file.items():
        cds_loc = [int(x) for x in test_cds[tscript].split(':')] 
        s,e = tuple(cds_loc) 
        
        if mrna_zoom:
            window_start = -18
            window_end = 60
            if is_coding(tscript) and s>= max(0,-window_start) and s+window_end < onehot.shape[0]:
                domain = list(range(window_start,window_end))
                cds_ism = ism[s+window_start:s+window_end,:]
                for metric,metric_file in metric_file_dict.items():  
                    array = metric_file[tscript] 
                    if metric == 'Taylor' or metric == 'IG':
                        # subtract gradient from endogenous character 
                        array = array - onehot_file[tscript] * array
                        if metric == 'IG':
                            # negate IG for similar wildtype>variant interpretation as IG
                            array = -array
                        array = array[:,2:6]
                    cds_metric = array[s+window_start:s+window_end,:]
                    corr,cos,mae,sign_match_pct = similarity_scores(cds_metric,cds_ism,corr_mode) 
                    entry = {'tscript' : tscript,
                            corr_mode : corr ,
                            'cosine_sim' : cos,
                            'MAE' : mae,
                            'Mutation method' : metric,
                            'sign_match_pct' : sign_match_pct,
                            'Model' : model_type ,
                            'replicate' : replicate}
                    storage.append(entry)
        else: 
            for metric,metric_file in metric_file_dict.items(): 
                array = metric_file[tscript] 
                if metric == 'Taylor' or metric == 'IG':
                    # subtract gradient from endogenous character 
                    array = array - onehot_file[tscript] * array
                    if metric == 'IG':
                        # negate IG for similar wildtype>variant interpretation as IG
                        array = -array
                    array = array[:,2:6]
                corr,cos,mae,sign_match_pct = similarity_scores(array,ism,corr_mode) 
                entry = {'tscript' : tscript,
                        corr_mode : corr ,
                        'cosine_sim' : cos,
                        'MAE' : mae,
                        'Mutation method': metric,
                        'sign_match_pct' : sign_match_pct,
                        'Model' : model_type ,
                        'replicate' : replicate}
                storage.append(entry)
    return storage

def maybe_load(metric_file_dict,metric,filename):
    
    if os.path.exists(filename):
        metric_file_dict[metric] = np.load(filename) 
    return metric_file_dict

def load_all_replicates(models):
    model_list = []
    with open(models) as inFile:
        for x in inFile.readlines():
            model_dir = x.rstrip().replace('/','').replace('.pt','')
            model_list.append(model_dir)
    return model_list

def ism_agreement(prefix,models1,models2,corr_mode,test_csv,parent='.'):
    ''' compare [MDIG,IG,Taylor] with ISM for a given model replicate '''
    
    sns.set_style(style="whitegrid",rc={'font.family' : ['Helvetica']})
    test_cds = load_CDS(test_csv)
    storage = [] 
    
    model_list1 = load_all_replicates(models1)
    model_list2 = load_all_replicates(models2)
    
    mdig_alphas = [0.10,0.25,0.50,0.75,1.00] 
    for model_type, model_list in zip(['bioseq2seq','EDC'],[model_list1,model_list2]): 
        for m in model_list:
            ism_file = np.load(f'{parent}/{m}/{prefix}.ISM.npz') 
            onehot_file = np.load(f'{parent}/{m}/{prefix}.onehot.npz') 
            # add optional metrics
            metric_file_dict = {} 
            for a in mdig_alphas:
                metric_file_dict = maybe_load(metric_file_dict,f'MDIG-{a:.2f}',
                                        f'{parent}/{m}/{prefix}.MDIG.max_{a:.2f}.npz') 
            metric_file_dict = maybe_load(metric_file_dict,'Taylor',
                                    f'{parent}/{m}/{prefix}.grad.npz') 
            metric_file_dict = maybe_load(metric_file_dict,'IG',
                                    f'{parent}/{m}/{prefix}.IG.npz') 
            results = calculate_metrics(ism_file,metric_file_dict,
                                        onehot_file,test_cds,model_type,
                                        m,corr_mode,mrna_zoom=False) 
            storage.extend(results) 
    df = pd.DataFrame(storage)
    print('ISM AGREEMENT') 
    print(df) 
    averaged = average_per_transcript(df)
    sns.set_style(style="whitegrid",rc={'font.family' : ['Helvetica']})
    # pairwise comparisons all bioseq2seq replicates
    order = ['MDIG-0.10','MDIG-0.25','MDIG-0.50','MDIG-0.75','MDIG-1.00','Taylor','IG']
    order = [x for x in order if x in metric_file_dict]
    xlabel = 'Intra-replicate agreement with ISM'
    plot_metrics(averaged,'full_ISM_agreement.svg',corr_mode,height=4.5,xlabel=xlabel,order=order)
    return averaged,df

def average_per_transcript(df):
    #return df.groupby(['Mutation method','Model','tscript','class_type']).mean().reset_index()
    return df.groupby(['Mutation method','Model','tscript']).mean().reset_index()

def self_agreement(prefix,models1,models2,corr_mode,parent='.'):
  
    ''' compare [MDIG,ISM,grad] across model replicates'''
     
    storage = [] 
    model_list1 = load_all_replicates(models1)
    model_list2 = load_all_replicates(models2)
    group1_combs = list(combinations(model_list1,2))
    group2_combs = list(combinations(model_list2,2))
    
    # pairwise comparisons all bioseq2seq replicates
    all_metrics = ['MDIG.max_0.10','MDIG.max_0.25','MDIG.max_0.50','MDIG.max_0.75','MDIG.max_1.00','ISM','grad','IG']
    for metric in all_metrics:
        for a,b in group1_combs:
            data_a = f'{parent}/{a}/{prefix}.{metric}.npz' 
            data_b = f'{parent}/{b}/{prefix}.{metric}.npz' 
            if exists(data_a) and exists(data_b):
                print(f'comparing {metric} in {a} and {b}') 
                results = calc_correlations(data_a,data_b,'bioseq2seq',corr_mode,metric.replace('.max_','-')) 
                storage.extend(results)

    for metric in all_metrics:
        # pairwise comparisons all EDC replicates
        for a,b in group2_combs:
            data_a = f'{parent}/{a}/{prefix}.{metric}.npz' 
            data_b = f'{parent}/{b}/{prefix}.{metric}.npz' 
            if exists(data_a) and exists(data_b): 
                print(f'comparing {metric} in {a} and {b}') 
                results = calc_correlations(data_a,data_b,'EDC',corr_mode,metric.replace('.max_','-')) 
                storage.extend(results)

    df = pd.DataFrame(storage)
    print('SELF AGREEMENT')
    averaged = average_per_transcript(df) 
    print(averaged) 
    averaged_remainder = averaged[averaged['Mutation method'] != 'ISM'] 
    averaged_ism = averaged[averaged['Mutation method'] == 'ISM'] 
   
    '''
    averaged_ism_bioseq2seq = averaged_ism[averaged_ism['Model'] == 'bioseq2seq']
    median = averaged_ism_bioseq2seq['pearson'].median()
    examples = [] 
    for class_type, group in averaged_ism_bioseq2seq.groupby('class_type'):
        pearson_diff = [(x,abs(y - median),y) for x,y in zip(group['tscript'],group['pearson'])]
        closeness_to_median = sorted(pearson_diff,key = lambda x : x[1])
        print(class_type,closeness_to_median[:5],median)
        for tscript,diff,pearson in closeness_to_median[:5]:
            examples.append(tscript)
    print(examples)
    '''
    sns.set_style(style="whitegrid",rc={'font.family' : ['Helvetica']})
    order = ['MDIG-0.10','MDIG-0.25','MDIG-0.50','MDIG-0.75','MDIG-1.00','Taylor','IG']
    xlabel = 'Inter-replicate agreement'
    plot_metrics(averaged_ism,'full_ISM_only_self_agreement.svg',corr_mode,height=1.5,xlabel=f'ISM inter-replicate agreement',order=None,show_legend=True) 
    plot_metrics(averaged_remainder,'full_self_agreement.svg',corr_mode,height=4.5,xlabel=xlabel,order=order)
    return averaged, df 

def plot_summary_scatter(combined):

    sns.set_style(style="ticks",rc={'font.family' : ['Helvetica']})
    plt.figure(figsize=(4.5,3.5))
    g = sns.scatterplot(combined,x ='pearson_self_median',y='pearson_ISM_median',hue='Mutation method',style='Model',s=100)
    #sns.despine()
    sns.move_legend(g,loc="upper left",bbox_to_anchor=(0.00,1.0),fontsize=8)
    plt.tight_layout() 
    plt.xlabel('Inter-replicate agreement\n(median Pearson r)',multialignment='center')
    plt.ylabel('Intra-replicate agreement with ISM\n(median Pearson r)',multialignment='center')
    plt.savefig('attribution_summary_scatter.svg')
    plt.close()

if __name__ == "__main__":

    args,unknown_args = parse_config()
    setup_fonts()
    
    #test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    #prefix = f'verified_test_RNA.{args.reference_class}.{args.position}'
    test_file = os.path.join(args.data_dir,args.val_prefix+'.csv')
    prefix = f'verified_val_RNA.{args.reference_class}.{args.position}'
    consensus_ism_df, unreduced_ism_df = ism_agreement(prefix,args.all_BIO_replicates,args.all_EDC_replicates,
                                                        'pearson',test_file,'experiments/output') 
    consensus_self_df, unreduced_self_df = self_agreement(prefix,args.all_BIO_replicates,args.all_EDC_replicates,
                                                        'pearson','experiments/output') 
    
    grouped_ism = consensus_ism_df.groupby(['Mutation method','Model'])
    grouped_self = consensus_self_df.groupby(['Mutation method','Model'])
    self_medians = grouped_self.median()
    ism_medians = grouped_ism.median()
    combined = ism_medians.merge(self_medians,how='inner',on=['Mutation method','Model'],suffixes=['_ISM_median','_self_median'])
    plot_summary_scatter(combined.reset_index())
