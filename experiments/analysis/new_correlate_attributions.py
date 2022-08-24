from scipy import stats
import orjson
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import combinations
from os import listdir
from os.path import isfile, join
import re

def calc_correlations(file_a,file_b,mode='spearman',metric='normed_attr'):

    storage = []
    fh_a = open(file_a)
    fh_b = open(file_b)
    sig = 0
    nan = 0
    non_matching = 0
    for la,lb in zip(fh_a.readlines(),fh_b.readlines()):
        fields_a = orjson.loads(la)
        fields_b = orjson.loads(lb)
       
        array_a = np.asarray([float(x) for x in fields_a[metric]]) 
        array_b = np.asarray([float(x) for x in fields_b[metric]]) 

        # ensure comparing the same sample
        src_a = fields_a['src']
        src_b = fields_b['src']
        assert src_a == src_b
    
        pval = 0.05
        
        if mode == 'spearman':
            result = stats.spearmanr(array_a,array_b)
        elif mode == 'pearson':
            if array_a.shape[0] > 2:
                result = stats.pearsonr(array_a,array_b)
        elif mode == 'kendall':
            result = stats.kendalltau(array_a,array_b)
        storage.append(result[0])
        if result[1] <= pval:
            sig +=1

    print(f'# non-matching = {non_matching}, len(storage) = {len(storage)}')
    return storage,sig


def compare(regex1,regex2,corr_mode,metric,parent='.'):
     
    onlyfiles = [join(parent,f) for f in listdir(parent) if isfile(join(parent, f))]
    group1 = [f for f in onlyfiles if re.search(regex1,f) if not None]
    group2 = [f for f in onlyfiles if re.search(regex2,f) if not None]
   
    group1_combs = list(combinations(group1,2))
    group2_combs = list(combinations(group2,2))

    group1_storage = []
    group2_storage = []
  
    group1_sig = 0
    for a,b in group1_combs:
        print(f'comparing {a} and {b}')
        corrs,sig = calc_correlations(a,b,mode=corr_mode,metric=metric)
        group1_storage.extend(corrs)
        group1_sig+=sig 
    
    print('bioseq2seq')
    num_comparisons = len(corrs)*len(group1_combs)
    print(f'Signification fraction = {group1_sig / num_comparisons}')
    print_corr(group1_storage,corr_mode)
    
    group2_sig = 0
    for a,b in group2_combs:
        print(f'comparing {a} and {b}')
        corrs,sig = calc_correlations(a,b,mode=corr_mode,metric=metric)
        group2_storage.extend(corrs)
        group2_sig+=sig
    
    print('EDC')
    print(f'Signification fraction = {group2_sig / num_comparisons}')
    print_corr(group2_storage,corr_mode)

    attr = regex1.split('.')[-1]

    _, bins, _ = plt.hist(group1_storage, bins=50,range=[-1,1],label='bioseq2seq',density=True)
    _ = plt.hist(group2_storage, bins=bins, alpha=0.5,label='EDC',density=True)
    dest = f'{attr}_{corr_mode}_{metric}_correlation_hist.svg'
    plt.legend(loc='upper left')
    plt.savefig(dest)
    plt.close()

def print_corr(storage,mode):

    storage = np.asarray(storage)
    storage = storage[~np.isnan(storage)]
    
    if mode == 'spearman':
        print('Spearman rho = {} +- {}'.format(np.mean(storage),np.std(storage)))
    elif mode == 'pearson':
        print('Pearson R = {} +- {}'.format(np.mean(storage),np.std(storage)))
    elif mode == 'kendall':
        print('Kendall tau = {} +- {}'.format(np.mean(storage),np.std(storage)))

if __name__ == "__main__":

    # global attr
    regex1 = 'bioseq2seq_\d.PC.saliency.rank_\d'
    regex2 = 'EDC_\d.PC.saliency.rank_\d'
    
    #compare(regex1,regex2,'spearman','normed_attr')
    #compare(regex1,regex2,'pearson','normed_attr')
    compare(regex1,regex2,'spearman','summed_attr_PC',parent=".")
    #compare(regex1,regex2,'pearson','summed_attr_PC')

    '''
    # local attr
    regex1 = 'bioseq2seq_\d.<PC>.saliency'
    regex2 = 'EDC_\d.<PC>.saliency'
    compare(regex1,regex2,'spearman','normed_attr')
    compare(regex1,regex2,'pearson','normed_attr')
    compare(regex1,regex2,'spearman','summed_attr')
    compare(regex1,regex2,'pearson','summed_attr')
    '''
