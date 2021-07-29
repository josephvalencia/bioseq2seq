import json
from scipy.stats import pearsonr,kendalltau,ttest_ind,spearmanr
import sys
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def multiple_correlation(attr_storage,attn_files,prefix):
    
    scores = []
    file_handlers = []
    attn_storage = defaultdict(list)
   
    # open layer files 
    for i in range(4):
        file_handlers.append(open(attn_files[i],'r'))
    
    # ingest all attention heads   
    for l,fh in enumerate(file_handlers):
        for line in fh:
            fields = json.loads(line)
            tscript_id = fields["TSCRIPT_ID"]
            #tscript_id = fields["ID"]
            for h in range(8):
                tgt_head = "layer{}head{}".format(l,h)
                #tgt_head = "summed_attr"
                attn = np.asarray(fields[tgt_head])
                attn_storage[tscript_id].append(attn)

    types = []
    max_attns = defaultdict(list)

    # match attention with attribution and process
    for tscript_id ,features in attn_storage.items():
        attr = [float(x) for x in attr_storage[tscript_id]]
        attr = np.asarray(attr) 
        features = np.vstack(features).T
        maximums = features.max(axis=0).tolist()
        for h,m in enumerate(maximums):
            max_attns[h].append(m)

        # perform multiple linear regression
        N,D = features.shape
        ones = np.ones(N).reshape(-1,1)
        t = np.arange(N).reshape(-1,1)
        features = np.concatenate([features,ones,t],axis=1)
        
        min_len = min(attr.shape[0],features.shape[0])
        attr = attr[:min_len]
        features = features[:min_len,:]
        beta_hat = np.linalg.pinv(features) @ attr
        beta_hat = beta_hat.reshape(-1,1)

        fitted = features @ beta_hat
        kendall = kendalltau(fitted,attr)

        # calculate multiple correlation coeff
        sum_sqr_tgt =  np.sum(attr)**2 / N 
        sum_sqr_regress = beta_hat.T @ features.T @ attr - sum_sqr_tgt 
        sum_sqr_total = np.dot(attr.T,attr) - sum_sqr_tgt
        R_sqr = sum_sqr_regress / sum_sqr_total
        R = np.sqrt(R_sqr)
        scores.append(R)

    for fh in file_handlers:
        fh.close()
  
    scores = np.asarray(scores)
    mu = np.nanmean(scores)
    std = np.nanstd(scores)
    median = np.nanmedian(scores)
    print(prefix,mu,std,median)

    textstr = '\n'.join((
    r'$\mu=%.3f$' % (mu, ),
    r'$\mathrm{median}=%.3f$' % (median, ),
    r'$\sigma=%.3f$' % (std, )))

    ax = sns.histplot(scores,binrange=(0,1),stat="probability",legend=False,alpha=0.5)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.075, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    ax.set(xlabel="Multiple Correlation (R)", ylabel='Density')
    corr_plot = prefix +"_corrs.svg"
    plt.savefig(corr_plot)
    plt.close()

def kendall_correlation(attr_storage,attn_file,tgt_head):

    scores = []
    num_sig = 0

    with open(attn_file) as inFile:

        for line in inFile:
            fields = json.loads(line)
            #tscript_id = fields["TSCRIPT_ID"]
            tscript_id = fields['ID'] 
            attn = fields[tgt_head]
            attr = attr_storage[tscript_id]

            if len(attr) < len(attn):
                attn = attn[:len(attr)]
            elif len(attn) < len(attr):
                attr = attr[:len(attn)]

            corr = kendalltau(attn,attr)
            scores.append(corr[0])
            
            if corr[1] <= 0.05:
                num_sig+=1

    mu = np.nanmean(scores)
    std = np.nanstd(scores)
    median = np.nanmedian(scores)
    print(mu,std,median,num_sig/len(attr_storage))

def strip_padding(src):
    
    for i in range(len(src)):
        if src[i] == "<":
            return i
    
    return len(src)

if __name__ == "__main__":

    
    '''
    avg_seq = "output/test/seq2seq/best_seq2seq_avg_pos_test.ig"
    zero_seq = "output/test/seq2seq/best_seq2seq_zero_pos_test.ig"
    avg_EDC = "output/test/ED_classify/best_ED_classify_avg_pos_test.ig"
    zero_EDC = "output/test/ED_classify/best_ED_classify_zero_pos_test.ig"
    
    with open(zero_seq) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["ID"]
            attr_storage[id] = fields["summed_attr"]

    kendall_correlation(attr_storage,zero_EDC,'summed_attr')
    ''' 
    
    # collate IG consensus
    bases = ['avg','zero']
    #bases = ['A','C','G','T']
    
    ED_file_list = ['output/test/ED_classify/best_ED_classify_'+b+'_pos.ig' for b in bases] 
    seq_file_list = ['output/test/seq2seq/best_seq2seq_'+b+'_pos.ig' for b in bases]
    
    attn_files = ["output/test/seq2seq/best_seq2seq_test_layer"+str(l)+".enc_dec_attns" for l in range(4)]
    avg_seq = "output/test/seq2seq/best_seq2seq_avg_pos_test.ig"
    zero_seq = "output/test/seq2seq/best_seq2seq_zero_pos_test.ig"
    
    attr_storage = {}
    with open(avg_seq) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["ID"]
            src = fields['src']
            l = strip_padding(src)
            attr_storage[id] = fields["summed_attr"][:l]
    multiple_correlation(attr_storage,attn_files,"seq2seq_summed_avg")
    
    attr_storage = {}
    with open(zero_seq) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["ID"]
            src = fields['src']
            l = strip_padding(src)
            attr_storage[id] = fields["summed_attr"][:l]
    multiple_correlation(attr_storage,attn_files,"seq2seq_summed_zero")

    attn_files = ["output/test/ED_classify/best_ED_classify_layer"+str(l)+".enc_dec_attns" for l in range(4)]
    avg_EDC = "output/test/ED_classify/best_ED_classify_avg_pos_test.ig"
    zero_EDC = "output/test/ED_classify/best_ED_classify_zero_pos_test.ig" 

    attr_storage = {}
    with open(avg_EDC) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["ID"]
            src = fields['src']
            l = strip_padding(src)
            attr_storage[id] = fields["summed_attr"][:l]
    multiple_correlation(attr_storage,attn_files,"ED_classify_avg_summed")

    attr_storage = {}
    with open(zero_EDC) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["ID"]
            src = fields['src']
            l = strip_padding(src)
            attr_storage[id] = fields["summed_attr"][:l]
    multiple_correlation(attr_storage,attn_files,"ED_classify_zero_summed")
