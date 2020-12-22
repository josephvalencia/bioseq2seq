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
            for h in range(8):
                tgt_head = "layer{}head{}".format(l,h)
                attn = np.asarray(fields[tgt_head])
                attn_storage[tscript_id].append(attn)

    types = []
    max_attns = defaultdict(list)

    # match attention with attribution and process
    for tscript_id ,features in attn_storage.items():

        attr = attr_storage[tscript_id]
        attr = np.abs(np.asarray(attr))
        features = np.vstack(features).T

        maximums = features.max(axis=0).tolist()
        for h,m in enumerate(maximums):
            max_attns[h].append(m)

        # perform multiple linear regression
        N,D = features.shape
        ones = np.ones(N).reshape(-1,1)
        features = np.concatenate([features,ones],axis=1)
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
    print(mu,std,median)
   

    max_means = []
    for h,m in sorted(max_attns.items(),key = lambda x : x[0]):
        print(h,sum(m)/len(m))
        max_means.append(sum(m)/len(m))

    max_means = np.asarray(max_means)
    max_means = max_means.reshape(4,8)
    
    ax = sns.heatmap(max_means,cmap="Blues",annot=True)
    ax.set(xlabel="Head",ylabel="Layer")
    plt.title("Maximum Attention Weight (mean)")
    heat_plot = prefix+"_heatmap.pdf"
    plt.savefig(heat_plot)
    plt.close()

    textstr = '\n'.join((
    r'$\mu=%.3f$' % (mu, ),
    r'$\mathrm{median}=%.3f$' % (median, ),
    r'$\sigma=%.3f$' % (std, )))

    ax = sns.histplot(scores,binrange=(0,1),stat="probability",legend=False,alpha=0.5)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.075, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    ax.set(xlabel="Multiple Correlation (R)", ylabel='Density')
    plt.title(" Multiple Correlation Density"+"({})".format(prefix))
    corr_plot = prefix +"_kendall_corrs.pdf"
    plt.savefig(corr_plot)
    plt.close()

def kendall_correlation(attr_storage,attn_file,tgt_head):

    scores = []
    num_sig = 0

    with open(attn_file) as inFile:

        for line in inFile:
            fields = json.loads(line)
            tscript_id = fields["TSCRIPT_ID"]
            
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

    attr_storage = {}

    '''
    #attr_file = "results/best_ED_classify/best_ED_classify.ig"
    attr_file = "results/best_seq2seq/best_seq2seq.ig"
    
    with open(attr_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["ID"]
            attr_storage[id] = fields["normed_attr"]
    
    for l in range(4):
        #attn_file = "results/best_ED_classify/best_ED_classify_layer"+str(l)+".enc_dec_attns"
        attn_file = "results/best_seq2seq/best_seq2seq_layer"+str(l)+".enc_dec_attns"
        for h in range(8):
            tgt_head = "layer{}head{}".format(l,h)
            print("tgt_head: ",tgt_head)
            kendall_correlation(attr_storage,attn_file,tgt_head)
    '''
    
    attr_file = "results/best_seq2seq/best_seq2seq.ig"

    with open(attr_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["ID"]
            src = fields['src']
            l = strip_padding(src)
            attr_storage[id] = fields["summed_attr"][:l]
    
    attn_files = ["results/best_seq2seq/best_seq2seq_layer"+str(l)+".enc_dec_attns" for l in range(4)]
    multiple_correlation(attr_storage,attn_files,"seq2seq_summed")

    attr_file = "results/best_ED_classify/best_ED_classify.ig"

    with open(attr_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["ID"]
            src = fields['src']
            l = strip_padding(src)
            attr_storage[id] = fields["summed_attr"][:l]

    attn_files = ["results/best_ED_classify/best_ED_classify_layer"+str(l)+".enc_dec_attns" for l in range(4)]
    multiple_correlation(attr_storage,attn_files ,"ED_classify_summed")
