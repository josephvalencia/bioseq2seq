import json
import sys
import gzip
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import re

def load_enc_dec_attn(cds_storage,attn_file,align_on="start",mode="attn",plt_std_error=False):

    samples = []

    domain = list(range(-634,999))

    before_lengths = []
    after_lengths = []

    with open(attn_file) as inFile:
        for l in inFile:
            fields = json.loads(l)

            if mode == "attn":
                tgt = "layer_0_pos_0"
                id_field = "TSCRIPT_ID"
            else:
                tgt = "attr"
                id_field = "ID" 

            id = fields[id_field]

            if id in cds_storage:
                
                cds = cds_storage[id]
                if cds != "-1" :
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    splits = [clean(x) for x in splits]
                    start,end = tuple([int(x) for x in splits])
                
                    if mode == "attn":
                        attn =  keep_nonzero(fields[tgt])
                    else:
                        attn = [float(x) for x in fields[tgt]]

                    if align_on == "start":
                        before_lengths.append(start)
                        after_lengths.append(len(attn) - start)
                    elif align_on == "end":
                        before_lengths.append(end)
                        after_lengths.append(len(attn) - end)
                    else:
                        raise ValueError("align_on must be 'start' or 'end'")

                    samples.append(attn)
 
    if align_on == "start":
        max_before = max(before_lengths)
        domain = list(range(-max_before,999))
        samples = [align_on_start(attn,start,max_before) for attn,start in zip(samples,before_lengths)]
    else:
        max_after = max(after_lengths)
        domain = list(range(-999,max_after))
        samples = [align_on_end(attn,end,max_after) for attn,end in zip(samples,before_lengths)]

    samples = np.asarray(samples)

    # mean and standard error over samples
    consensus = np.nanmean(samples,axis=0)
   #mean_by_mod(consensus[max_before:max_before+400])

    error = np.nanstd(samples,axis=0) / np.sqrt(samples.shape[0])

    #plt.stem(domain, consensus, 'k', color='#CC4F1B')
    plt.stem(domain,consensus,use_line_collection=True)

    if plt_std_error:
        plt.fill_between(domain, consensus-2*error, consensus+2*error,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    
    #plt.xlim(50,100)
    plt.xlim(-10,13)
    plt.xticks(np.arange(-10,14))
    plt.title("Attention Layer 3")
    plt.xlabel("Relative Postition (CDS)")
    plt.ylabel("Attention Score")
    plt.tight_layout()
    plt.savefig("attention_mean3_zoomed.pdf")
    plt.close()

def getLongestORF(mRNA):
    ORF_start = -1
    ORF_end = -1
    longestORF = 0
    for startMatch in re.finditer('ATG',mRNA):
        remaining = mRNA[startMatch.start():]
        if re.search('TGA|TAG|TAA',remaining):
            for stopMatch in re.finditer('TGA|TAG|TAA',remaining):
                ORF = remaining[0:stopMatch.end()]
                if len(ORF) % 3 == 0:
                    if len(ORF) > longestORF:
                        ORF_start = startMatch.start()
                        ORF_end = stopMatch.end()
                        longestORF = len(ORF)
                    break

    return ORF_start,ORF_end

def mean_by_mod(attn):

    idx = np.arange(attn.shape[0])
    zero = idx % 3 == 0
    one = idx % 3 == 1
    two = idx % 3 == 2

    means = [np.nanmean(attn[mask]) for mask in [zero,one,two]]

    sns.barplot(x=[0,1,2],y=means)
    plt.xlabel("Pos. rel. to start mod 3")
    plt.ylabel("Mean Attention")
    plt.title("Attention by Frame")
    plt.savefig("attention_frames_mean1.pdf")
    plt.close()

def load_CDS(combined_file,include_lnc=False):

    print("parsing",combined_file)

    df = pd.read_csv(combined_file,sep="\t")
    df['RNA_LEN'] = [len(x) for x in df['RNA'].values.tolist()]
    df = df[df['RNA_LEN'] < 1000]
    
    ids_list = df['ID'].values.tolist()
    cds_list = df['CDS'].values.tolist()
    rna_list = df['RNA'].values.tolist()

    if include_lnc:
        temp = []
        # name largest ORF as CDS for lncRNA
        for i in range(len(cds_list)):
            curr = cds_list[i]
            if curr != "-1":
                temp.append(curr)
            else:
                start,end = getLongestORF(rna_list[i])
                fake_cds = "*{}:{}".format(start,end)
                temp.append(fake_cds)
        cds_list = temp

    return dict((x,y) for x,y in zip(ids_list,cds_list))

def keep_nonzero(attn):

    nonzero = []

    for a in attn:
        f = float(a)
        if f == 0.0:
            break
        else:
            nonzero.append(f)

    return nonzero

def align_on_start(attn,cds_start,max_start):

    max_len = 999
    
    indices = list(range(len(attn)))
    indices = [x-cds_start for x in indices]

    left_remainder = max_start - cds_start
    prefix = [np.nan for x in range(left_remainder)]
    right_remainder = max_len - indices[-1] -1
    suffix = [np.nan for x in range(right_remainder)]

    total = prefix +attn+ suffix
    return total

def align_on_end(attn,cds_end,max_end):

    max_len = 999

    indices = list(range(len(attn)))
    indices = [x-cds_end for x in indices]

    left_remainder = max_len - cds_end
    prefix = [np.nan for x in range(left_remainder)]
    right_remainder = max_end - indices[-1] -1
    suffix = [np.nan for x in range(right_remainder)]

    total = prefix +attn + suffix
    return total

if __name__ == "__main__":
    
    attn_file = sys.argv[1]
    combined_file = sys.argv[2]
    mode = sys.argv[3]

    cds_storage = load_CDS(combined_file)
    load_enc_dec_attn(cds_storage,attn_file,align_on ="start",mode=mode)
    
