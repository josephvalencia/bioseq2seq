import sys
import json
from scipy.stats import pearsonr,kendalltau
from scipy.special import softmax
from captum.attr import visualization as viz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter

def plot_heatmap(name,array):
    
    array = np.asarray(array)
    k_largest_inds = get_top_k(array)
    array = np.expand_dims(array,axis=0)
    #array = array / np.linalg.norm(array,ord=2)    
    #array = softmax(array,axis=0)
        
    ax = sns.heatmap(np.transpose(array),cmap="Reds")
    '''
        for idx in k_largest_inds:
        txt = "({},{})".format(idx,labels[idx])
        plt.annotate(txt,(idx,1))
    '''

    plt.savefig(name+"_heatmap.png")
    plt.close()

def plot_stem(name,array,labels):

    array = np.asarray(array)
    array = np.expand_dims(array,axis=1)
    array = array / np.linalg.norm(array,ord=2)

    ax = plt.stem(array,use_line_collection=True)
    plt.savefig(name+"_stemplot.png")
    plt.close()

def get_top_k(array,k=15):

    k = k if k < len(array) else len(array)
    array = np.asarray(array)

    k_largest_inds = np.argpartition(array,-k)[-k:]
    k_largest_inds = k_largest_inds.tolist()

    return k_largest_inds

def all_plots(mode = "attn"):
    
    saved_file = sys.argv[1]
    storage = {}
    txt_storage = []

    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"
            id = fields[id_field]
            
            #tgt_field = "layer_0_pos_0"
            tgt_field = "attr"

            array = fields[tgt_field]
            name = id + "_" + mode
            
            plot_heatmap(name,array)
            #plot_stem(name,array,labels)
            
            topk_data = (id,get_top_k(array))
            txt_storage.append(topk_data)

    topk_file = mode+"_topk_idx.txt"

    with open(topk_file,'w') as outFile:
        for tscript,indices in txt_storage:
            index_str = ",".join([str(x) for x in indices])
            outFile.write("{},{}\n".format(tscript,index_str))

def plot_attn_attr_corr():
    storage = {}
    corrs = []
    count = 0

    attn_file = sys.argv[1]
    attr_file = sys.argv[2]
    attn_field = "layer_0_pos_0"
    
    print("Loading all attentions")
    with open(attn_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["TSCRIPT_ID"]
            storage[id] = fields[attn_field]

    print("Loading all attributions")
    with open(attr_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["ID"]
            attr = fields["attr"]
            
            if id in storage: # and (id.startswith("XM_") or id.startswith("NM_")):
                count +=1
                attn = storage[id]
                print("len(attr) {} , len(attn) {}".format(len(attr),len(attn)))

                if len(attr) < len(attn):
                    attn = attn[:len(attn)]
                elif len(attn) < len(attr):
                    attr = attr[:len(attn)]

                correlation = kendalltau(attr,attn)
                print("count {}, ID {}, corr {}".format(count,id,correlation))
                corrs.append(correlation[0])

    mu = np.mean(corrs)
    median = np.median(corrs)
    sigma = np.std(corrs)

    textstr = '\n'.join((
    r'$\mu=%.3f$' % (mu, ),
    r'$\mathrm{median}=%.3f$' % (median, ),
    r'$\sigma=%.3f$' % (sigma, )))

    ax = sns.distplot(corrs)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax.set(xlabel="Kendall tau", ylabel='Density')

    plt.title("Enc-Dec vs normed")
    plt.savefig("layer3head7_normed_kendall_corrs.png")
    plt.close()

def top_k_to_substrings():
    top_k_csv = sys.argv[1]
    self_attn_file = sys.argv[2]

    all_seqs = {}
    storage = []

    print("Loading self attention")
    # ingest self-attention and extract RNA sequence
    with open(self_attn_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["TSCRIPT_ID"]
            seq = fields["seq"]
            all_seqs[id] = seq
    
    print("Loading top k indexes")
    # ingest top k indexes from attribution/attention
    with open(top_k_csv) as inFile:
        for l in inFile:
            fields = l.rstrip().split(",")
            id = fields[0]
            seq = all_seqs[id]
            
            substrings = []
            bound = 10
            
            # get window around indexes
            for i in fields[1:]:
                i = int(i)
                start = i-bound-1 if i-bound-1 > 0 else 0
                end = i+bound if i+bound < len(seq) -1 else len(seq) -1
                substr = seq[start:end]
                substrings.append(substr)
            
            entry = [id]+ substrings
            storage.append(entry)

    top_kmer_file = top_k_csv.split(".")[0] +".subseq.csv"

    print("Writing to "+top_kmer_file)
    # write output
    with open(top_kmer_file,'w') as outFile:
        for s in storage:
            outFile.write(",".join(s)+"\n")

def countkmers():

    substrings_file = sys.argv[1]
    counter = Counter()

    with open(substrings_file) as inFile:
        for l in inFile:
            fields = l.split(",")
            substrings = fields[1:]
            counter.update(substrings)

    counts = counter.most_common(200)

    with open("normed_attr_most_common.txt" ,'w') as outFile:
        for substr, count in counts:
            outFile.write("{},{}\n".format(substr,count))
            
if __name__ == "__main__":

    #top_k_to_substrings()
    #all_plots(mode="attr")
    plot_attn_attr_corr()
    #countkmers()