import sys,random
import json
import os
from scipy.stats import pearsonr,kendalltau
from scipy.special import softmax
from captum.attr import visualization as viz
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logomaker
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages

def plot_heatmap(name,array):
    
    array = np.asarray(array)
    k_largest_inds = get_top_k(array)
    
    array = np.expand_dims(array,axis=0)
    # array = array / np.linalg.norm(array,ord=2)    
    # array = softmax(array,axis=0)

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

def plot_line(name,array):

    array = np.asarray(array)
    plt.plot(array)
    plt.savefig(name+"_lineplot.pdf")
    plt.close()

def get_top_k(array,k=15):

    k = k if k < len(array) else len(array)
    array = np.asarray(array)

    k_largest_inds = np.argpartition(array,-k)[-k:]
    k_largest_inds = k_largest_inds.tolist()

    return k_largest_inds

def get_min_k(array,k=15):

    k = k if k < len(array) else len(array)
    array = np.asarray(array)

    k_smallest_inds = np.argpartition(array,k)[:k]
    k_smallest_inds = k_smallest_inds.tolist()

    return k_smallest_inds

def top_indices(saved_file,mode= "attn"):
    
    storage = {}
    txt_storage = []

    out_name = saved_file.split(".")[0]

    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"
            tgt_field = "layer_0_pos_0" if mode == "attn" else "attr"

            id = fields[id_field]
            array = fields[tgt_field]

            name = id + "_" + mode
            topk_data = (id,get_top_k(array))
            #topk_data = (id,get_min_k(array))
            #plot_line(id,array)
            txt_storage.append(topk_data)

    topk_file = out_name+"_"+mode+"_topk_idx.txt"

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
    attn_prefix = attn_file.split(".")[0]
    
    print("Loading all attentions")
    with open(attn_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["TSCRIPT_ID"]
            storage[id] = fields[attn_field]
    
    example_indexes = random.sample(range(len(storage)),10)
    examples = []

    print("Loading all attributions")
    with open(attr_file) as inFile:
        idx = 0
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
                
                if idx in example_indexes:
                    examples.append((id,attr,attn))

                print("count {}, ID {}, corr {}".format(count,id,correlation))
                corrs.append(correlation[0])
                idx+=1
        
        filename = attn_prefix +"_corr_examples.pdf"

        with PdfPages(filename) as pdf:
            for id,attr,attn in examples:
             
                plt.plot(attr,'k',color='#CC4F1B')
                plt.plot(attn,'k',linestyle='dashed')
                plt.ylim(-0.1,0.1)
                ax = plt.gca()

                ax.set(xlabel="Pos. Relative to CDS", ylabel="Attention Score")
                ax.set_title(id)

                plt.tight_layout(rect=[0,0.03,1,0.95])
                pdf.savefig()
                plt.close()

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
    corr_plot = attn_prefix +"_normed_kendall_corrs.pdf"
    plt.savefig(corr_plot)
    plt.close()

def top_k_to_substrings(top_k_csv,self_attn_file):
    
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
            seq = all_seqs[id].rstrip()
            
            substrings = []
            left_bound = 2
            right_bound = 2
            
            # get window around indexes
            for i in fields[1:]:
                i = int(i)
                start = i-left_bound if i-left_bound > 0 else 0
                end = i+right_bound+1 if i+right_bound+1 < len(seq) -1 else len(seq) -1
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

def countkmers(substrings_file,top_file,logo_file):

    counter = Counter()

    with open(substrings_file) as inFile:
        for l in inFile:
            fields = l.rstrip().split(",")
            substrings = fields[1:]
            counter.update(substrings)

    counts = counter.most_common(20)

    consensus_list = []

    with open(top_file,'w') as outFile:
        for substr, count in counts:
            outFile.write("{},{}\n".format(substr,count))

            if len(substr) == 5:
                consensus_list.extend([substr]*count)

    counts_mat = logomaker.alignment_to_matrix(consensus_list)
    info_mat = logomaker.transform_matrix(counts_mat, from_type='counts', to_type='information')
    logo = logomaker.Logo(info_mat,color_scheme='classic')
    logo.ax.set_xlabel('Position',fontsize=14)
    logo.ax.set_ylabel('Information (bits)',fontsize=14)
    plt.tight_layout()

    plt.savefig(logo_file)

def codon_scores(saved_file,self_attn_file,mode="attn"):

    all_seqs = {}

    print("Loading self attention")
    # ingest self-attention and extract RNA sequence
    with open(self_attn_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields["TSCRIPT_ID"]
            seq = fields["seq"]
            all_seqs[id] = seq

    tri_counts = defaultdict(int)
    tri_scores = defaultdict(float)

    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"
            tgt_field = "layer_0_pos_0" if mode == "attn" else "attr"

            id = fields[id_field]
            array = fields[tgt_field]
            seq = all_seqs[id]

            if len(seq) != len(array):
                array = [float(x) for x in array[:len(seq)]]
            #assert len(seq) == len(array) ,"Size mismatch between sequence and scores"

            tri_nucleotide_scores = [sum(array[i:i+3]) for i in range(len(array)-3)]
            tri_nucleotides = [seq[i:i+3] for i in range(len(seq)-3)]

            for codon,score in zip(tri_nucleotides,tri_nucleotide_scores):
                tri_counts[codon]+=1
                tri_scores[codon]+=score

        tri_df = pd.DataFrame()

    avg = 0
    total = 0
    # normalize and print
    for codon in tri_counts.keys():
        avg += tri_scores[codon]
        total +=tri_counts[codon] 
        tri_scores[codon] /= tri_counts[codon]
        if 'N' not in codon and 'Y' not in codon and 'R' not in codon:
            print("{} : score {} , count : {}".format(codon,tri_scores[codon],tri_counts[codon]))        
    
    print("all codon average : score {} , count : {}".format(avg/total,total))        
    
def run_attributions(saved_file,mode="attn"):

    prefix = saved_file.split(".")[0]
    
    self_attn_file = "large/dev.self_attns"
    #self_attn_file = "covid_1k.self_attns"
    kmer_indices_file = prefix+"_"+mode+"_topk_idx.txt"
    substrings_file = prefix+"_"+mode+"_topk_idx.subseq.csv"
    top_file = prefix+"_"+mode+"_most_common.txt"
    logo_file = prefix+"_"+mode+"_logo.pdf"

    #top_indices(saved_file,mode=mode)
    #top_k_to_substrings(kmer_indices_file,self_attn_file)
    #countkmers(substrings_file,top_file,logo_file)
    codon_scores(saved_file,self_attn_file,mode)

if __name__ == "__main__":

    '''
    for h in range(8):
        if h == 6:
            continue
        head = "large/layer3head"+str(h)+".enc_dec_attns"
        run_attributions(head)
    '''
    #run_attributions("covid_1k.enc_dec_attns")
    run_attributions("large/layer3head3.enc_dec_attns","attn")
    #run_attributions("large/layersummedheadzero.attr","attr")
    #plot_attn_attr_corr()