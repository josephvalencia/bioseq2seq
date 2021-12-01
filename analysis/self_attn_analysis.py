import os,sys
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import json
import pprint
import logomaker
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import Counter, defaultdict
from matplotlib.backends.backend_pdf import PdfPages

def plot_entropy(entropy,tscript_name,head_name):
    """ Plot nucleotide position vs Shannon entropy of attention.
        Args:
    """
    entropy = np.asarray(entropy)
    x = np.arange(entropy.shape[0])

    plt.plot(x,entropy)
    plt.ylabel("Entropy (bits)")
    plt.xlabel("Nucleotide")
    plt.title("Attention Entropy")

def plot(values,cds_start,cds_end,xlabel,ylabel,title,mask_diagonal = False):

    if mask_diagonal:
        x,values = mask_diagonal(values)
    else:
        x = np.arange(values.shape[0])

    if cds_start != -1 and cds_end != -1:
        colors = get_colors(values,cds_start,cds_end)
        plt.scatter(x,values,s=1,c=colors)
        annotate_cds(cds_start,cds_end)
    else:
        plt.scatter(x,values,s=1)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

def get_colors(values,cds_start,cds_end):

    # colorbrewer categorical color-blind safe green, orange, purple
    color_map = {0:"#66c2a5",1:"#fc8d62",2:"#8da0cb"}

    cds_len = cds_end - cds_start

    five_prime_colors = ["gray"] * cds_start
    cds_colors = [color_map[i%3] for i in range(cds_len)]
    three_prime_colors = ["gray"] * (values.shape[0] - cds_end)

    return five_prime_colors+cds_colors+three_prime_colors

def annotate_cds(start,end):

    width = 0.75
    color = "dimgray"

    plt.axhline(start,linestyle = "-.",linewidth = width,c=color)
    plt.axhline(end,linestyle = "-.",linewidth = width,c=color)
    plt.axvline(start,linestyle = "-.",linewidth = width,c=color)
    plt.axvline(end,linestyle = "-.",linewidth = width,c=color)

def mask_diagonal(values,threshold=1):

    x = np.arange(values.shape[0])
    non_diagonal = np.abs(values - x) > threshold
    non_diagonal_mask = np.nonzero(non_diagonal)

    values = values[non_diagonal_mask]
    x = x[non_diagonal_mask]
    return x,values

def plot_max(max_attns,cds_start,cds_end,tscript_name,head_name,line = False,no_diagonal = False):
    """ Plot nucleotide position vs max index of attention.
        Args:
    """
    xlabel = "Nuc Index"
    ylabel = "Max Index"
    title = head_name
    plot(max_attns,cds_start,cds_end,xlabel,ylabel,title)

def plot_maxdist(max_attns,tscript_name,head_name):

    distance = max_attns - np.arange(max_attns.shape[0])
    sns.histplot(distance,bins=2*max_attns.shape[0],stat='probability',kde=False)
    plt.xlabel("Distance to Max")
    plt.ylabel("Count")
    plt.title(head_name)

def summarize_maxes(layer,tscript_id,storage):

    layer_num = layer['layer']
    heads = layer['heads']

    for h in heads:
        max_attns = np.asarray(h['max'])
        absolute = tabulate(max_attns)
        layerhead = "layer{}head{}".format(layer_num,h['head'])
        abs_val, abs_weight = absolute[0]
        distance = max_attns - np.arange(max_attns.shape[0])
        relative = tabulate(distance)
        rel_val, rel_weight = relative[0]
        entry = {'tscript_id': tscript_id, 'head' : layerhead , 'abs_weight' : abs_weight, 'abs_val' : abs_val , 'rel_weight' : rel_weight , 'rel_val' : rel_val}
        storage.append(entry)

def tabulate(vals):

    unique, counts = np.unique(vals,return_counts=True)
    total = vals.shape[0]
    counts = [(i,c/total) for i,c in zip(unique.tolist(),counts.tolist())]
    counts = sorted(counts,key= lambda x : x[1],reverse=True)
    return counts

def plot_center(centers,cds_start,cds_end,tscript_name,head_name,line = False,no_diagonal = False):
    """ Plot nucleotide position vs center index of attention.
        Args: """
    
    centers = np.asarray(centers)
    x = np.arange(centers.shape[0])

    xlabel = "Nuc Index"
    ylabel = "Center Index"
    title = "Center of Attention" +tscript_name+"."+head_name
    filename = "self_output/"+tscript_name+"/"+head_name+"_center.pdf"

    plot(centers,cds_start,cds_end,xlabel,ylabel,title,filename)

def plot_heatmap(tscript_name,head_name,attns):
    """ Plot attention weights (L,L)
    """
    df = pd.DataFrame.from_records(attns.cpu().numpy())

    sns.heatmap(df,cmap="Blues")

    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.title("Attention Heatmap"+tscript_name+"."+head_name)
    plt.savefig("self_output/"+tscript_name+"/"+head_name+"_heatmap.pdf")
    plt.close()

def summarize_positional_heads(saved_attn):

    storage = []
    
    subset = set()
    sub_file = "output/test/redundancy/test_reduced_80_ids.txt" 
    
    with open(sub_file) as inFile:
        for l in inFile:
            subset.add(l.rstrip())
    
    with open(saved_attn) as inFile:
        for line in inFile:
            decoded = json.loads(line)
            tscript_id = decoded['TSCRIPT_ID']
            
            if tscript_id in subset:
                seq = decoded['seq']
                cds_start = decoded['CDS_START']
                cds_end = decoded['CDS_END']
                layers = decoded['layers']
                for n,layer in enumerate(layers):
                    summarize_maxes(layer,tscript_id,storage)

    pos_df = pd.DataFrame(storage)
    print(pos_df)
    mean = pos_df.groupby('head').mean()
    var = pos_df.groupby('head').var()
    mode = pos_df.groupby('head').agg(lambda x: pd.Series.mode(x).values[0])

    a = mean.merge(var,on='head',suffixes = ('_mean','_var'))
    a = a.merge(mode,on='head')
    b = ['abs_weight' ,'abs_val', 'rel_weight' ,'rel_val']
    a.columns = a.columns.map(lambda x : x+'_mode' if x in b  else x)
    a = a.drop(columns=a.columns.difference(['abs_weight_mean','rel_weight_mean','rel_val_var','rel_val_mode']))
    return a

def pipeline(seq_attn_file,ED_attn_file):

    seq_df = summarize_positional_heads(seq_attn_file).reset_index()
    seq_df['model'] = ['bioseq2seq' for _ in range(len(seq_df))]
    seq_df.to_csv('bioseq2seq_self_attn_maxes.csv',sep='\t')
    
    ED_df = summarize_positional_heads(ED_attn_file).reset_index()
    ED_df['model'] = ['EDC' for _ in range(len(ED_df))]
    ED_df.to_csv('EDC_self_attn_maxes.csv',sep='\t')
    
    seq_df = pd.read_csv('bioseq2seq_self_attn_maxes.csv',sep='\t')
    ED_df = pd.read_csv('EDC_self_attn_maxes.csv',sep='\t')
    
    # relative position
    seq_relative = [x for x in seq_df['rel_weight_mean'].values.tolist()] 
    ED_relative = [x for x in ED_df['rel_weight_mean'].values.tolist()] 
    min_val = min(min(seq_relative),min(ED_relative)) 
    max_val = max(max(seq_relative),max(ED_relative)) 
    self_attn_heatmap_relative(seq_df,min_val,max_val,'bioseq2seq')
    self_attn_heatmap_relative(ED_df,min_val,max_val,'EDC')

    # absolute position
    seq_abs = [x for x in seq_df['abs_weight_mean'].values.tolist()] 
    ED_abs = [x for x in ED_df['abs_weight_mean'].values.tolist()] 
    min_val = min(min(seq_abs),min(ED_abs)) 
    max_val = max(max(seq_abs),max(ED_abs)) 
    self_attn_heatmap_absolute(seq_df,min_val,max_val,'bioseq2seq')
    self_attn_heatmap_absolute(ED_df,min_val,max_val,'EDC')

def self_attn_heatmap_relative(a,vmin,vmax,prefix):

    rel_weights = a['rel_weight_mean'].values.reshape(4,8)
    rel_val_mode = a['rel_val_mode'].values.reshape(4,8)
    rel_val_var = a['rel_val_var'].values.reshape(4,8)

    inconsistent = rel_val_var > 0.1
    
    annotations = rel_val_mode.astype(str)
    annotations = annotations.ravel().tolist()
    annotations = [x if x.startswith('-') else '+'+x for x in annotations]
    annotations = np.asarray(annotations).reshape(4,8)
    annotations[inconsistent] = ""
   
    heatmap(rel_weights,"Greens",vmin,vmax,prefix,annotations=annotations)
    plt.tight_layout()
    plt.savefig(prefix+'_self_attn_maxes_rel_pos.svg')
    plt.close()

def self_attn_heatmap_absolute(a,vmin,vmax,prefix):

    abs_weights = a['abs_weight_mean'].values.reshape(4,8)
    
    heatmap(abs_weights,"Oranges",vmin,vmax,prefix)
    plt.tight_layout()
    plt.savefig(prefix+'_self_attn_maxes_abs_pos.svg')
    plt.close()

def heatmap(weights,colors,vmin,vmax,prefix,annotations=None):

    if not annotations is None:
        annotations = annotations.T

    ax = sns.heatmap(data=weights.T,annot=annotations,fmt='s',square=True,cmap=colors,vmin=vmin,vmax=vmax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    lw = 1.5
    ax.axvline(x=0, color='k',linewidth=lw)
    ax.axvline(x=weights.shape[0], color='k',linewidth=lw)
    ax.axhline(y=0, color='k',linewidth=lw)
    ax.axhline(y=weights.shape[1], color='k',linewidth=lw)
    ax.set_xlabel(' Layer ({})'.format(prefix))
    ax.set_ylabel(' Head')
    ax.set_title('')

def max_PDF(layer,cds_start,cds_end,tscript_id):

    layer_num = layer['layer']
    heads = layer['heads']
    filename = "self_output/{}/layer{}_max.pdf".format(tscript_id,layer_num)

    with PdfPages(filename) as pdf:
        for shard in [heads[:4],heads[4:]]:
            plot_layer_max(layer_num,shard,cds_start,cds_end,tscript_id)
            pdf.savefig()
            plt.close()

def maxdist_PDF(layer,tscript_id):

    layer_num = layer['layer']
    heads = layer['heads']
    filename = "self_output/{}/layer{}_maxdist.pdf".format(tscript_id,layer_num)

    with PdfPages(filename) as pdf:
        for shard in [heads[:4],heads[4:]]:
            plot_layer_maxdist(layer_num,shard,tscript_id)
            pdf.savefig()
            plt.close()

def entropy_PDF(layer,tscript_id):

    layer_num = layer['layer']
    heads = layer['heads']

    filename = "self_output/{}/layer{}_entropy.pdf".format(tscript_id,layer_num)

    with PdfPages(filename) as pdf:
        for shard in [heads[:4],heads[4:]]:
            plot_layer_entropy(layer_num,shard,tscript_id)
            pdf.savefig()
            plt.close()

def plot_layer_max(layer_num,heads,cds_start,cds_end,tscript_id):

    top = 10

    for i,head in enumerate(heads):
        head_name = "Head {}".format(head['head'])
        max_attns = head['max']
        plt.subplot(2,2,i+1)
        plot_max(np.asarray(max_attns),cds_start,cds_end,tscript_id,head_name)

    suptitle = "Max {} Layer {}".format(tscript_id,layer_num)
    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_layer_maxdist(layer_num,heads,tscript_id):

    top = 10

    for i,head in enumerate(heads):
        head_name = "Head {}".format(head['head'])
        max_attns = head['max']
        plt.subplot(2,2,i+1)
        plot_maxdist(np.asarray(max_attns),tscript_id,head_name)

    suptitle = "Max Distance Distribution {} Layer {}".format(tscript_id,layer_num)
    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_layer_entropy(layer_num,heads,tscript_id):

    for i,head in enumerate(heads):

        head_name = "Head {}".format(head['head'])
        entropy = head['h_x']
        plt.subplot(2,2,i+1)
        plot_entropy(np.asarray(entropy),tscript_id,head_name)

    suptitle = "Entropy {} Layer {}".format(tscript_id,layer_num)
    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def maxdist_txt(layer,tscript_id,seq):

    layer_num = layer['layer']
    heads = layer['heads']
    filename = "self_output/{}/layer{}_maxpos.txt".format(tscript_id,layer_num)

    top = 10

    with open(filename,'w') as outFile:

        for head in heads:

            head_name = "Head {}\n".format(head['head'])
            outFile.write(head_name)
            max_attns = head['max']

            counts = Counter(max_attns).most_common(top)

            triplets = [seq[i-1:i+2] for i,_ in counts]
            triplet_counts = [x for _,x in Counter(triplets).most_common(top)]
            total = sum(triplet_counts)
            triplet_probs = [float(x)/total for x in triplet_counts]
            triplet_entropy = -sum([np.log2(x)*x for x in triplet_probs])

            for idx,count in counts:
                triplet = seq[idx-1:idx+2]
                outFile.write(" idx : {}, count : {} , triplet : {} \n".format(idx,count,triplet))

            outFile.write(" H(codon) : {}\n".format(triplet_entropy))
            outFile.write("\n")

if __name__ == "__main__":

    pipeline(sys.argv[1],sys.argv[2])
