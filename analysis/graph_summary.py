import os,sys
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import json
import pprint
import logomaker
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
    sns.distplot(distance,bins=2*max_attns.shape[0],kde=False)
    plt.xlabel("Distance to Max")
    plt.ylabel("Count")
    plt.title(head_name)

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

def pipeline(saved_attn):

    with open(saved_attn) as inFile:
        for line in inFile:

            decoded = json.loads(line)
            tscript_id = decoded['TSCRIPT_ID']
            seq = decoded['seq']
            cds_start = decoded['CDS_START']
            cds_end = decoded['CDS_END']
            layers = decoded['layers']
            print(tscript_id)

            if not os.path.isdir("self_output/"+tscript_id+"/"):
                os.mkdir("self_output/"+tscript_id+"/")
            
            for n,layer in enumerate(layers):
                max_PDF(layer,cds_start,cds_end,tscript_id)
                maxdist_PDF(layer,tscript_id)
                maxdist_txt(layer,tscript_id,seq)
                entropy_PDF(layer,tscript_id)


def layer_entropy_heatmap(saved_attn):

    activations = defaultdict(lambda: defaultdict(float))
    total_nucs = 0

    with open(saved_attn) as inFile:
        for line in inFile:

            decoded = json.loads(line)
            tscript_id = decoded['TSCRIPT_ID']
            seq = decoded['seq']
            cds_start = decoded['CDS_START']
            cds_end = decoded['CDS_END']
            layers = decoded['layers']

            if not os.path.isdir("self_output/"+tscript_id+"/"):
                os.mkdir("self_output/"+tscript_id+"/")
            
            for n,layer in enumerate(layers):
                
                sequence_logo(layer,tscript_id)
                max_PDF(layer,cds_start,cds_end,tscript_id)
                maxdist_PDF(layer,tscript_id)
                maxdist_txt(layer,tscript_id,seq)
                entropy_PDF(layer,tscript_id)

                layer_num = layer['layer']
                heads = layer['heads']
                h_len = 0

                for h in range(8):
                    entropy = heads[h]["h_x"]
                    curr_entropy = np.asarray([float(x) for x in entropy])

                    if np.any(np.isnan(curr_entropy)):
                        print(tscript_id)

                    h_len = len(curr_entropy)
                    activations[n][h] += np.sum(curr_entropy)

                total_nucs += h_len

    for n,inside in activations.items():
        for h,val in inside.items():
            inside[h] = val / total_nucs
    
    df = pd.DataFrame.from_dict(activations)
    sns.heatmap(df.transpose(),cmap="Blues")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Mean Entropy")
    plt.savefig("attn_head_entropy_heatmap.pdf")
    plt.close()

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

    pipeline(sys.argv[1])
