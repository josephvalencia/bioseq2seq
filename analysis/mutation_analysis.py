import sys,random
import json
import os,re,time
from tqdm import tqdm
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter

from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
import seaborn as sns

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

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
                        ORF_end = startMatch.start()+stopMatch.end()
                        longestORF = len(ORF)
                    break
    return ORF_start,ORF_end

def class_separation(saved_file):

    storage = []
    
    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "ID"
            tscript_id = fields[id_field]
            summed = np.asarray(fields['summed_attr']) / 1000
            status = "<PC>" if tscript_id.startswith("NM_") or tscript_id.startswith("XM_") else "<NC>"
            entry = {'transcript' : tscript_id , 'summed' : np.sum(summed), 'status' : status}
            print(entry)
            storage.append(entry)

    df = pd.DataFrame(storage)
    nc = df[df['status'] == '<NC>']
    pc = df[df['status'] == '<PC>']
    
    min_pc = np.round(pc['summed'].min(),decimals=3)
    max_pc = np.round(pc['summed'].max(),decimals=3)
    min_nc = np.round(nc['summed'].min(),decimals=3)
    max_nc = np.round(nc['summed'].max(),decimals=3)
    minimum = min(min_pc,min_nc)
    maximum = max(max_pc,max_nc)

    bins = np.arange(minimum,maximum,0.0005)
    pc_density,pc_bins = np.histogram(pc['summed'].values, bins=bins, density=True)
    pc_density = pc_density / pc_density.sum()
    nc_density,nc_bins = np.histogram(nc['summed'].values, bins=bins, density=True)
    nc_density = nc_density / nc_density.sum()
    jsd = jensenshannon(nc_density,pc_density)
    print('JS Divergence:',jsd)

    sns.histplot(df,x="summed",kde=False,hue="status",stat="density")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig(saved_file+'_class_hist.svg')
    plt.close()

def mutation_analysis(saved_file,df,tgt_field,baseline):

    storage = []
    
    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields['ID']
            array = fields[tgt_field]
            seq = df.loc[id,'RNA']
            tscript_type = df.loc[id,'Type']
            if tscript_type == "<PC>":               
                # use provided CDS
                cds = df.loc[id,'CDS']
                if cds != "-1":
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    cds_start,cds_end = tuple([int(clean(x)) for x in splits])
                else:
                    cds_start,cds_end = getLongestORF(seq)
            else:
                # use start and end of longest ORF
                cds_start,cds_end = getLongestORF(seq)
            
            array = [float(x) / 1000 for x in array]
            legal_chars = {'A','C','G','T'}
            allowed = lambda codon : all([x in legal_chars for x in codon])
            
            positive_component = 0
            negative_component = 0
            # traverse CDS
            print('Transcript {} , length {}'.format(id,cds_end-cds_start))
            for i in range(cds_start,cds_end-3,3):
                codon = seq[i:i+3]
                if allowed(codon):
                    codon_scores = array[i:i+3]
                    for j in range(len(codon_scores)):
                        new_codon = list(codon)
                        new_codon[j] = baseline
                        new_codon = ''.join(new_codon)
                        substitution = '{}-{}'.format(j+1,baseline)
                        
                        # MDIG is negative of IG
                        delta = -codon_scores[j] / (cds_end-cds_start) 
                        if delta > 0:
                            positive_component+=delta
                        elif delta < 0:
                            negative_component+=delta
                        
                        location = i - cds_start
                        entry = {'original' : codon , 'mutated' : new_codon , 'substitution' : substitution, 'delta' : delta, 'loc' : location}
                        storage.append(entry)

    return storage

def codon_multigraph_weighted_degree(aa,codons_df,i):

    G = nx.MultiDiGraph(directed=True)
    
    codons_df = codons_df.reset_index()
    original = codons_df['original'].tolist()
    mutated = codons_df['mutated'].tolist()
    delta = codons_df['delta'].tolist()

    for v1,v2,wt in zip(original,mutated,delta):
        G.add_edge(v1,v2, weight=wt)
    
    storage = []
    for node in G:
        wt_in_degree = G.in_degree(node,weight="weight") /  G.in_degree(node)
        wt_out_degree = G.out_degree(node,weight="weight") / G.out_degree(node)
        score = wt_in_degree - wt_out_degree
        #print('{}\t{}\t{}\t{}\t{}'.format(aa,node,wt_in_degree,wt_out_degree,score))
        entry = {'amino acid' : aa,'codon' : str(node),  'score' : score}
        storage.append(entry)

    '''
    ranking = sorted(storage,reverse=True)
    for i,(val,node) in enumerate(ranking):
        print('{} , {} , {}'.format(i,node,val))
    
    #a = nx.nx_agraph.to_agraph(G)
    e_pos = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.0]
    e_neg = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.0]
    layout = nx.spring_layout(G)  # positions for all nodes

    plt.subplot(7,3,i+1)
    plt.title(aa)
    nx.draw_networkx_nodes(G, layout, node_size=500)
    nx.draw_networkx_edges(G, layout, edgelist=e_pos, arrowstyle="->")
    #nx.draw_networkx_edges(G, layout, edgelist=e_neg, arrowstyle="->",edge_color="r")
    nx.draw_networkx_labels(G, layout, font_size=18, font_family="sans-serif")
    
    #a.draw('{}_codons.svg'.format(aa),prog='dot')
    #plt.savefig('{}_codons.svg'.format(aa))
    #plt.close()
    '''
    return storage

def codon_graph_katz(aa,codons_df,i):

    G = nx.DiGraph(directed=True)
    
    codons_df = codons_df.reset_index()
    original = codons_df['original'].tolist()
    mutated = codons_df['mutated'].tolist()
    delta = codons_df['delta'].tolist()

    for v1,v2,wt in zip(original,mutated,delta):
        if wt > 0 :
            # remove 'inconsistent' edges where improvement positive in both directions
            if G.has_edge(v2,v1):
                G.remove_edge(v2,v1)
            else:
                G.add_edge(v1,v2)
   
    storage = []
    
    if not nx.classes.function.is_empty(G):
        strong_connected = nx.algorithms.components.is_strongly_connected(G)
        centrality = nx.katz_centrality(G)
        for node in G:
            entry = {'amino acid' : aa,'codon' : str(node),  'score' : centrality[node]}
            storage.append(entry)
    
    return storage

def build_score_change_file(df):

    codonMap = {'TTT':'F', 'TTC':'F', 'TTA':'L', 'TTG':'L', 'TCT':'S', 
                'TCC':'S', 'TCA':'S', 'TCG':'S', 'TAT':'Y', 'TAC':'Y', 
                'TAA':'*', 'TAG':'*', 'TGT':'C', 'TGC':'C', 'TGA':'*', 
                'TGG':'W', 'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L',
                'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P', 'CAT':'H', 
                'CAC':'H', 'CAA':'Q', 'CAG':'Q', 'CGT':'R', 'CGC':'R', 
                'CGA':'R', 'CGG':'R', 'ATT':'I', 'ATC':'I', 'ATA':'I', 
                'ATG':'M', 'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
                'AAT':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K', 'AGT':'S', 
                'AGC':'S', 'AGA':'R', 'AGG':'R', 'GTT':'V', 'GTC':'V', 
                'GTA':'V', 'GTG':'V', 'GCT':'A', 'GCC':'A', 'GCA':'A', 
                'GCG':'A', 'GAT':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E',
                'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G',}
    
    storage = []

    for b in ['A','G','C','T']:
        mdig_file = "output/test/seq2seq/best_seq2seq_{}_pos_test.ig".format(b)
        storage += mutation_analysis(mdig_file,df,"summed_attr",b)
    
    #score_change_matrix = pd.DataFrame(storage)
    summary = pd.DataFrame(storage) 
    summary['abs_delta'] = [np.abs(x) for x in summary['delta'].tolist()]
    summary['aa_original'] = [codonMap[c] for c in summary['original'].tolist()] 
    summary['aa_mutated'] = [codonMap[c] for c in summary['mutated'].tolist()] 
    summary.to_csv('MDIG_scores_by_codon.csv',sep="\t")

def mdig_heatmap():

    summary = pd.read_csv('MDIG_scores_by_codon.csv',sep="\t") 
    
    ''' 
    summary = summary.sort_values(by=['aa_original','original'],ascending=[True,False]) 
    is_synonymous = (summary['aa_original'] == summary['aa_mutated'])  & (summary['delta'] != 0.0)
    synonymous = summary[is_synonymous]
    by_aa = synonymous.groupby('aa_original')
    
    num_iter = 100
    entries = []
    
    for idx, (aa, df_aa) in enumerate(by_aa):
        by_mutations_mean = df_aa.groupby(['original','mutated']).mean()
        print(by_mutations_mean)
        entries.extend(codon_multigraph_weighted_degree(aa,by_mutations_mean,idx))
    
    trials = pd.DataFrame(entries)
    trials.to_csv('weighted_degree.csv',sep="\t")
    quit() 
    #mean_df = trials.groupby(['amino acid','codon']).mean()
    #mean_df.to_csv('sampled_katz.csv',sep="\t")
    '''
   
    # overall
    overall = summary.groupby(['aa_original','original','substitution']).median()
    overall = overall.sort_values(by=['aa_original','original'],ascending=[True,False]) 
    
    # colorbar
    cmap = sns.diverging_palette(220, 10, s=80, l=50, as_cmap=True)
    vmin = overall['delta'].min()
    vmax = overall['delta'].max()
    norm = MidpointNormalize(vmin=vmin,vmax=vmax,midpoint=0.0)
  
    figure, axes = plt.subplots(figsize =(6, 1))
    figure.subplots_adjust(bottom = 0.5)
    figure.colorbar(matplotlib.cm.ScalarMappable(norm = norm,
        cmap = cmap),
        cax = axes, orientation ='horizontal',
        label ='median MDIG score')
    plt.savefig('MDIG_changes_colorbar.svg')
    plt.close()
    #axes.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    
    # Draw each cell as a scatter point with varying size and color
    g = sns.relplot(data=overall,
            x="original",
           y="substitution",
           hue="delta",
           size="abs_delta",
           palette=cmap,
           edgecolor=".7",
           legend=False,
           height=12,
           sizes=(0,150),
           hue_norm = norm,)
   
    # annotate
    g.set(xlabel="codon", ylabel="mutation (position-base)", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(x=0.025,y=0.05)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    plt.tight_layout()
    plt.savefig('MDIG_changes_heatmap.svg')
    plt.close()

    summary['partition'] = [int(x)//150 for x in summary['loc'].values.tolist()]
    max_p = summary['partition'].max()
    
    # by partition
    for p in range(max_p):
        group = summary[summary['partition'] == p]
        group = group.groupby(['aa_original','original','substitution']).median()
        group = group.sort_values(by=['aa_original','original'],ascending=[True,False]) 
        g = sns.relplot(data=group,
                x="original",
               y="substitution",
               hue="delta",
               size="abs_delta",
               palette=cmap,
               edgecolor=".7",
               legend=False,
               height=12,
               sizes=(0,150),
               hue_norm = norm,)
        g.set(xlabel="codon", ylabel="mutation (position-base)", aspect="equal")
        g.despine(left=True, bottom=True)
        g.ax.margins(x=0.025,y=0.05)
        for label in g.ax.get_xticklabels():
            label.set_rotation(90)
        plt.tight_layout()
        plt.savefig('MDIG_changes_heatmap_partition_{}.svg'.format(p))
        plt.close()

if __name__ == "__main__":

    #df = pd.read_csv("../Fa/test.csv",sep="\t")
    #df = df.set_index('ID')
    #build_score_change_file(df)
    sns.set_theme(style="whitegrid")
    mdig_heatmap()
