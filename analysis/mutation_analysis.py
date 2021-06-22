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

def mutation_analysis(saved_file,df,tgt_field,baseline,subset):

    storage = []
    count = 0 
    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields['ID']
            array = fields[tgt_field]
            seq = df.loc[id,'RNA']
            tscript_type = df.loc[id,'Type']
            
            if id in subset:
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
                            #delta = -codon_scores[j] / (cds_end-cds_start) 
                            delta = -codon_scores[j]
                            if delta > 0:
                                positive_component+=delta
                            elif delta < 0:
                                negative_component+=delta
                            
                            location = (i+j) - cds_start
                            entry = {'original' : codon , 'mutated' : new_codon , 'substitution' : substitution, 'delta' : delta, 'loc' : location, 'frame' : j+1}
                            storage.append(entry)
                count+=1
    print(count)
    return storage

def codon_multigraph_weighted_degree(aa,codons_df,partition):

    G = nx.MultiDiGraph(directed=True)
    
    codons_df = codons_df.reset_index()
    original = codons_df['original'].tolist()
    mutated = codons_df['mutated'].tolist()
    delta = codons_df['delta'].tolist()

    for v1,v2,wt in zip(original,mutated,delta):
        # disallow 'inconsistent' edges where improvement positive in both directions
        '''
        if G.has_edge(v2,v1) and G[v2][v1][0]['weight'] > 0 and wt > 0:
            G.remove_edge(v2,v1)
        else:
            G.add_edge(v1,v2,weight=wt)
        '''
        G.add_edge(v1,v2,weight=wt)

    
    storage = []
    for node in G:
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        wt_in_degree =  0 if in_degree == 0 else G.in_degree(node,weight="weight") /  in_degree
        wt_out_degree = 0 if out_degree == 0 else G.out_degree(node,weight="weight") / out_degree
        score = wt_in_degree - wt_out_degree
        entry = {'amino acid' : aa,'codon' : str(node),  'score' : score, 'partition' : partition}
        storage.append(entry)

        out_edges = G.edges(node,data=True)
        transitions = [(v,data['weight']) for u,v,data in out_edges]
        transitions = sorted(transitions, key = lambda x : x[1],reverse=True)
        print(node,transitions)    

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
    
    subset = []
    sub_file = "output/test/redundancy/test_reduced_80_ids.txt" 
    with open(sub_file) as inFile:
        for l in inFile:
            subset.append(l.rstrip())

    for b in ['A','G','C','T']:
        mdig_file = "output/test/seq2seq/best_seq2seq_{}_pos_test.ig".format(b)
        storage += mutation_analysis(mdig_file,df,"summed_attr",b,subset)
    
    summary = pd.DataFrame(storage) 
    summary['abs_delta'] = [np.abs(x) for x in summary['delta'].tolist()]
    summary['aa_original'] = [codonMap[c] for c in summary['original'].tolist()] 
    summary['aa_mutated'] = [codonMap[c] for c in summary['mutated'].tolist()] 
    summary.to_csv('MDIG_scores_by_codon.csv',sep="\t")

def synonymous_graph_centralities(synonymous):
    
    entries = []
    by_aa = synonymous.groupby(['aa_original','partition'])
    
    for (aa,partition), df_aa in by_aa:
        by_mutations_mean = df_aa.groupby(['original','mutated']).mean()
        entries.extend(codon_multigraph_weighted_degree(aa,by_mutations_mean,partition))
    
    trials = pd.DataFrame(entries)
    trials.to_csv('weighted_degree.csv',sep="\t")

def mdig_pipeline():

    summary = pd.read_csv('MDIG_scores_by_codon.csv',sep="\t") 
    summary = summary.sort_values(by=['aa_original','original','substitution'],ascending=[True,False,True]) 
    summary['partition']  = [x // 150 for x in summary['loc'].tolist()] 
    
    is_synonymous = (summary['aa_original'] == summary['aa_mutated']) 
    summary['class'] = np.where(is_synonymous,'Synonymous','Nonsynonymous')
    synonymous = summary[(summary['class'] == 'Synonymous') & (summary['delta'] != 0.0)]
    non_synonymous = summary[(summary['class'] == 'Nonsynonymous') & (summary['delta'] != 0.0)]
   
    # build codon graphs and calculate centralities
    synonymous_graph_centralities(synonymous)

    '''
    sns.histplot(data=summary_nonzero,x='delta',hue='class')
    plt.savefig('substitution_changes.svg')
    plt.close()
    by_class_a = summary_nonzero.groupby('class')[['class','delta']].mean()
    print("by_class_a",by_class_a)
    by_frame_synonymous = synonymous.groupby('substitution')[['substitution','delta']].mean()
    by_frame_nonsynonymous = non_synonymous.groupby('substitution')[['substitution','delta']].mean()
    both = by_frame_nonsynonymous.merge(by_frame_synonymous,left_index=True,right_index=True,suffixes=('_nonsynonymous','_synonymous'))
    both['synonymous_diff'] = both['delta_synonymous'] - both['delta_nonsynonymous']
    print("difference",both)
    '''
    
    # mask zeros with NaNs as they should not count towards median
    zero = summary.replace(0,np.NaN)
    overall = zero.groupby(['aa_original','original','substitution'])[['delta','abs_delta']].median()
    
    # build norm
    vmin = overall['delta'].min()
    vmax = overall['delta'].max()
    norm = MidpointNormalize(vmin=vmin,vmax=vmax,midpoint=0.0)
    cmap = sns.diverging_palette(220, 10, s=80, l=50, as_cmap=True)

    # plot colorbar
    colorbar(cmap,norm)

    # overall plot
    overall_filename = 'MDIG_changes_heatmap.svg'
    bubble_plot(overall,norm,cmap,overall_filename)

    # by partition
    max_p = summary['partition'].max()
    for p in range(max_p+1):
        group = summary[summary['partition'] == p]
        group = group.replace(0,np.NaN)
        group = group.groupby(['aa_original','original','substitution'])[['delta','abs_delta']].median()
        group = group.sort_values(by=['aa_original','original','substitution'],ascending=[True,False,True]) 
        group_filename = 'MDIG_changes_heatmap_partition_{}.svg'.format(p)
        bubble_plot(group,norm,cmap,group_filename)

def colorbar(cmap,norm):

    figure, axes = plt.subplots(figsize =(6, 1))
    figure.subplots_adjust(bottom = 0.5)
    figure.colorbar(matplotlib.cm.ScalarMappable(norm = norm,
        cmap = cmap),
        cax = axes, orientation ='horizontal',
        label ='median MDIG score')
    plt.savefig('MDIG_changes_colorbar.svg')
    plt.close()

def bubble_plot(df,norm,cmap,filename):

    # Draw each cell as a scatter point with varying size and color
    g = sns.relplot(data=df,
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
    
    # tweak styling
    g.set(xlabel="codon", ylabel="mutation (position-base)", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(x=0.025,y=0.05)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    
    # save
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def mutation_resistant_locations(saved_file,df,tgt_field,subset):

    loc_storage = [] 
    pct_storage = []

    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields['ID']
            array = fields[tgt_field]
            seq = df.loc[id,'RNA']
            top_bases = fields['top_bases']
            tscript_type = df.loc[id,'Type']
            
            if id in subset:
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
               
                L = len(array)
                array = np.asarray(array)
                zero_max = array == 0.0
                zeros = np.where(zero_max)[0].tolist()
                for z in zeros:
                    loc = z - cds_start   
                    max_base = top_bases[z] 
                    entry = {"transcript" : id , "type" : tscript_type , "location" : loc, 'max_base' : max_base} 
                    loc_storage.append(entry)
                stats = {"transcript" : id ,"type" : tscript_type , "n_zeros" : len(zeros) , "len" : L , "CDS_len" : cds_end-cds_start}
                pct_storage.append(stats)
        
    pct_df = pd.DataFrame(pct_storage)
    pct_df['pct_total'] =  pct_df['n_zeros'] / pct_df['len']
    sns.histplot(data=pct_df,x="pct_total",hue="type",common_bins=True,stat="density",element='step')
    plt.savefig("zero_percentage.svg")
    plt.close()
    
    pos_df = pd.DataFrame(loc_storage)
    loc_by_type = pos_df.groupby(['type','max_base'])['location'].mean()
    len_by_type = pct_df.groupby(['type'])['CDS_len'].describe()
    print(loc_by_type)
    print(len_by_type)
    sns.displot(data=pos_df,col='max_base',hue="type",x="location",common_bins=True,stat="count",element='step')
    plt.savefig("zero_positions.svg")
    plt.close()

if __name__ == "__main__":

    #df = pd.read_csv("../Fa/test.csv",sep="\t")
    #df = df.set_index('ID')
    #build_score_change_file(df)
    #sns.set_theme(style="whitegrid")
    #mutation_resistant_locations('max_MDIG.ig',df,'summed_attr')
    mdig_pipeline()
