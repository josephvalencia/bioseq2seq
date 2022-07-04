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

from utils import parse_config, add_file_list, load_CDS

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

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
            summed = [float(x) for x in fields['summed_attr']]
            summed = np.asarray(summed)
            status = "<PC>" if tscript_id.startswith("NM_") or tscript_id.startswith("XM_") else "<NC>"
            entry = {'transcript' : tscript_id , 'summed' : np.sum(summed), 'status' : status}
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
    count = 0 
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
            
            array = [float(x) for x in array]
            legal_chars = {'A','C','G','T'}
            allowed = lambda codon : all([x in legal_chars for x in codon])
            
            positive_component = 0
            negative_component = 0
            # traverse CDS
            if tscript_type == "<PC>": 
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
                            #delta = codon_scores[j] / (cds_end-cds_start) 
                            delta = codon_scores[j]
                            
                            if delta > 0:
                                positive_component+=delta
                            elif delta < 0:
                                negative_component+=delta
                            
                            location = (i+j) - cds_start
                            entry = {'original' : codon , 'mutated' : new_codon , 'substitution' : substitution,\
                                    'delta' : delta, 'loc' : location, 'frame' : j+1, 'cds_length' : cds_end - cds_start}
                            storage.append(entry)
            count+=1
    return storage

def codon_multigraph_weighted_degree(aa,codons_df,partition,cluster_maxes,transition_maxes):

    G = nx.MultiDiGraph(directed=True)
    
    codons_df = codons_df.reset_index()
    original = codons_df['original'].tolist()
    mutated = codons_df['mutated'].tolist()
    delta = codons_df['delta'].tolist()
    consistent = False

    for v1,v2,wt in zip(original,mutated,delta):
        # disallow 'inconsistent' edges where improvement positive in both directions
        if consistent: 
            if G.has_edge(v2,v1) and G[v2][v1][0]['weight'] > 0 and wt > 0:
                G.remove_edge(v2,v1)
            else:
                G.add_edge(v1,v2,weight=wt)
        else: 
            G.add_edge(v1,v2,weight=wt)

    storage = []
    for node in G:
        # identify top synonymous mutation for each codon 
        out_edges = G.edges(node,data=True)
        transitions = [(v,data['weight']) for u,v,data in out_edges]
        transitions = sorted(transitions, key = lambda x : x[1],reverse=True)
        if len(transitions) > 0: 
            top_codon,top_wt = transitions[0]
            top_codon = top_codon if top_wt > 0 else node 
            transition_maxes[node] = top_codon
        else:
            transition_maxes[node] = node

        # calculate weight centralities 
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        wt_in_degree =  0 if in_degree == 0 else G.in_degree(node,weight="weight") /  in_degree
        wt_out_degree = 0 if out_degree == 0 else G.out_degree(node,weight="weight") / out_degree
        score = wt_in_degree - wt_out_degree
        entry = {'amino acid' : aa,'codon' : node,  'score' : score, 'partition' : partition}
        storage.append(entry)

    # set maximum in graph as the optimal codon
    graph_order = sorted(storage, key=lambda x : x['score'], reverse=True)
    cluster_center = graph_order[0]['codon']
    for node in G:
        cluster_maxes[node] = cluster_center

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

def build_score_change_file(df,attr_info):

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
    path_list = attr_info['path_list']
    labels = attr_info['bases']
    for f,b in zip(path_list,labels):
        storage += mutation_analysis(f,df,"summed_attr",b)
    
    summary = pd.DataFrame(storage) 
    summary['aa_original'] = [codonMap[c] for c in summary['original'].tolist()] 
    summary['aa_mutated'] = [codonMap[c] for c in summary['mutated'].tolist()] 
    
    return summary

def synonymous_graph_centralities(synonymous,mut_file):
    
    entries = []
    centers = {}
    transitions = {}

    by_aa = synonymous.groupby(['aa_original','partition_rel'])
    #by_aa = synonymous.groupby('aa_original') 
    for (aa,partition), df_aa in by_aa:
    #for aa , df_aa in by_aa:
        by_mutations_mean = df_aa.groupby(['original','mutated']).mean()
        #entries.extend(codon_multigraph_weighted_degree(aa,by_mutations_mean,partition))
        entries.extend(codon_multigraph_weighted_degree(aa,by_mutations_mean,partition,centers,transitions))
   
    centers['ATG'] = 'ATG'
    transitions['ATG'] = 'ATG'

    center_codons_file = f'{mut_file}optimal_codons_by_center.json'
    with open(center_codons_file,'w') as inFile:
        json.dump(centers,inFile)
    
    edge_codons_file = f'{mut_file}optimal_codons_by_edge.json'
    with open(edge_codons_file,'w') as inFile:
        json.dump(transitions,inFile)
    
    trials = pd.DataFrame(entries)
    trials.to_csv('weighted_degree.csv',sep="\t")

def analyze_stops(missense,mut_dir):

    plt.figure(figsize=(10,6))
    by_partition = missense.groupby(['partition_rel'])
    #sns.violinplot(data=missense,x='partition',y='delta')
    g = sns.lineplot(data=missense,x='partition_rel',y='delta',ci=95,err_style='band')

    step = 150
    partitions = sorted(missense['partition_rel'].unique())
    bins = [f'[{x*step},{(x+1)*step})' for x in partitions]
    bins[-1] = '[900-1000)'
    print(partitions,bins)
    
    plt.xticks(partitions,bins)
    plt.ylabel('Mean MDIG of missense mutation')
    plt.xlabel('partition')
    plt.tight_layout()
    missense_file = f'{mut_dir}missense_scores.svg'
    plt.savefig(missense_file)
    plt.close()

    '''
    for partition, df_p in by_partition:
        by_mutations_mean = df_p['delta'].mean()
        avoidance_scores.append(by_mutations_mean)
    '''

def mdig_pipeline(summary,mut_dir):

    summary = summary.sort_values(by=['aa_original','original','substitution'],ascending=[True,False,True]) 
   
    # relative is a percentile location in the CDS, absolute is a fixed window size 
    summary['partition_rel'] = [round(x/y,1) for x,y in zip(summary['loc'].tolist(),summary['cds_length'].tolist())] 
    summary['partition_abs']  = [x // 150 for x in summary['loc'].tolist()] 

    # plot relative
    by_frame = summary.groupby(['frame','partition_rel']).mean().reset_index()
    sns.barplot(data=by_frame,x='partition_rel',y='delta',hue='frame') 
    rel_filename = f'{mut_dir}_by_frame_partition_rel.svg'
    plt.savefig(rel_filename)
    plt.close()

    # plot absolute
    by_frame = summary.groupby(['frame','partition_abs']).mean().reset_index()
    sns.barplot(data=by_frame,x='partition_abs',y='delta',hue='frame') 
    abs_filename = f'{mut_dir}_by_frame_partition_abs.svg'
    plt.savefig(abs_filename)
    plt.close()

    # analyze missense mutations
    is_missense = ((summary['aa_original'] != '*' ) & (summary['aa_mutated'] == '*'))  & (summary['delta'] != 0.0)
    missense = summary[is_missense]
    analyze_stops(missense,mut_dir)
    
    # build codon graphs and calculate centralities
    is_synonymous = (summary['aa_original'] == summary['aa_mutated']) 
    synonymous = summary[(is_synonymous) & (summary['delta'] != 0.0)]
    non_synonymous = summary[(~is_synonymous) & (summary['delta'] != 0.0)]
    synonymous_graph_centralities(synonymous,mut_dir)
  
    '''
    # synonymous difference
    by_frame_synonymous = synonymous.groupby('original')[['original','delta']].mean()
    by_frame_nonsynonymous = non_synonymous.groupby('original')[['original','delta']].mean()
    both = by_frame_nonsynonymous.merge(by_frame_synonymous,left_index=True,right_index=True,suffixes=('_nonsynonymous','_synonymous'))
    both['synonymous_diff'] = both['delta_synonymous'] - both['delta_nonsynonymous']
    both = both.sort_values(by='synonymous_diff',ascending=False).reset_index()
    original = both['original'].values.tolist()
    diff = both['synonymous_diff'].values.tolist()
    '''
    
    # mask zeros with NaNs as they should not count towards median
    nonzero = summary.replace(0,np.NaN)
    nonzero['status'] = np.where(is_synonymous,'Synonymous','Nonsynonymous')
    
    # build norm
    overall = nonzero.groupby(['aa_original','original','substitution'])[['delta']].mean()
    vmin = overall['delta'].min()
    vmax = overall['delta'].max()
    
    five = overall['delta'].quantile(0.05)
    ninety_five = overall['delta'].quantile(0.95)
    
    #norm = MidpointNormalize(vmin=vmin,vmax=vmax,midpoint=0.0)
    #norm = MidpointNormalize(vmin=five,vmax=ninety_five,midpoint=0.0) 
    norm = mcolors.SymLogNorm(linthresh=1e-7,vmin=vmin,vmax=vmax)
    
    # plot colorbar
    cmap = sns.diverging_palette(220, 10, s=80, l=50, as_cmap=True)
    colorbar(cmap,norm,mut_dir)

    # overall plot
    overall_filename = f'{mut_dir}MDIG_changes_heatmap.svg'
    bubble_plot(overall,norm,cmap,overall_filename)

    # by partition
    for m in ['partition_abs','partition_rel']:
        for p in summary[m].unique():
            group = summary[summary[m] == p]
            group = group.replace(0,np.NaN)
            group = group.groupby(['aa_original','original','substitution'])[['delta']].mean()
            group = group.sort_values(by=['aa_original','original','substitution'],ascending=[True,False,True]) 
            group_filename = f'{mut_dir}MDIG_changes_heatmap_{m}_{p}.svg'
            bubble_plot(group,norm,cmap,group_filename)

def colorbar(cmap,norm,mut_dir):

    figure, axes = plt.subplots(figsize =(6, 1))
    figure.subplots_adjust(bottom = 0.5)
    figure.colorbar(matplotlib.cm.ScalarMappable(norm = norm,
        cmap = cmap),
        cax = axes, orientation ='horizontal',
        label ='median MDIG score')
    plt.savefig(f'{mut_dir}MDIG_changes_colorbar.svg')
    plt.close()

def bubble_plot(df,norm,cmap,filename):

    # Draw each cell as a scatter point with varying size and color
    df['abs_delta'] = [np.abs(x) for x in df['delta'].tolist()]
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

def mutation_resistant_locations(saved_file,df,tgt_field):

    loc_storage = [] 
    pct_storage = []
    coding_seqs = []
    noncoding_seqs = []
    counter = Counter()
    c = 0

    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id = fields['ID']
            array = fields[tgt_field]
            seq = df.loc[id,'RNA']
            top_bases = fields['top_bases']
            tscript_type = df.loc[id,'Type']
            
            if tscript_type == "<PC>":               
                # use provided CDS
                cds = df.loc[id,'CDS']
                if cds != "-1":
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    cds_start,cds_end = tuple([int(clean(x)) for x in splits])
                    c+=1 
                    L = len(array)
                    array = np.asarray(array)
                    zero_max = array == 0.0
                    zeros = np.where(zero_max)[0].tolist()
                    frames = [(z-cds_start) % 3 for z in zeros]

                    contiguous = continuous_sections(zeros)
                    ''' 
                    for i,(s,e) in enumerate(contiguous):
                        if s > L-10:
                            substr = seq[-20:]
                        elif s < 10:
                            substr = seq[:20]
                        else:
                            substr = seq[s-10:s+11]
                        description = "loc[{}:{}]".format(s+1,e+1)
                        record = SeqRecord(Seq(substr),
                                                id=id+"_"+str(i),
                                                description=description)
                        coding = id.startswith('NM') or id.startswith('XM') 
                        if coding:
                            coding_seqs.append(record)
                        else:
                            noncoding_seqs.append(record)
                    
                    for z in zeros:
                        loc = z - cds_start   
                        max_base = top_bases[z] 
                        entry = {"transcript" : id , "type" : tscript_type , "location" : loc, 'max_base' : max_base} 
                        loc_storage.append(entry)
                    
                    stats = {"transcript" : id ,"type" : tscript_type , "n_zeros" : len(zeros) , "len" : L , "CDS_len" : cds_end-cds_start}
                    pct_storage.append(stats)
                    '''
                    counter.update(frames)
    '''
    with open('NC_motifs_MDIG.fa','w') as outFile:
        SeqIO.write(noncoding_seqs, outFile, "fasta")

    with open('PC_motifs_MDIG.fa','w') as outFile:
        SeqIO.write(coding_seqs, outFile, "fasta")

    pct_df = pd.DataFrame(pct_storage)
    pct_df['pct_total'] =  pct_df['n_zeros'] / pct_df['len']
    sns.histplot(data=pct_df,x="pct_total",hue="type",common_bins=True,stat="density",element='step')
    plt.savefig("zero_percentage.svg")
    plt.close()
    
    pos_df = pd.DataFrame(loc_storage)
    loc_by_type = pos_df.groupby(['type','max_base'])['location'].mean()
    len_by_type = pct_df.groupby(['type'])['CDS_len'].describe()
    sns.displot(data=pos_df,col='max_base',hue="type",x="location",common_bins=True,stat="count",element='step')
    plt.savefig("zero_positions.svg")
    plt.close()
    '''
    
    print('Constrained positions')
    for i in range(3):
        print(i,counter[i])
    print(c)

def continuous_sections(zeros):

    continuous = []
    temp = [None,None]

    for z in zeros:
        if temp[0] is None:
            temp[0] = z
        else:
            if temp[1] is None:
                if z == temp[0]+1:
                    temp[1] = z
                else:
                    temp[0] = None
            else:
                if z == temp[1]+1:
                    temp[1] = z
                else:
                    continuous.append(tuple(temp))
                    temp = [None,None]

    return continuous

def mdig_pipeline_from_config():

    args,unknown_args = parse_config()
    test_file = args.test_csv
    test_cds = load_CDS(test_file)
    df_test = pd.read_csv(test_file,sep='\t').set_index('ID')
    
    # load attribution files from config
    best_seq_MDIG = add_file_list(args.best_seq_MDIG,'bases')
    
    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}/'
    mutation_dir  =  f'{output_dir}mut/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(mutation_dir):
        os.mkdir(mutation_dir)
    
    mutation_df = build_score_change_file(df_test,best_seq_MDIG)
    mdig_pipeline(mutation_df,mutation_dir)

if __name__ == "__main__":

    mdig_pipeline_from_config()
