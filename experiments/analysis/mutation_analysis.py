import os,re
from tqdm import tqdm
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import parse_config, setup_fonts, build_output_dir
from Bio.SeqUtils import seq3

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import seaborn as sns

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

def mutation_analysis(saved_file,df,metric,exclude=False):

    storage = []
    saved = np.load(saved_file)
    onehot_file = None
    if metric == 'grad':
        onehot_file = np.load(saved_file.replace('grad','onehot')) 
    
    if exclude:
        homology = pd.read_csv("test_maximal_homology.csv")
        reduced = homology['score'] <=80
        homology = homology.loc[reduced]
        allowed = set()
        allowed.update(homology['ID'].tolist())
    
    for tscript,array in tqdm(saved.items()):
        if exclude and tscript not in allowed:
            continue
        seq = df.loc[tscript,'RNA']
        tscript_type = df.loc[tscript,'Type']
        if metric == 'grad':
            onehot = onehot_file[tscript]
            taylor = array - onehot*array
            array = taylor[:,2:6]
        if tscript_type == "<PC>":               
            # use provtscripted CDS
            cds = df.loc[tscript,'CDS']
            if cds != "-1":
                splits = cds.split(":")
                clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                cds_start,cds_end = tuple([int(clean(x)) for x in splits])
            else:
                cds_start,cds_end = getLongestORF(seq)
        else:
            # use start and end of longest ORF
            cds_start,cds_end = getLongestORF(seq)
        
        legal_chars = {'A','C','G','T'}
        allowed = lambda codon : all([x in legal_chars for x in codon])
      
        # traverse CDS
        if tscript_type == "<PC>": 
            for i in range(cds_start,cds_end-3,3):
                codon = seq[i:i+3]
                if allowed(codon):
                    codon_scores = array[i:i+3,:]
                    for frame in [0,1,2]:
                        for base_index,base in enumerate('ACGT'):
                            new_codon = [c if idx != frame else base for (idx,c) in enumerate(codon)] 
                            new_codon = ''.join(new_codon)
                            substitution = '{}-{}'.format(frame+1,base)
                            delta = codon_scores[frame,base_index] 
                            location = i+frame - cds_start
                            entry = {'tscript' : tscript, 'original' : codon , 'mutated' : new_codon , 'substitution' : substitution,\
                                    'delta' : delta, 'loc' : location, 'frame' : frame+1, 'cds_length' : cds_end - cds_start}
                            storage.append(entry)
    return storage

def build_or_load_score_change_file(df,ism_file,mut_file,metric='MDIG',exclude=False):

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
   
    if not os.path.exists(mut_file):
        print(f'building {mut_file}')
        storage = mutation_analysis(ism_file,df,metric,exclude=exclude)
        summary = pd.DataFrame(storage)
        summary.to_csv(mut_file)
        print(f'{mut_file} built')
    else:
        print(f'Loading {mut_file}')
        summary = pd.read_csv(mut_file)
        print(f'{mut_file} loaded')
     
    summary['aa_original'] = [codonMap[c] for c in summary['original'].tolist()] 
    summary['aa_mutated'] = [codonMap[c] for c in summary['mutated'].tolist()] 
    return summary


def plot_mutations_by_aa(aa_original,df,metric,mut_dir):

    print(f'Plotting {aa_original}')
    df = df[df['delta'] != 0.0]
    is_synonymous = (df['aa_original'] == df['aa_mutated']) 
    is_nonsense =  ~(is_synonymous) & (df['aa_mutated'] == '*')
    
    synonymous = df[is_synonymous]
    missense = df[~is_synonymous & ~is_nonsense]
    nonsense = df[~is_synonymous & is_nonsense]
   

    plt.figure(figsize=(3.5,2.25))
    sns.lineplot(data=synonymous,x='percentile',y='delta',hue='mutation',alpha=0.9)
    try: 
        n_codons = len(pd.unique(df['original']))
        if n_codons > 2:
            sns.lineplot(data=synonymous,x='percentile',y='delta',label='Synonymous',c='blue',linestyle='dashed')
    except:
        print('Uh oh')
    g = sns.lineplot(data=missense,x='percentile',y='delta',label='Missense',c='red',linestyle='dashed')
    long_name = seq3(aa_original)
    sns.move_legend(g,title=f'{long_name} point mutation',title_fontsize=8,fontsize=8,loc="upper left", bbox_to_anchor=(1, 1))
    plt.axhline(y=0,color='black', linestyle=':')    
    plt.ylabel(f'Mean $\Delta$S',fontsize=8)
    plt.xlabel('Fraction of CDS',fontsize=8) 
    sns.despine()
    filename =  f'{mut_dir}/{aa_original}_{metric}_mutation_progress.svg'
    plt.tight_layout()
    plt.savefig(filename)
    print(f'Saved {filename}')
    plt.close()
    
    # position-independent average of all mutations
    synonymous = synonymous[synonymous['loc'] < 150]
    mean_by_long_mut = synonymous.groupby('long_mutation')['delta'].mean().reset_index()
    count_by_long_mut = synonymous.groupby('long_mutation')['delta'].count().reset_index()
    long_means = mean_by_long_mut.merge(count_by_long_mut,on='long_mutation',suffixes=('_mean','_count'))
    long_means = long_means.rename(columns={'delta_mean' : 'Mean', 'delta_count' : 'Count'})
    return long_means

def calculate_fractions(location_list,cds_length_list,n_bins=25):
    
    bins = np.asarray([(1/n_bins)*x for x in range(n_bins+1)])
    fractions = [ x/y for x,y in zip(location_list,cds_length_list)] 
    inds = np.digitize(fractions,bins)
    left_bins = bins[inds-1]
    right_bins = bins[inds]
    return left_bins

def parse_entry(entry):

    if any([c == ' ' for c in entry]):
        return None
    else:
        return entry[0], float(entry[1])

def build_codon_table(filename):
   
    raw_frequencies = {}
    search_string = re.search('(.*)/(.*)_codon_table.txt',filename)
    if search_string is not None:
        species = search_string.group(2)
    else:
        species = filename

    with open(filename,'r') as inFile:
        fields = inFile.read().rstrip().replace('\n','\t').split('\t')
    
    for i in range(0,len(fields),3):
        entry = fields[i:i+3]
        result = parse_entry(entry)
        if result:
            raw_frequencies[result[0]] = result[1]
    return species,raw_frequencies

def build_means(df):

    by_base = df.groupby('substitution')['delta']
    means = by_base.mean().reset_index()
    counts = by_base.count().reset_index()
    print(counts) 
    fields = [x.split('-') for x in means['substitution'].tolist()]
    means['position'] = [int(x)-1 for x,y in fields]
    means['base'] = [y for x,y in fields]
    means = means.pivot(index='position',columns='base',values='delta') 
    return means

def codon_positions(synonymous,missense,mut_dir,metric):
    
    ''' plot by codon position and base '''

    plt.figure(figsize=(3.5,2.25))
    
    print('Synonymous counts')
    means_synon = build_means(synonymous)
    # add missing 2nd positions and re-sort 
    means_synon.loc[1] = [np.nan,np.nan,np.nan,np.nan]
    means_synon = means_synon.sort_index() 
    
    print('Non-synonymous counts')
    means_missense = build_means(missense)
   
    # synon
    vmax = max(np.nanmax(np.abs(means_synon.to_numpy())),np.nanmax(np.abs(means_missense.to_numpy()))) 
    g = sns.heatmap(data=means_synon,center=0,vmin=-vmax,vmax=vmax,square=True,cmap='RdBu_r')
    plt.savefig(f'{mut_dir}/mean_{metric}_synonymous_codon_positions.svg')
    print(f'saved {mut_dir}/mean_{metric}_synonymous_codon_positions.svg')
    plt.close()
    # non-synon 
    g = sns.heatmap(data=means_missense,center=0,vmin=-vmax,vmax=vmax,square=True,cmap='RdBu_r')
    plt.savefig(f'{mut_dir}/mean_{metric}_missense_codon_positions.svg')
    print(f'saved {mut_dir}/mean_{metric}_missense_codon_positions.svg')
    plt.close()

def mutation_pipeline(summary,mut_dir,metric):

    sns.set_style(style="white",rc={'font.family' : ['Helvetica']})
    
    summary['long_mutation'] = [f'{x}>{y}' for x,y in zip(summary['original'],summary['mutated'])]
    summary['mutation'] = ['>'+y.replace('T','U') for y in summary['mutated']]
    summary['baseline'] = [x.split('-')[1] for x in summary['substitution']]
    summary['percentile'] = calculate_fractions(summary['loc'].tolist(),summary['cds_length'].tolist()) 

    # plot mean snonynmous by base and position
    is_synonymous = (summary['aa_original'] == summary['aa_mutated']) & (summary['aa_original'] != '*') 
    synonymous = summary[is_synonymous] 
    synonymous = synonymous[synonymous['delta'] != 0.0]
    
    is_missense = (summary['aa_original'] != summary['aa_mutated']) & (summary['aa_original'] != '*') & (summary['aa_mutated'] != '*') 
    missense = summary[is_missense] 
    missense = missense[missense['delta'] != 0.0]
    codon_positions(synonymous,missense,mut_dir,metric) 
    
    storage = []
    for aa_original,group in summary.groupby('aa_original'):
        if aa_original != '*': 
            long_results = plot_mutations_by_aa(aa_original,group,metric,mut_dir)
            storage.append(long_results)

    long_differences = pd.concat(storage)
    long_differences.to_csv(f'{mut_dir}{metric}_codon_deltas.csv')
    
    # plot by percentile
    plt.figure(figsize=(2,2.5)) 
    by_frame = summary.groupby(['frame','baseline']).mean().reset_index()
    sns.barplot(data=by_frame,x='baseline',y='delta',hue='frame')
    plt.ylabel(f'Mean {metric}')
    plt.tight_layout() 
    rel_filename = f'{mut_dir}{metric}_by_frame_baseline.svg'
    plt.savefig(rel_filename)
    plt.close()
    
def mutation_pipeline_from_config():
     
    args,unknown_args = parse_config()
    setup_fonts() 
   
    # setup raw attribution data from test set
    prefix = args.test_prefix.replace('test','test_RNA')
    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    df_test = pd.read_csv(test_file,sep='\t').set_index('ID')
    best_BIO_ISM_verified = os.path.join(args.best_BIO_DIR,f'verified_test_RNA.{args.reference_class}.{args.position}.ISM.npz')
    best_EDC_ISM_verified = os.path.join(args.best_EDC_DIR,f'verified_test_RNA.{args.reference_class}.{args.position}.ISM.npz')
    best_BIO_ISM = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.ISM.npz')
    best_BIO_MDIG = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.MDIG.max_0.50.npz')
   
    output_dir = build_output_dir(args)
    mutation_dir  =  f'{output_dir}/mut/'
    if not os.path.isdir(mutation_dir):
        os.mkdir(mutation_dir)

    # ISM verified on both tasks, needed for sanity checks
    mut_file = best_BIO_ISM_verified.replace('.npz','_mutation_scores.csv') 
    mutation_df = build_or_load_score_change_file(df_test,best_BIO_ISM_verified,mut_file,metric='ISM',exclude=True)
    mut_file = best_EDC_ISM_verified.replace('.npz','_mutation_scores.csv') 
    mutation_df = build_or_load_score_change_file(df_test,best_EDC_ISM_verified,mut_file,metric='ISM',exclude=True)

    # BIO run for both ISM
    mut_file = best_BIO_ISM.replace('.npz','_mutation_scores.csv')
    mutation_df = build_or_load_score_change_file(df_test,best_BIO_ISM,mut_file,metric='ISM',exclude=True)
    mutation_pipeline(mutation_df,mutation_dir,'ISM')
    # and MDIG 
    mut_file = best_BIO_MDIG.replace('.npz','_mutation_scores.csv')
    mutation_df = build_or_load_score_change_file(df_test,best_BIO_MDIG,mut_file,metric='MDIG',exclude=True)
    mutation_pipeline(mutation_df,mutation_dir,'MDIG')

if __name__ == "__main__":

    mutation_pipeline_from_config()
