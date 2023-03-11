import sys,random
import os,re,time
from tqdm import tqdm
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import parse_config, load_CDS, setup_fonts
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import seq3

from scipy.stats import pearsonr,spearmanr
import numpy as np
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
# FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated
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

def mutation_analysis(saved_file,df,metric):

    storage = []
    saved = np.load(saved_file)
    onehot_file = None
    
    if metric == 'grad':
        onehot_file = np.load(saved_file.replace('grad','onehot')) 
    for tscript,array in tqdm(saved.items()):
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

def build_or_load_score_change_file(df,ism_file,mut_file,metric='MDIG'):

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
        storage = mutation_analysis(ism_file,df,metric)
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


def plot_mutations_by_aa(aa_original,df):

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
    filename =  f'{aa_original}_mutation_progress.svg'
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

def mutation_pipeline(summary,mut_dir):

    sns.set_style(style="white",rc={'font.family' : ['Helvetica']})
   
    tai_df = pd.read_csv('human_TAI_Tuller.csv')
    tai_dict = {x : y for x,y in zip(tai_df['Codon'],tai_df['Human'])} 
    
    summary['long_mutation'] = [f'{x}>{y}' for x,y in zip(summary['original'],summary['mutated'])]
    summary['mutation'] = ['>'+y.replace('T','U') for y in summary['mutated']]
    summary['baseline'] = [x.split('-')[1] for x in summary['substitution']]
    summary['percentile'] = calculate_fractions(summary['loc'].tolist(),summary['cds_length'].tolist()) 
    
    storage = []
    for aa_original,group in summary.groupby('aa_original'):
        if aa_original != '*': 
            long_results = plot_mutations_by_aa(aa_original,group)
            storage.append(long_results)

    _,human_usage = build_codon_table('data/codon_usage_tables/homo_sapiens_codon_table.txt')
    long_differences = pd.concat(storage)
    tai_deltas = []
    usage_deltas = []
    for long_mutation,mean in zip(long_differences['long_mutation'],long_differences['Mean']):
        src,dest = tuple(long_mutation.split('>'))
        delta_tai = tai_dict[dest] - tai_dict[src]
        delta_usage = human_usage[dest] - human_usage[src]
        usage_deltas.append(delta_usage) 
        tai_deltas.append(delta_tai)
    long_differences['delta_TAI'] = tai_deltas 
    long_differences['delta_usage'] = usage_deltas 
    print('DIFF')
    print(long_differences)
    
    long_differences.to_csv('codon_deltas.csv')
    s_deltas = long_differences['Mean']
    pearson = pearsonr(s_deltas,tai_deltas)[0]
    spearman = spearmanr(s_deltas,tai_deltas)[0]
    g = sns.jointplot(data=long_differences,x='Mean',y='delta_TAI',kind='reg')
    plt.xlabel(r'$\Delta$S synonymous mutation') 
    plt.ylabel(r'$\Delta$tAI synonymous mutation') 
    full = f'Pearson={pearson:.3f}\nSpearman={spearman:.3f}'
    plt.text(0.05, 0.9,full,
            transform=g.ax_joint.transAxes)
    plt.savefig('delta_usage_regression.svg')
    plt.close()

    # plot by percentile
    plt.figure(figsize=(2,2.5)) 
    by_frame = summary.groupby(['frame','baseline']).mean().reset_index()
    sns.barplot(data=by_frame,x='baseline',y='delta',hue='frame')
    plt.ylabel('Mean MDIG')
    plt.tight_layout() 
    rel_filename = f'{mut_dir}_by_frame_baseline.svg'
    plt.savefig(rel_filename)
    plt.close()
    
def mutation_pipeline_from_config():
     
    args,unknown_args = parse_config()
    setup_fonts() 
    
    prefix = args.test_prefix.replace('test','test_RNA')
    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    df_test = pd.read_csv(test_file,sep='\t').set_index('ID')
    #best_BIO_ISM = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.ISM.npz')
    best_BIO_ISM = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.MDIG.max_0.50.npz')
    print(best_BIO_ISM)
    
    ''' 
    prefix = args.train_prefix.replace('train','train_RNA')
    train_file = os.path.join(args.data_dir,args.train_prefix+'.csv')
    df_train = pd.read_csv(train_file,sep='\t').set_index('ID')
    best_BIO_MDIG = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.MDIG.npz')
    ''' 
    
    #best_BIO_MDIG = os.path.join(args.best_EDC_DIR,f'verified_test_RNA.{args.reference_class}.{args.position}.ISM.npz')
    #best_BIO_GRAD = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.grad.npz')
    #best_BIO_ISM = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.ISM.npz')
    
    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}/'
    mutation_dir  =  f'{output_dir}mut/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(mutation_dir):
        os.mkdir(mutation_dir)
   
    mut_file = best_BIO_ISM.replace('.npz','_mutation_scores.csv')
    mutation_df = build_or_load_score_change_file(df_test,best_BIO_ISM,mut_file,metric='ISM')
    #mut_file = best_BIO_MDIG.replace('.npz','_mutation_scores.csv')
    #mutation_df = build_or_load_score_change_file(df_train,best_BIO_MDIG,mut_file,metric='MDIG')
    mutation_pipeline(mutation_df,mutation_dir)

if __name__ == "__main__":

    mutation_pipeline_from_config()
