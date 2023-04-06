import sys,random
import os,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.signal import convolve
from collections import defaultdict

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

from utils import parse_config, build_EDA_file_list, getLongestORF, get_CDS_loc, build_output_dir

def select_index(array,mode='argmax',excluded=None,smoothed=False):
  
    slice_window = 10
    if smoothed:
        full_width = 2*slice_window + 1
        array = convolve(np.abs(array),np.ones(full_width),mode="same") / full_width
   
    region_len = array.shape[-1]
    if mode == 'argmax':
        center =  np.argmax(array).item()
        # if argmax close to region edge, adjust center inwards to prevent slicing outside region
        center = min(region_len-slice_window-1,center)
        center = max(slice_window,center)
        return center
    elif mode == 'random':
        # just avoid the edges of the functional region 
        if excluded is None:
            return random.randrange(slice_window,region_len-slice_window - 1)
        # try not to overlap the excluded indices. 
        else:
            # must avoid 2*slice_window to account for slice_window around both excluded and sampled 
            excluded_idx = [excluded+x for x in range(-2*slice_window,2*slice_window)]
            # also avoid the edges of the functional region 
            other_idx = [x for x in range(slice_window,region_len-slice_window) if x not in excluded_idx]
            if len(other_idx) > 0: 
                return random.sample(other_idx,1)[0]
            # in a few cases impossible to avoid excluded, so just duplicate location, reducing statistical power marginally
            else:
                return excluded
    else:
        raise ValueError("Not an index selection strategy")

def reduce_over_mutations(array,mode='PC',mask=None):

    if mask is not None:
        array = np.where(mask,0.0,array)

    # by default, mutation scores represent \Delta S = \Delta log(P(PC) / log(NC))
    if mode == 'PC':
        # an endogenous nucleotide is as PC as the most negative \Delta S from a point mutation 
        array = np.min(array,axis=1)
    else:
        # an endogenous nucleotide is as NC as the most positive \Delta S from a point mutation 
        array = np.max(array,axis=1)
    
    return np.abs(array)

def slice_functional_region(array,start,end,seq,region='full'):
 
    min_len = 50

    # slice 
    if region == '5-prime':
        subarray = array[:start,:]
        subseq = seq[:start]
        return (subarray,subseq) if len(subarray) > min_len else None 
    elif region == '3-prime':
        subarray = array[end:,:]
        subseq = seq[end:]
        return (subarray,subseq) if len(subarray) > min_len else None 
    elif region == 'CDS':
        subarray = array[start:end,:]
        subseq = seq[start:end]
        return (subarray,subseq) if len(subarray) > min_len else None 
    else:
        return array,seq

def adjust_indices_by_region(index,start,end,region):

    if region == '3-prime':
        return end + index 
    elif region == 'CDS':
        return start + index 
    # 5-prime and full both do not need adjustment 
    else:
        return index

def get_codon_map():
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
    return codonMap

def diff(codon1,codon2):
    
    locs = []
    for i,(c1,c2) in enumerate(zip(codon1,codon2)):
        if c1 != c2:
            locs.append(i)
    return locs

def mask_frame_independent(array,seq,mask_level=3):
    
    mask = np.zeros_like(array)
    codonMap = get_codon_map()

    # always mask stops
    nuc_to_j = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}
    stop_codons  =  ['TAG','TGA','TAA']
    banned_codons = stop_codons 
    close_to_stop = defaultdict(list) 

    # UGG observed to dominate like a stop
    if mask_level == 2: 
        codons.append('TGG')
    # mask anything a point mutation away from a stop 
    if mask_level == 3:
        for codon in codonMap.keys():
            for stop in stop_codons:
                diffs = diff(codon,stop)
                if len(diffs) == 1 and codon not in stop_codons:
                    loc = diffs[0]
                    close_to_stop[codon].append((stop,loc,nuc_to_j[stop[loc]]))
        banned_codons += close_to_stop.keys()
    
    # always mask starts
    banned_codons.append('ATG')
    
    for codon in banned_codons:    
        for match in re.finditer(codon,seq):
            if codon in close_to_stop:
                for stop,loc,j in close_to_stop[codon]: 
                    i = match.start() + loc
                    mask[i,j] = 1
            else: 
                mask[match.start():match.end(),:] = 1

    return mask 

def maskORF(array,seq):

    codon_map = get_codon_map()
    legal_chars = {'A','C','G','T'}
    allowed = lambda codon : all([x in legal_chars for x in codon])
   
    # initialize mask
    mask = [] 
    start_mask = np.ones(shape=(3,4))
    endpoint = len(seq) - 3 if len(seq) % 3 == 0 else 3 * (len(seq) // 3)
    remainder = len(seq) - endpoint
    stop_mask = np.ones(shape=(remainder,4))
    mask.append(start_mask) 
   
    # traverse ORF
    for i in range(3,endpoint,3):
        codon = seq[i:i+3]
        if allowed(codon):
            codon_storage = [] 
            for frame in [0,1,2]:
                innermost = [] 
                for base_index,base in enumerate('ACGT'):
                    new_codon = [c if idx != frame else base for (idx,c) in enumerate(codon)] 
                    new_codon = ''.join(new_codon)
                    if new_codon in ['TGA','TAG','TAA']:
                        innermost.append(1)
                    else:
                        innermost.append(0)
                inner_mask = np.asarray(innermost)
                codon_storage.append(inner_mask)
            codon_mask = np.stack(codon_storage,axis=0)
        else:
            codon_mask = np.ones(shape=(3,4))
        mask.append(codon_mask)
    
    mask.append(stop_mask)
    mask = np.concatenate(mask,axis=0)

    # illegalize all common codons regardless of frame
    untargeted_mask = mask_frame_independent(array,seq)
    mask = np.logical_or(mask,untargeted_mask).astype(int)
    return mask

def mask_locations(array,seq,region):

    mask = np.zeros_like(array)

    if region == '5-prime' or region == '3-prime':
        mask = mask_frame_independent(array,seq) 
    elif region == 'CDS':
        mask = mask_frame_independent(array,seq)
    else:
        pass
    
    nonzero = np.count_nonzero(mask)
    detailed = np.count_nonzero(mask,axis=1)
    #print(f'masked = {nonzero}/{mask.size} = {nonzero/mask.size:.3f}')
    return mask

def top_indices(saved_file,df,groups,metrics,mode="attn",head_idx=0,reduction_mode='PC',region="full",apply_mask=False):
    
    negative_storage = []
    positive_storage = []
    
    loaded = np.load(saved_file)
    onehot_file = None
    if mode == 'grad':
        onehot_file = np.load(saved_file.replace('grad','onehot')) 

    for tscript,array in tqdm(loaded.items()):
        
        if mode == 'attn':
            array = array[head_idx,:]
        elif mode == 'grad':
            # Taylor approx
            array = array - onehot_file[tscript] * array
            array = array[:,2:6]

        name = tscript + "_" + mode
        coding = tscript.startswith('XM') or tscript.startswith('NM') 
        start,end = get_CDS_loc(df.loc[tscript,'CDS'],df.loc[tscript,'RNA'])
        seq = df.loc[tscript,'RNA']
        
        functional_region = slice_functional_region(array,start,end,seq,region=region)
        if functional_region is None:
            continue
        
        subarray,subseq = functional_region
        unreduced = subarray
        
        mask = None
        if apply_mask:
            mask = mask_locations(subarray,subseq,region)
            unreduced = np.where(mask,0.0,unreduced)
        if mode in ['MDIG','ISM','grad']:
            subarray = reduce_over_mutations(subarray,mode=reduction_mode,mask=mask)

        both_storage = [positive_storage,negative_storage]

        for i,(g,m,dataset) in enumerate(zip(groups,metrics,both_storage)): 
            result = None
            # collect positive and negative indices based on args
            if (g == 'PC' and coding) or (g == 'NC' and not coding):
                # if an argmax has already been sampled from the same sequence, take a random nonoverlapping index
                if i==1 and m=='random' and groups[0] == groups[1]:
                    excluded = positive_storage[-1][1]
                    selected = select_index(subarray,mode=m,excluded=excluded)
                    adjusted = adjust_indices_by_region(selected,start,end,region)
                    max_in_window =  np.max(subarray[selected-10:selected+11].tolist())
                    argmax_in_window = np.argmax(subarray[selected-10:selected+11])
                    result = (tscript,adjusted,max_in_window,argmax_in_window,start,end,len(seq),unreduced_scores)
                # otherwise restricted only by mode
                else:
                    selected = select_index(subarray,mode=m) 
                    adjusted = adjust_indices_by_region(selected,start,end,region)
                    max_in_window = np.max(subarray[selected-10:selected+11].tolist())
                    argmax_in_window = np.argmax(subarray[selected-10:selected+11])
                    result = (tscript,adjusted,max_in_window,argmax_in_window,start,end,len(seq))
            
            # None type means length minimums were not passed so skip
            if result is not None:
                dataset.append(result) 
    
    return positive_storage, negative_storage

def indices_to_substrings(index_list,motif_fasta,df,region):
    
    storage = []
    sequences = []
    
    # ingest top k indexes from attribution/attention
    for tscript,idx,score,offset,s,e,l in index_list:
        seq = df.loc[tscript,'RNA']
        # enforce uniform window
        left_bound = 10
        right_bound = 10
        start = idx-left_bound
        end = idx+right_bound+1
        substr = seq[start:end]
        description = f"offset={offset},loc[{start}:{end}],ORF[{s}:{e}],tscript_len={l}"
        record = SeqRecord(Seq(substr),
                                id=f'{tscript}-kmer',
                                description=description)
        sequences.append(record)

    with open(motif_fasta,'w') as outFile:
        SeqIO.write(sequences, outFile, "fasta")

def run_attributions(saved_file,df,parent_dir,groups,metrics,mode="attn",layer_idx=0,head_idx=0,reduction='PC',region="full",mask=False):

    if mode == "attn":
        attr_name = f'EDA_layer{layer_idx}head{head_idx}'
    else:
        attr_name = os.path.split(saved_file)[1]
        fields = attr_name.split('.')
        attr_name = '_'.join(fields[:-1])

    prefix = f'{parent_dir}/{attr_name}/' 
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    
    # results files
    positive_motifs_file = prefix+"positive_motifs.fa"
    negative_motifs_file = prefix +"negative_motifs.fa"
    hist_file = prefix+"pos_hist.svg"
    positive_indices, negative_indices = top_indices(saved_file,df,groups,metrics,mode=mode,head_idx=head_idx,reduction_mode=reduction,region=region,apply_mask=mask)
    indices_to_substrings(positive_indices,positive_motifs_file,df,region)
    indices_to_substrings(negative_indices,negative_motifs_file,df,region)
    cmd = 'streme -p {}positive_motifs.fa -n {}negative_motifs.fa -oc {}streme_out -rna -minw 3 -maxw 9 -pvt 1e-2 -patience 3' 
    print(cmd.format(*[prefix]*3))

def tuple_list_to_df(indices):
    
    tscripts = [x[0] for x in indices]
    starts = [x[1] for x in indices]
    return pd.DataFrame({'ID' : tscripts, 'start' : starts})

def run(attributions,df,attr_dir,g,m,mode,reduction,region,mask):

    # reduction defines how L x 4 mutations are reduced to L x 1 importances, see reduce_over_mutations()
    # g[0] and g[1] are transcript type for pos and neg sets
    # m[0] and m[1] are method for selecting loci of interest for pos and neg sets
    negative_reduction_label = '' if m[1] == 'random' else f'-{reduction}'   
    positive_reduction_label = '' if m[0] == 'random' else f'-{reduction}'   
    a = f'pos={g[0]}.{m[0]}{positive_reduction_label}'
    b = f'neg={g[1]}.{m[1]}{negative_reduction_label}'
    trial_name = f'{a}_{b}'
    # build directories and run
    best_BIO_dir = f'{attr_dir}/best_seq2seq_{region}_{trial_name}'
    if not os.path.isdir(best_BIO_dir):
        os.mkdir(best_BIO_dir)
    run_attributions(attributions,df,best_BIO_dir,g,m,mode,reduction=reduction,region=region,mask=mask)

def attribution_loci_pipeline(): 

    args, unknown_args = parse_config()
    
    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    train_file = os.path.join(args.data_dir,args.train_prefix+'.csv')
    val_file = os.path.join(args.data_dir,args.val_prefix+'.csv')
    df_test = pd.read_csv(test_file,sep='\t').set_index('ID')
    df_train = pd.read_csv(train_file,sep='\t').set_index('ID')
    
    # load attribution files from config
    best_BIO_EDA = build_EDA_file_list(args.best_BIO_EDA,args.best_BIO_DIR)
    best_EDC_EDA = build_EDA_file_list(args.best_EDC_EDA,args.best_EDC_DIR)

    output_dir = build_output_dir(args)
    mask = '--mask' in unknown_args
    if mask:
        attr_dir = f'{output_dir}/attr_masked'
    else: 
        attr_dir = f'{output_dir}/attr_unmasked'

    # make subdir for attribution loci results  
    if not os.path.isdir(attr_dir):
        os.mkdir(attr_dir)

    # load attribution files from config
    best_BIO_EDA = build_EDA_file_list(args.best_BIO_EDA,args.best_BIO_DIR)
    best_EDC_EDA = build_EDA_file_list(args.best_EDC_EDA,args.best_EDC_DIR)
    
    mode = 'MDIG-train' if '--mdig' in unknown_args else 'ISM-test'
    if mode == 'ISM-test': 
        prefix = args.test_prefix.replace('test','test_RNA')
    elif mode == 'MDIG-train': 
        prefix = args.train_prefix.replace('train','train_RNA')
    
    best_BIO_mdig = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.MDIG.npz')
    best_EDC_mdig = os.path.join(args.best_EDC_DIR,f'{prefix}.{args.reference_class}.{args.position}.MDIG.npz')
    best_BIO_grad = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.grad.npz')
    best_EDC_grad = os.path.join(args.best_EDC_DIR,f'{prefix}.{args.reference_class}.{args.position}.grad.npz')
    best_BIO_ISM = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.ISM.npz')
   
    groups = [['PC','NC'],['NC','PC'],['PC','PC'],['NC','NC']]
    same_selection_methods = ['argmax','random']
    cross_selection_methods = ['argmax','argmax']
    reduction_methods = ['PC','NC']
    regions = ['5-prime','CDS','3-prime']
    
    for i,g in enumerate(groups):
        for reduction in reduction_methods:
            for region in regions:
                m = same_selection_methods if i>1 else cross_selection_methods
                if mode == 'ISM-test': 
                    run(best_BIO_ISM,df_test,attr_dir,g,m,'ISM',reduction=reduction,region=region,mask=mask)
                elif mode == 'MDIG-train': 
                    run(best_BIO_mdig,df_train,attr_dir,g,m,'MDIG',reduction=reduction,region=region,mask=mask)
   
    # now do the random-only search
    groups = [['PC','NC'],['NC','PC']]
    selection_methods = ['random','random']
    reduction = 'random' 
    for g in groups:
        for region in regions:
            if mode == 'ISM-test': 
                run(best_BIO_ISM,df_test,attr_dir,g,selection_methods,'ISM',reduction=reduction,region=region,mask=mask)
            elif mode == 'MDIG-train': 
                run(best_BIO_mdig,df_train,attr_dir,g,selection_methods,'MDIG',reduction=reduction,region=region,mask=mask)

if __name__ == "__main__":
    
    attribution_loci_pipeline() 
