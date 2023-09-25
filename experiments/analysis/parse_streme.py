import xml.etree.ElementTree as ET
import re, os
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
from Bio.motifs.matrix import PositionWeightMatrix as PWM
from Bio import SeqIO
from scipy.cluster.hierarchy import fcluster, linkage
from utils import parse_config, build_output_dir
from collections import Counter

def parse_xml(xml_file,trial,trial_type,head,region):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    motifs = None
    total_pos_count = 0
    total_neg_count = 0
    background = None
    for child in root:
        if child.tag == "model":
            model = child
            for inner in model:
                if inner.tag == 'train_positives':
                    total_pos_count += int(inner.attrib['count'])
                elif inner.tag == 'test_positives':
                    total_pos_count += int(inner.attrib['count'])
                elif inner.tag == 'train_negatives':
                    total_neg_count += int(inner.attrib['count'])
                elif inner.tag == 'test_negatives':
                    total_neg_count += int(inner.attrib['count'])
                elif inner.tag == 'background_frequencies':
                    alphabet_array = inner[0]
                    background = {} 
                    for val in alphabet_array:
                        base = val.attrib['letter_id']
                        base = 'T' if base == 'U' else base
                        background[base] = float(val.text)
        if child.tag == "motifs":
            motifs = child
            break
    
    if motifs is not None and model is not None:
        results = []
        # iterate through motifs
        for m in motifs:
            attrib = m.attrib
            site_dist = [float(x) for x in attrib['site_distr'].split()]
            motif_id = attrib['id']
            pval = float(attrib['test_pvalue'])
            pos = int(attrib['train_pos_count'])+int(attrib['test_pos_count'])
            neg = int(attrib['train_neg_count'])+int(attrib['test_neg_count'])
            pos_sites = (pos,total_pos_count)
            neg_sites = (neg,total_neg_count)
            total_sites = (pos+neg,total_pos_count+total_neg_count)
            storage = []
            for pos in m.iter('pos'):
                storage.append(pos.attrib)
            prob_df = pd.DataFrame(storage)
            dtypes = {'A' : float , 'G' : float , 'C' : float , 'U' : float}
            prob_df = prob_df.astype(dtypes)
            info_df = logomaker.transform_matrix(df=prob_df,from_type='probability',to_type='information')
            total_IC = info_df.values.sum()
            entry = {'filename' : xml_file, 'short_motif_id' : motif_id,'Trial' : trial, 
                    'trial_type' : trial_type, 'Region' : region, 'motif_id' : trial+'_'+motif_id,
                    'p-value' : pval, 'Information Content' : total_IC, 'pos_sites' : pos_sites,
                    'neg_sites' : neg_sites, 'total_sites' : total_sites, 'PWM' : prob_df.values,'dist' : site_dist,'background' : background} 
            results.append(entry)
        return results
    else:
        print(f'No motifs found in {head}')
        return [] 
        
def write_to_meme(df,meme_file):
 
    all_pwms = df.to_dict('records')
    # save to meme
    with open(meme_file,'w+') as f:
        f.write('MEME version 5\n')
        f.write('\n')
        f.write('ALPHABET= ACGT\n')
        f.write('\n')
        f.write('strands: + -\n')
        f.write('\n')
        f.write('Background letter frequencies\n')
        f.write('A 0.27 C 0.23 G 0.23 T 0.27\n')
        f.write('\n')

    for entry in all_pwms:
        with open(meme_file,'a') as f:
            f.write('MOTIF '+entry['motif_id']+'\n')
            f.write('letter-probability matrix: alength= 4 w= {} nsites={} E={}\n'.format(entry['PWM'].shape[0],entry['total_sites'][0],entry['E-value']))
        with open(meme_file,'ab') as f:
            np.savetxt(f, entry['PWM'],fmt='%.6f')
        with open(meme_file,'a') as f:
            f.write('\n')
    print(f'Motifs saved at {meme_file}')

def tomtom_clustering(meme_file):
    
    '''all-by-all comparison and clustering. Code taken from https://github.com/jvierstra/motif-clustering/blob/master/Workflow_v2.1beta-human.ipynb'''
    
    parent = os.path.split(meme_file)[0]
    cmd = ['tomtom',meme_file,meme_file,'-min-overlap','1','-norc','-oc',f'{parent}/tomtom']
    print(' '.join(cmd)) 
    subprocess.run(cmd) 
    results = pd.read_csv(f'{parent}/tomtom/tomtom.tsv',sep='\t',skipfooter=3,engine='python')
    sim = results.pivot_table(index='Query_ID', columns='Target_ID', values='E-value', fill_value=np.nan)
    x = sim.values
    w = np.triu(x) +  np.triu(x, 1).T
    v = np.tril(x) + np.tril(x, -1).T
    sim.iloc[:,:] = np.nanmin(np.dstack([w, v]), axis=2)
    sim.fillna(100, inplace=True)
    sim = -np.log10(sim)
    sim[np.isinf(sim)] = 10
    Z = linkage(sim, method = 'complete', metric = 'correlation')
    cl = fcluster(Z, 0.7, criterion='distance')
    print(f'Number of motif clusters: {max(cl)}')

    motif_annot_df = pd.DataFrame({'motif_id' : sim.index, 'Cluster' : cl})
    return motif_annot_df

def subtract_random_matches(df,meme_file):
    
    parent = os.path.split(meme_file)[0]
    storage = [] 
    for group,df_group in df.groupby('Region'):
        df_other_class = df_group[df_group['trial_type'] == 'other_class']
        df_purely_random = df_group[df_group['trial_type'] == 'purely_random']
        df_same_seq = df_group[df_group['trial_type'] == 'same_seq']
        other_meme = f'{parent}_{group}_other_class.meme'
        random_meme = f'{parent}_{group}_random.meme'
        # same_seq does not need to be cross-checked 
        storage.append(df_same_seq) 
        # both must exist to run TOMTOM safely 
        if len(df_other_class) > 0  and len(df_purely_random) > 0 :
            write_to_meme(df_other_class,other_meme)
            write_to_meme(df_purely_random,random_meme)
            cmd = ['tomtom',other_meme,random_meme,'-min-overlap','1','-norc','-oc',f'{parent}/tomtom_{group}']
            print(' '.join(cmd)) 
            subprocess.run(cmd) 
            results = pd.read_csv(f'{parent}/tomtom_{group}/tomtom.tsv',sep='\t',skipfooter=3,engine='python')
            sim = results.pivot_table(index='Query_ID', columns='Target_ID', values='E-value', fill_value=np.nan)
            non_unique = sim.reset_index()['Query_ID'].tolist()
            print(f'Discarding {non_unique} due to match with purely random') 
            unique = df_other_class[~df_other_class['motif_id'].isin(non_unique)] 
            storage.append(unique)
        # if no random detections, other class is automatically novel
        elif len(df_other_class) > 0:
            storage.append(df_other_class)
    
    with_clusters = pd.concat(storage)
    return with_clusters

def path_to_image_html(path):
    #return '<img src="'+ path + '" width="240" >'
    return '<img src="'+ path + '" width="160" >'

def path_to_image_html_skinny(path):
    #return '<img src="'+ path + '" width="120" >'
    return '<img src="'+ path + '" width="90" >'

def float_formatter(value):
    return '{:.2f}'.format(value)

def scientific_formatter(value):
    return '{:.2E}'.format(value)

def ratio_formatter(value):
    return '{}/{}\n({:.1f}%)'.format(value[0],value[1],100*value[0]/value[1])

def pct_formatter(value):
    return '{:.1f}%'.format(value)

def save_site_dist(dist,maxes,filename,figsize,mode):

    diff = 21 - len(dist)
    dist += [0] * diff
    plt.figure(figsize=figsize) 
    bins = np.arange(0,21)
    plt.bar(bins,dist,label='Motif Start')
   
    diff = 21 - len(maxes)
    maxes += [0] * diff
    plt.step(bins,maxes,where='mid',c='red',label=f'Max {mode}')
    plt.xlabel('Pos. in window of max importance') 
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_frame_dist(dist,filename,figsize):

    bar_colors = ['tab:blue','tab:orange','tab:red']
    plt.figure(figsize=figsize) 
    bins = [0,1,2]
    plt.bar(bins,dist,color=bar_colors)
    plt.xticks(bins,bins)
    plt.xlabel('Frame')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_pos_hist(hist_info,filename,figsize):

    plt.figure(figsize=figsize) 
    vals,bins = hist_info 
    plt.hist(vals,bins=bins)
    plt.xlabel('Fraction of region')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_pwm_logo(pwm,filename):

    prob_df = pd.DataFrame(pwm,columns=['A','C','G','U'])
    dtypes = {'A' : float , 'G' : float , 'C' : float , 'U' : float}
    prob_df = prob_df.astype(dtypes)
    info_df = logomaker.transform_matrix(df=prob_df,from_type='probability',to_type='information')
    
    # create figure
    num_cols = len(prob_df)
    num_rows = 4
    height_per_row = 0.5
    width_per_col = 0.8 
    figsize=[width_per_col * num_cols, height_per_row * num_rows]
    logo = logomaker.Logo(info_df,figsize=figsize,alpha=1.0)
    logo.ax.set_ylim([0,2])
    logo.ax.set_yticks([0,0.5,1.0,1.5,2],fontsize=8)
    logo.ax.set_xticks(list(range(num_cols)),fontsize=8)
    sns.despine() 
    plt.ylabel('Bits',fontsize=10)
    plt.tight_layout()
    print(f'saving {filename}')
    plt.savefig(filename)
    plt.close()
    return figsize

def calculate_fractions(location_list,length_list,n_bins=50):
    
    bins = np.asarray([(1/n_bins)*x for x in range(n_bins+1)])
    fractions = [ x/y for x,y in zip(location_list,length_list)] 
    return fractions,bins

def readable_dataset_name(dataset):
    
    # skip pos=/neg=
    fields = dataset[4:].split('.')
    tscript_type = 'mRNAs' if fields[0] == "PC" else 'lncRNAs'
    if fields[1] == 'random':
        name = f'{tscript_type} (random)'
    else: 
        tgt_class = 'PC' if 'PC' in fields[1] else 'NC'
        # HTML element for up arrow 
        name = f'{tscript_type} (&#8593 {tgt_class})'
    return name

def write_to_html(df,html_file):
    
    df['Motif #'] = list(range(len(df)))
    df['Region'] = ['ORF' if x == 'CDS' else x for x in df['Region']]
    
    columns={'Information Content' : 'Information',
            'total_sites' : 'Total Sites',
            'Positive Set' : 'Positive Set (sites)',
            'Negative Set' : 'Negative Set (sites)',
            'pos_sites' :'Pos. Sites',
            'neg_sites': 'Neg. Sites'}
    df = df.rename(columns=columns)
    final_cols = ['Motif #','Region','Positive Set (sites)','Negative Set (sites)','Pos. Sites',
                    'Neg. Sites', 'Cluster','Logo','Start site in region','Start site in window','Offset from ORF','E-value','p-value','Information']
    df = df[final_cols] 
    
    formatters = {'Logo' : path_to_image_html,'Start site in window' : path_to_image_html, 'Start site in region' : path_to_image_html,
            'Information': float_formatter,'Pos. Sites' : ratio_formatter, 'Neg. Sites' : ratio_formatter, 
             'E-value' : scientific_formatter,'p-value' : scientific_formatter,
             'Offset from ORF' : path_to_image_html_skinny}
    df.to_html(html_file,escape=False,formatters=formatters,index=False)
    print(f'HTML summary saved at {html_file}')

def write_to_excel(df,xlsx_file,parent):

    writer = pd.ExcelWriter(xlsx_file, engine='xlsxwriter')
    df.style.set_properties(text_align='center')
    df.to_excel(writer, sheet_name='Sheet1')
    worksheet = writer.sheets['Sheet1']
    worksheet.set_default_row(75)
    max_row, max_col = df.shape
    for i,logo in enumerate(df['Logo'].tolist()):
        worksheet.insert_image(i+1,max_col+1,f'{parent}/{logo}',{'x_scale' : 0.5,'y_scale' : 0.5})
    writer.close()
    print(f'Excel format written at {xlsx_file}') 

def renumber_clusters(clusters):

    next_val = 0
    remap = {}
    new_clusters = []
    for c in clusters:
        if c not in remap:
            new_clusters.append(next_val)
            remap[c] = next_val
            next_val += 1
        else:
            new_clusters.append(remap[c])
    return new_clusters

def fasta_to_df(fasta_file):

    storage = []
    search_string = "offset=(\d*),loc\[(\d*):(\d*)\],ORF\[(\d*):(\d*)\],tscript_len=(\d*)"
    
    for record in SeqIO.parse(fasta_file,'fasta'):
        description = record.description
        match = re.search(search_string,description)
        if match is not None:
            entry = {'seq_ID' : record.id, 'seq' : str(record.seq), 'offset' : int(match.group(1)),
                    'start' : int(match.group(2)), 'end' : int(match.group(3)),
                    'start_ORF' : int(match.group(4)), 'end_ORF' : int(match.group(5)),'tscript_len' : int(match.group(6))}
            storage.append(entry)

    df = pd.DataFrame(storage)
    return df

def add_positional_info(df):

    in_frame_pct = []
    frames = []
    maxes = []
    positions = []

    for xml_file,region,motif_name,pwm,background in zip(df['filename'],df['Region'],
                                                df['short_motif_id'],df['PWM'],df['background']):
        # load positive seqs and STREME tsv 
        seq_file = os.path.join(os.path.split(xml_file)[0],'sequences.tsv')
        fasta_file = os.path.join(os.path.split(os.path.split(xml_file)[0])[0],'positive_motifs.fa') 
        seq_df = pd.read_csv(seq_file,sep='\t')
        raw_df = fasta_to_df(fasta_file)
        
        # use true positive motifs 
        true_pos_seqs = (seq_df['motif_ID'] == motif_name) & (seq_df['seq_Class'] == 'tp')
        seq_df = seq_df[true_pos_seqs] 
        raw_df = raw_df.merge(seq_df,on='seq_ID')
        
        # build PSSM from motif found by STREME
        quasi_counts = (10000 * pwm).astype(int)
        alphabet = ['A','C','G','T']
        values = {alphabet[i] : quasi_counts[:,i].tolist() for i in range(4)} 
        pssm = PWM(alphabet=alphabet,counts=values).log_odds(background=background)
        bad_count = 0
        position_list = []
        
        # find position and frame of best PSSM match from true positive seqs
        for tscript,seq,motif,max_score in zip(raw_df['seq_ID'],raw_df['seq'],raw_df['motif_ID'],raw_df['seq_Score']):
            best = 0
            best_position = -1
            width = len(motif.split('-')[-1])
            for position, score in pssm.search(seq,threshold=0.0):
                if position >= 0 and score > best: 
                   best = score
                   best_position = position
            if best_position != -1: 
                match = seq[best_position:best_position+width]
                position_list.append(best_position)
            else:
                bad_count +=1
                position_list.append(np.nan)

        
        # add match pos to start pos of window 
        match_position = [s+pos for s,pos in zip(raw_df['start'],position_list)]
        raw_df['match_position'] = match_position
     
        '''
        if region == 'CDS':
            test = raw_df[raw_df['motif_ID'] == '1-NRNCUGGNN']
            full_df = pd.read_csv('data/mammalian_200-1200_train.csv',sep='\t').set_index('ID')
            counter = Counter()
            for tscript,seq,start_orf,start,match_pos in zip(test['seq_ID'],test['seq'],test['start_ORF'],test['start'],test['match_position']):
                remainder = (start - start_orf) % 3
                frame = (3-remainder) % 3 
                tscript = tscript.replace('-kmer','')
                protein = full_df.loc[tscript,'Protein']
                codon_idx = (start - start_orf) // 3 
                protein_slice = protein[codon_idx:codon_idx+7]
                translated = Seq(seq[frame:]).translate()
                counter.update([c for c in str(translated)])
                print(tscript,seq,frame,translated,protein_slice)
            print(counter.most_common())
        '''

        # count pos of maximum importance in window
        offsets = raw_df['offset']
        counter = Counter()
        counter.update(raw_df['offset'])
        offset_counts = [counter[i] for i in range(21)]
        maxes.append(offset_counts) 
        
        # build histogram of pos within region
        items = list(zip(raw_df['match_position'],raw_df['start_ORF'],raw_df['end_ORF'],raw_df['tscript_len']))
        region_lens = [length_by_region(region,s,e,l) for i,s,e,l in items]
        rel_pos = [rel_position_by_region(i,region,s,e,l) for i,s,e,l in items]
        fractions,bins = calculate_fractions(rel_pos,region_lens)
        positions.append((fractions,bins))
        
        # count frames relative to start
        frame_diff = (raw_df['match_position']  - raw_df['start_ORF'])  % 3 
        frame_start = (3-frame_diff) % 3 
        counter = Counter()
        counter.update(frame_start.tolist())
        frame_counts = [counter[i] for i in range(3)]
        frames.append(frame_counts) 

    df['maxes'] = maxes
    df['frames'] = frames
    df['rel_pos_hist'] = positions
    return df

def rel_position_by_region(idx,region,start_ORF,end_ORF,tscript_len):
    
    if region == '5-prime':
        return idx
    elif region == 'CDS':
        return idx - start_ORF
    elif region == '3-prime':
        return idx - end_ORF
    else:
        return idx

def length_by_region(region,start_ORF,end_ORF,tscript_len):
    
    if region == '5-prime':
        return start_ORF
    elif region == 'CDS':
        return end_ORF - start_ORF
    elif region == '3-prime':
        return tscript_len - end_ORF
    else:
        return tscript_len

def process(storage,args,parent,prefix,mode):
    
    all_df = pd.DataFrame(storage)
    print(all_df) 
    all_df['E-value'] = [x*len(all_df) for x in all_df['p-value'].tolist()]
    significant = all_df[all_df['E-value'] < 1e-3].copy()
    print('SIGNIFICANT',significant) 
 
    motifs_dir = os.path.join(parent,f'{prefix}_{args.reference_class}_{args.position}_{mode}_PLOTS')
    print(motifs_dir) 
    if not os.path.exists(motifs_dir):
        os.mkdir(motifs_dir)
     
    file_prefix = os.path.join(parent,f'{prefix}_{args.reference_class}_{args.position}_{mode}')

    meme_file = f'{file_prefix}_significant_motifs.meme'
    write_to_meme(significant,meme_file)
    cluster_df = tomtom_clustering(meme_file)
    original_motifs = subtract_random_matches(significant,meme_file)
    print('NOVEL ISM DETECTIONS',original_motifs)
    print('CLUSTERS',cluster_df[['motif_id','Cluster']])
   
    df = original_motifs
    df = df.merge(cluster_df,on='motif_id',how='left')
    df = add_positional_info(df) 
    print(df) 
    for motif_id,PWM,dist,rel_pos_hist,frames,maxes,region in zip(df['motif_id'],df['PWM'],df['dist'],
                                            df['rel_pos_hist'],df['frames'],df['maxes'],df['Region']):
        filename = f'{motifs_dir}/{region}_{motif_id}_logo.svg'
        figsize = save_pwm_logo(PWM,filename)
        filename = f'{motifs_dir}/{region}_{motif_id}_start_hist.svg'
        save_site_dist(dist,maxes,filename,[4.0,2.0],mode)
        filename = f'{motifs_dir}/{region}_{motif_id}_rel_pos_hist.svg'
        save_pos_hist(rel_pos_hist,filename,[4.0,2.0])
        filename = f'{motifs_dir}/{region}_{motif_id}_frame_hist.svg'
        save_frame_dist(frames,filename,[2.4,2])

    short = f'{prefix}_{args.reference_class}_{args.position}_{mode}_PLOTS/'
    items = list(zip(df['motif_id'],df['Region']))
    # add figures and labels
    print(df) 
    df['Cluster' ] = renumber_clusters(df['Cluster'].tolist())
    df['Consensus'] = [x.split('-')[-1] for x in df['motif_id']]
    df['Logo'] = [f'{short}/{y}_{x}_logo.svg' for x,y in items]
    df['Start site in window'] = [f'{short}/{y}_{x}_start_hist.svg' for x,y in items]
    df['Offset from ORF'] = [f'{short}/{y}_{x}_frame_hist.svg' for x,y in items]
    df['Start site in region'] = [f'{short}/{y}_{x}_rel_pos_hist.svg' for x,y in items]
    df['Positive Set'] = [readable_dataset_name(x.split('_')[0]) for x in df['Trial']]
    df['Negative Set'] = [readable_dataset_name(x.split('_')[1]) for x in df['Trial']]
    print(df) 
    
    meme_file = f'{file_prefix}_significant_ALL_motifs.meme'
    write_to_meme(df,meme_file)
    html_file = f'{file_prefix}_significant_ALL_motifs.html'
    write_to_html(df,html_file) 

def get_trial_name(g,m,reduction):

    # reduction defines how L x 4 mutations are reduced to L x 1 importances, see reduce_over_mutations()
    # g[0] and g[1] are transcript type for pos and neg sets
    # m[0] and m[1] are method for selecting loci of interest for pos and neg sets
    positive_reduction_label = '' if m[0] == 'random' else f'-{reduction}'   
    negative_reduction_label = '' if m[1] == 'random' else f'-{reduction}'   
    a = f'pos={g[0]}.{m[0]}{positive_reduction_label}'
    b = f'neg={g[1]}.{m[1]}{negative_reduction_label}'
    trial_name = f'{a}_{b}'
    return trial_name

if __name__ == "__main__":
    
    args, unknown_args = parse_config()

    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    train_file = os.path.join(args.data_dir,args.train_prefix+'.csv')
    val_file = os.path.join(args.data_dir,args.val_prefix+'.csv')
    df_test = pd.read_csv(test_file,sep='\t').set_index('ID')
    df_train = pd.read_csv(train_file,sep='\t').set_index('ID')
    
    output_dir = build_output_dir(args) 

    mask = '--mask' in unknown_args
    if mask:
        attr_dir = f'{output_dir}/attr_masked'
    else: 
        attr_dir = f'{output_dir}/attr_unmasked'
    
    mode = 'MDIG-train' if '--mdig' in unknown_args else 'ISM-test'
    
    groups = [['PC','NC'],['NC','PC'],['PC','PC'],['NC','NC']]
    same_selection_methods = ['argmax','random']
    cross_selection_methods = ['argmax','argmax']
    reduction_methods = ['PC','NC']
    regions = ['5-prime','CDS','3-prime']
    
    storage = []
    
    for i,g in enumerate(groups):
        for region in regions: 
            for reduction in reduction_methods:
                trial_type = 'same_seq' if i>1 else 'other_class'
                m = same_selection_methods if i>1 else cross_selection_methods
                trial_name = get_trial_name(g,m,reduction)
                BIO_dir = f'{attr_dir}/best_seq2seq_{region}_{trial_name}/'
                
                if mode == 'ISM-test':  
                    prefix = args.test_prefix.replace('test','test_RNA')
                    best_BIO_ISM = os.path.join(BIO_dir,f'{prefix}_{args.reference_class}_{args.position}_ISM','streme_out/streme.xml')
                    if os.path.exists(best_BIO_ISM): 
                        results = parse_xml(best_BIO_ISM,trial_name,trial_type=trial_type,head='ISM',region=region)
                        storage.extend(results)
                elif mode == 'MDIG-train': 
                    prefix = args.train_prefix.replace('train','train_RNA')
                    best_BIO_MDIG = os.path.join(BIO_dir,f'{prefix}_{args.reference_class}_{args.position}_MDIG','streme_out/streme.xml')
                    if os.path.exists(best_BIO_MDIG): 
                        results = parse_xml(best_BIO_MDIG,trial_name,trial_type=trial_type,head='MDIG',region=region)
                        storage.extend(results)
                else:
                    print(f'{best_BIO_ISM} does not exist')
    
    # purely random search
    groups = [['PC','NC'],['NC','PC']]
    selection_methods = ['random','random']
    reduction = 'random'
    trial_type = 'purely_random'
    for g in groups:
        for region in regions:
            trial_name = get_trial_name(g,selection_methods,reduction) 
            BIO_dir = f'{attr_dir}/best_seq2seq_{region}_{trial_name}/'
            
            if mode == 'ISM-test':  
                prefix = args.test_prefix.replace('test','test_RNA')
                best_BIO_ISM = os.path.join(BIO_dir,f'{prefix}_{args.reference_class}_{args.position}_ISM','streme_out/streme.xml')
                if os.path.exists(best_BIO_ISM): 
                    results = parse_xml(best_BIO_ISM,trial_name,trial_type=trial_type,head='ISM',region=region)
                    storage.extend(results) 
            elif mode == 'MDIG-train': 
                prefix = args.train_prefix.replace('train','train_RNA')
                best_BIO_MDIG = os.path.join(BIO_dir,f'{prefix}_{args.reference_class}_{args.position}_MDIG','streme_out/streme.xml')
                if os.path.exists(best_BIO_MDIG): 
                    results = parse_xml(best_BIO_MDIG,trial_name,trial_type=trial_type,head='MDIG',region=region)
                    storage.extend(results) 
    
    if mode == 'ISM-test':  
        process(storage,args,attr_dir,prefix,'ISM')
    elif mode == 'MDIG-train': 
        process(storage,args,attr_dir,prefix,'MDIG')
