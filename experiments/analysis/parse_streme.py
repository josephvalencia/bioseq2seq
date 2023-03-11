import xml.etree.ElementTree as ET
import re, os, sys
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess

from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from utils import parse_config

def parse_xml(xml_file,trial,trial_type,head,region):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    motifs = None
    total_pos_count = 0
    total_neg_count = 0
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
        if child.tag == "motifs":
            motifs = child
            break
    if motifs is not None and model is not None:
        results = []
        # iterate through motifs
        for m in motifs:
            attrib = m.attrib
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
            entry = {'Trial' : trial, 'trial_type' : trial_type, 'Region' : region, 'motif_id' : trial+'_'+motif_id,
                    'p-value' : pval, 'Information Content' : total_IC, 'pos_sites' : pos_sites,
                    'neg_sites' : neg_sites, 'total_sites' : total_sites, 'PWM' : prob_df.values} 
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
    return '<img src="'+ path + '" width="240" >'

def float_formatter(value):
    return '{:.2f}'.format(value)

def scientific_formatter(value):
    return '{:.2E}'.format(value)

def ratio_formatter(value):
    return '{}/{}\n({:.1f}%)'.format(value[0],value[1],100*value[0]/value[1])

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
    
    columns={'Information Content' : 'Information',
            'total_sites' : 'Total Sites',
            'Positive Set' : 'Positive Set (sites)',
            'Negative Set' : 'Negative Set (sites)',
            'pos_sites' :'Pos. Sites',
            'neg_sites': 'Neg. Sites'}
    df = df.rename(columns=columns)
    
    final_cols = ['Region','Positive Set (sites)','Negative Set (sites)','Pos. Sites',
                    'Neg. Sites', 'E-value','p-value','Information','Logo','Cluster']
    df = df[final_cols] 
    
    formatters = {'Logo' : path_to_image_html,'Information': float_formatter,
             'Pos. Sites' : ratio_formatter, 'Neg. Sites' : ratio_formatter, 
             'E-value' : scientific_formatter,'p-value' : scientific_formatter}
    df.to_html(html_file,escape=False,formatters=formatters,index=False)
    print(f'HTML summary saved at {html_file}')

def renumber_clusters(clusters):

    next_val = 1
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

def process(storage,args,parent,prefix,mode):
    
    df = pd.DataFrame(storage)
    df['E-value'] = [x*len(df) for x in df['p-value'].tolist()]
    significant = df[df['E-value'] < 1e-3].copy()
    
    print('SIGNIFICANT',significant)    
    meme_file = os.path.join(parent,f'{prefix}_{args.reference_class}_{args.position}_{mode}_significant_motifs.meme')
    write_to_meme(significant,meme_file)
    cluster_df = tomtom_clustering(meme_file)
    original_motifs = subtract_random_matches(significant,meme_file)
    print('NOVEL ISM DETECTIONS',original_motifs)
    print('CLUSTERS',cluster_df[['motif_id','Cluster']])

    significant = original_motifs
    significant = significant.merge(cluster_df,on='motif_id',how='left')
    for motif_id,PWM,region in zip(significant['motif_id'],significant['PWM'],significant['Region']):
        filename = f'{parent}/{prefix}_{args.reference_class}_{args.position}_{mode}_{region}_{motif_id}_logo.svg'
        save_pwm_logo(PWM,filename)
    
    ''''
    nonredundant = []
    for name,group in with_cluster.groupby('cluster'):
        group = group.sort_values(by=['E-value','Information Content'],ascending=[True,False]) 
        best = group.iloc[0]
        for motif_id,PWM in zip(group['motif_id'],group['PWM']): 
            filename = f'{parent}/{prefix}_{args.reference_class}_{args.position}_{mode}_{motif_id}_logo.svg'
            save_pwm_logo(PWM,filename)
        nonredundant.append(best)
    '''
    
    significant['Cluster' ] = renumber_clusters(significant['Cluster'].tolist())
    logo_list = [f'{prefix}_{args.reference_class}_{args.position}_{mode}_{y}_{x}_logo.svg' for x,y in zip(significant['motif_id'],significant['Region'])]
    significant['Consensus'] = [x.split('-')[-1] for x in significant['motif_id']]
    significant['Logo'] = logo_list
    significant['Positive Set'] = [readable_dataset_name(x.split('_')[0]) for x in significant['Trial']]
    significant['Negative Set'] = [readable_dataset_name(x.split('_')[1]) for x in significant['Trial']]
    
    meme_file = os.path.join(parent,f'{prefix}_{args.reference_class}_{args.position}_{mode}_significant_ALL_motifs.meme')
    write_to_meme(significant,meme_file)
    html_file = os.path.join(parent,f'{prefix}_{args.reference_class}_{args.position}_{mode}_significant_ALL_motifs.html')
    write_to_html(significant,html_file) 
  
    '''
    xlsx_file = os.path.join(parent,f'{prefix}_{args.reference_class}_{args.position}_{mode}_significant_ALL_motifs.xlsx')
    writer = pd.ExcelWriter(xlsx_file, engine='xlsxwriter')
    significant.style.set_properties(text_align='center')
    significant.to_excel(writer, sheet_name='Sheet1')
    worksheet = writer.sheets['Sheet1']
    worksheet.set_default_row(75)
    (max_row, max_col) = significant.shape
    for i,logo in enumerate(logo_list):
        worksheet.insert_image(i+1,max_col+1,f'{parent}/{logo}',{'x_scale' : 0.5,'y_scale' : 0.5})
    writer.close()
    print(f'Excel format written at {xlsx_file}') 
    '''

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
    
    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}'
    attr_dir  =  f'{output_dir}/attr'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # make subdir for attribution loci results  
    if not os.path.isdir(attr_dir):
        os.mkdir(attr_dir)
    
    groups = [['PC','NC'],['NC','PC'],['PC','PC'],['NC','NC']]
    same_selection_methods = ['argmax','random']
    cross_selection_methods = ['argmax','argmax']
    reduction_methods = ['PC','NC']
    regions = ['5-prime','CDS','3-prime','full']
    
    storage = []
    mode = 'ISM-test'
    #mode = 'MDIG-train'
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
