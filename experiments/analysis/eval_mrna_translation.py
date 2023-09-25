import sys,re,os
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import getLongestORF,parse_config, setup_fonts, build_output_dir,palette_by_model
from tqdm import tqdm

def parse_needle_results(entry,save_dir):

    match_tscript = re.search('ID: (.*) coding_prob : ([0|1]\.\d*)',entry[0])
    #match_tscript = re.search('ID: (.*)',entry[0])
    is_coding = lambda x : x.startswith('NM') or x.startswith('XM')
    
    max_results = None
    true_results = None
    tscript = None
    alignment = None

    if match_tscript:
        tscript = match_tscript.group(1)
        if is_coding(tscript): 
            
            needle_file = f'{save_dir}/{tscript}_ALL.needle'
            if os.path.exists(needle_file): 
                with open(needle_file,'r') as inFile:
                    data = inFile.read()
                max_results = max_identity(data)

            needle_file = f'{save_dir}/{tscript}_TRUE.needle'
            if os.path.exists(needle_file): 
                with open(needle_file,'r') as inFile:
                    data = inFile.read()
                identity_pattern = "# Identity:(\s*)(\d*)\/(\d*) \((.*)%\)\n"
                name_pattern = "# 2: (.*)\n"
                id_match = re.findall(identity_pattern,data)[0]
                name_match = re.findall(name_pattern,data)[0]
                true_results = (name_match,int(id_match[-2]),float(id_match[-1]))
                msa_list = list(AlignIO.parse(needle_file, "emboss"))
                alignment = msa_list[0]
    
    return tscript, max_results,alignment,true_results

def max_identity(data):
    
    identity_pattern = "# Identity:(\s*)(\d*)\/(\d*) \((.*)%\)\n"
    name_pattern = "# 1: (.*)\n"
    id_matches = re.findall(identity_pattern,data)
    name_matches = re.findall(name_pattern,data)
    scores = [(y,int(x[-2]),float(x[-1])) for x,y in zip(id_matches,name_matches)]
    scores = sorted(scores,key = lambda x : x[1],reverse=True) 
    scores = sorted(scores,key = lambda x : x[2],reverse=True)
    print(scores) 
    return scores[0]

def align_proteins(entry,save_dir,test_df):

    match_tscript = re.search('ID: (.*) coding_prob : ([0|1]\.\d*)',entry[0])
    is_coding = lambda x : x.startswith('NM') or x.startswith('XM')
    if match_tscript and is_coding(match_tscript.group(1)):
        tscript = match_tscript.group(1) 
        results = []
        match_pred = re.search('PRED: <PC>(.*) SCORE: (.*)',entry[1])
        # only evaluate true positives, ignore false negatives 
        if match_pred:
            results.append(make_record(f'{tscript}.peptide',match_pred.group(1)))
            predicted_file = f'{save_dir}/{tscript}_translation.fa'
            with open(predicted_file,'w') as outfile:
                SeqIO.write(results,outfile,"fasta")
            
            # original protein seq
            true_file = f'{save_dir}/{tscript}_true_PROTEIN.fa'
            if not os.path.exists(true_file): 
                seq = test_df.loc[tscript]['Protein']
                record = make_record(tscript,seq) 
                with open(true_file,'w') as outfile:
                    SeqIO.write([record],outfile,"fasta")
            
            # original RNA seq
            RNA_file = f'{save_dir}/{tscript}_RNA.fa'
            if not os.path.exists(RNA_file): 
                rna = test_df.loc[tscript]['RNA']
                record = make_record(tscript,rna) 
                with open(RNA_file,'w') as outfile:
                    SeqIO.write([record],outfile,"fasta")
       
            # 3-frame translation
            small_ORF_file = f'{save_dir}/{tscript}_ORF_translations.fa'
            if not os.path.exists(small_ORF_file):
                cmd_args = ['getorf','-sequence',RNA_file,'-outseq',small_ORF_file,'-find', '1','-noreverse']
                subprocess.run(cmd_args)

            orf_cmd = f"needle -asequence {predicted_file} -bsequence {small_ORF_file} -outfile {save_dir}/{tscript}_ALL.needle -gapopen 10 -gapextend 0.5 -brief"
            print(orf_cmd)
            prot_cmd = f"needle -asequence {predicted_file} -bsequence {true_file} -outfile {save_dir}/{tscript}_TRUE.needle -gapopen 10 -gapextend 0.5 -brief"
            print(prot_cmd)

def make_record(id,seq):
    return SeqRecord(Seq(seq),id=id,description='')

def align(filename,testfile):

    parent = os.path.split(filename)[0]
    save_dir = os.path.join(parent,'translations')
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir) 
  
    test_df = pd.read_csv(testfile,sep='\t')
    with open(filename) as inFile:
        lines = inFile.readlines()
        storage = []
        for i in tqdm(range(0,len(lines),6)):
            entry = lines[i:i+6]
            align_proteins(entry,save_dir,test_df)


def update_position_counts(count_array,len_array,alignment):

    pred_seq = alignment[0]
    true_seq = alignment[1]
    max_len = min(len(true_seq),len(count_array))
     
    for i in range(max_len):
        len_array[i] += 1 
        if pred_seq[i] == true_seq[i]:
            count_array[i] +=1
    #identity = sum([1 for a,b in zip(pred_seq,true_seq) if a == b])
    #print(f'% id = {identity/len(true_seq)}')

def parse(filename,testfile):
    
    records = []
    parent = os.path.split(filename)[0]
    save_dir = os.path.join(parent,'translations')
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir) 
   
    is_coding = lambda x : x.startswith('NM') or x.startswith('XM')
    test_df = pd.read_csv(testfile,sep='\t').set_index('ID')
    id_label =  'align_id'
    
    count_arr = [0]*60
    len_arr = [0]*60

    with open(filename) as inFile:
        lines = inFile.readlines()
        storage = []
        for i in tqdm(range(0,len(lines),6)):
            entry = lines[i:i+6]
            tscript,result,alignment,true_result = parse_needle_results(entry,save_dir)
            # true positive mRNA 
            if result and true_result: 
                # alignment with true protein must dominate all other ORFs 
                correct_ORF = true_result[-1] >= result[-1]
                print(f'result={result}, true_result={true_result}')
                update_position_counts(count_arr,len_arr,alignment)
                row = {'ID' : tscript, 'peptide' : result[0], 'Protein length': true_result[1],\
                        id_label : true_result[2], 'correct_ORF' : correct_ORF,'positive' : 'TP'}
                storage.append(row)
            # false negative mRNA 
            elif is_coding(tscript):
                row = {'ID' : tscript, 'peptide' : None, 'Protein length': 0,\
                        id_label : 0, 'correct_ORF' : False,'positive' : 'FN'}
                storage.append(row)
    
    df = pd.DataFrame(storage)
    homology = pd.read_csv("test_maximal_homology.csv")
    df = df.merge(homology,on='ID') 
    reduced = df['score'] <=80
    df = df.loc[reduced]
    
    TP_only = True 
    if TP_only:
        df = df[df['positive'] == 'TP']
    
    count_arr = np.asarray(count_arr)
    len_arr = np.asarray(len_arr)
    align_probs = count_arr / len_arr 
    correct_orf_count = sum([1 for x in df['correct_ORF'] if x])
    mean = df['align_id'].mean()
    std = df['align_id'].std()
    perfect_matches = df[df[id_label] == 100.0]
    
    support = len(df) 
    perfect = len(perfect_matches)
    print(f'calculated from {support} mRNAs, of which {correct_orf_count} ({100*correct_orf_count/support:.1f}%) predict the true CDS, {perfect} ({100*perfect/support:.1f}%) had a perfect protein decoding, % align id (mean={mean:.2f},std={std:.2f})')
    return {'trial' : filename, 'support' : support, 'correct_CDS' : correct_orf_count, 'perfect' : perfect, 'align_id_mean' : mean, 'align_id_std' : std, 'align_probs' : align_probs} 

def get_model_names(filename):
    model_list = [] 
    with open(filename) as inFile:
        model_list += [x.rstrip().replace('/','').replace('.pt','') for x in inFile.readlines()]
    return model_list

def plot_align_probs(df,output_dir):

    fig = plt.figure(figsize=(6.5,2.5))
    pbl = palette_by_model()
    for model,sub_df in df.groupby('Model'):
        data = np.stack(sub_df['align_probs'].tolist()).mean(axis=0,keepdims=False)
        plt.plot(data,label=model,c=pbl[model])
  
    sns.despine()
    plt.legend()
    plt.ylabel('Fraction aligned w/ protein')
    plt.xlabel('Amino acid')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/alignment_stats.svg')
    plt.close()

def rename(name):

    if name == 'CDS':
        return 'start (LFN)'
    elif name == 'EDC':
        return 'class (LFN)'
    elif name == 'EDC_CNN':
        return 'class (CNN)'
    elif name == 'bioseq2seq':
        return 'seq (LFN)'
    elif name == 'bioseq2seq_CNN':
        return 'seq (CNN)'
    elif name == 'bioseq2seq_CNN_lambd_0.05':
        return 'seq-wt (CNN)'
    elif name == 'bioseq2seq_lambd_0.1':
        return 'seq-wt (LFN)'
    elif name == 'seq2start_CNN':
        return 'start (CNN)' 
    elif name == 'rnasamba':
        return 'RNAsamba'
    else:
        return name

if __name__ == "__main__":

    args,unknown_args = parse_config()
    setup_fonts() 
    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    prefix = args.test_prefix.replace('test','test_RNA')
    output_dir = build_output_dir(args)
    bio_models = args.all_BIO_replicates
    EDC_models = args.all_EDC_replicates
    cnn_EDC_models = args.all_EDC_CNN_replicates
    EDC_eq_models = args.all_EDC_small_replicates
    cnn_models = args.all_CNN_replicates
    lfnet_weighted_models = args.all_LFNet_weighted_replicates
    cnn_weighted_models = args.all_CNN_weighted_replicates
    start_models = args.all_start_replicates
    start_cnn_models = args.all_start_CNN_replicates

    parent = 'experiments/output'
    all_models = get_model_names(lfnet_weighted_models) #+get_model_names(cnn_weighted_models)
    #all_models = get_model_names(bio_models)+get_model_names(cnn_models)
    #all_models += get_model_names(lfnet_weighted_models)+get_model_names(cnn_weighted_models)

    if 'parse' in unknown_args:
        mode = 'parse'
    elif 'align' in unknown_args:
        mode = 'align'
    else:
        raise ValueError("mode must be \'align\' or \'parse\'")
   
    storage = []
    for model in all_models:
        best_preds = os.path.join('experiments/output',os.path.join(model,f'{prefix}_full_preds.txt'))
        if mode == 'parse':
            print(model)
            results = parse(best_preds,test_file)
            storage.append(results) 
            print(results)
        if mode == 'align': 
            align(best_preds,test_file)

    if mode == 'parse':
        results_df = pd.DataFrame(storage)
        short_string = results_df['trial'].str.extract(r'([^\/]*)_(\d)_')
        results_df['Model'] = short_string[0].apply(rename)
        results_df['rep'] = short_string[1]
        results_df['pct_found'] =  100 *results_df['correct_CDS'] / results_df['support']
        f,(ax1,ax2) = plt.subplots(1,2,figsize=(6.5,2),sharey=True) 
        g1 = sns.stripplot(data=results_df,y='Model',x='pct_found',hue='Model',size=4,linewidth=1,palette=palette_by_model(),ax=ax1,legend=False)
        g1.set_xlabel('CDS prediction accuracy (%) ') 
        g2 = sns.stripplot(data=results_df,y='Model',x='align_id_mean',hue='Model',size=4,linewidth=1,palette=palette_by_model(),ax=ax2,legend=False)
        g2.set_xlabel('Mean alignment ID (%)')
        sns.despine() 
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ORF_stats.svg')
        plt.close()
        print(results_df)
        plot_align_probs(results_df,output_dir)
