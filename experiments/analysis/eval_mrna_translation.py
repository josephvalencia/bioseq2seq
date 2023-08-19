import sys,re,os
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import getLongestORF,parse_config, setup_fonts
from tqdm import tqdm

def parse_needle_results(entry,save_dir):

    match_tscript = re.search('ID: (.*) coding_prob : ([0|1]\.\d*)',entry[0])
    #match_tscript = re.search('ID: (.*)',entry[0])
    is_coding = lambda x : x.startswith('NM') or x.startswith('XM')
    
    max_results = None
    true_results = None
    tscript = None

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

    return tscript, max_results,true_results

def max_identity(data):

    identity_pattern = "# Identity:(\s*)(\d*)\/(\d*) \((.*)%\)\n"
    name_pattern = "# 1: (.*)\n"
    id_matches = re.findall(identity_pattern,data)
    name_matches = re.findall(name_pattern,data)
    scores = [(y,int(x[-2]),float(x[-1])) for x,y in zip(id_matches,name_matches)]
    scores = sorted(scores,key = lambda x : x[1],reverse=True) 
    scores = sorted(scores,key = lambda x : x[2],reverse=True)
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
            seq = test_df.loc[tscript]['Protein']
            record = make_record(tscript,seq) 
            true_file = f'{save_dir}/{tscript}_true_PROTEIN.fa'
            with open(true_file,'w') as outfile:
                SeqIO.write([record],outfile,"fasta")
            
            # original RNA seq
            rna = test_df.loc[tscript]['RNA']
            record = make_record(tscript,rna) 
            RNA_file = f'{save_dir}/{tscript}_RNA.fa'
            with open(RNA_file,'w') as outfile:
                SeqIO.write([record],outfile,"fasta")
       
            # 3-frame translation
            small_ORF_file = f'{save_dir}/{tscript}_ORF_translations.fa'
            cmd_args = ['getorf','-sequence',RNA_file,'-outseq',small_ORF_file,'-find', '1','-noreverse']
            subprocess.run(cmd_args)

            orf_cmd = f"needle -asequence {predicted_file} -bsequence {small_ORF_file} -outfile {save_dir}/{tscript}_ALL.needle -gapopen 10 -gapextend 0.5 -brief"
            print(orf_cmd)
            prot_cmd = f"needle -asequence {predicted_file} -bsequence {true_file} -outfile {save_dir}/{tscript}_TRUE.needle -gapopen 10 -gapextend 0.5 -brief"
            print(prot_cmd)

def make_record(id,seq):
    return SeqRecord(Seq(seq),id=id,description='')

def parse(filename,testfile,mode='parse'):

    records = []
    parent = os.path.split(filename)[0]
    save_dir = os.path.join(parent,'translations')
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir) 
   
    is_coding = lambda x : x.startswith('NM') or x.startswith('XM')
    test_df = pd.read_csv(testfile,sep='\t').set_index('ID')
    id_label =  'align_id'
    with open(filename) as inFile:
        lines = inFile.readlines()
        storage = []
        for i in tqdm(range(0,len(lines),6)):
            entry = lines[i:i+6]
            if mode == 'align': 
                align_proteins(entry,save_dir,test_df)
            elif mode == 'parse':
                tscript,result,true_result = parse_needle_results(entry,save_dir)
                # true positive mRNA 
                if result and true_result: 
                    correct_ORF = true_result[-1] >= result[-1]
                    row = {'ID' : tscript, 'peptide' : result[0], 'Protein length': true_result[1], id_label : true_result[2], 'correct_ORF' : correct_ORF}
                    storage.append(row)
                # false negative mRNA 
                elif is_coding(tscript):
                    row = {'ID' : tscript, 'peptide' : None, 'Protein length': 0, id_label : 0, 'correct_ORF' : False}
                    storage.append(row)
            else:
                raise ValueError("mode must be align or parse")
        
        if mode == 'parse': 
            df = pd.DataFrame(storage)
            homology = pd.read_csv("test_maximal_homology.csv")
            df = df.merge(homology,on='ID') 
            reduced = df['score'] <=80
            df = df.loc[reduced]
            
            correct_orf_count = sum([1 for x in df['correct_ORF'] if x])
            mean = df['align_id'].mean()
            std = df['align_id'].std()
            perfect_matches = df[df[id_label] == 100.0]
            valid = perfect_matches['peptide'].tolist() 
            histname = os.path.join(parent,'mRNA_translation_hist.svg')
            f, (ax1,ax2) = plt.subplots(1,2,figsize=[7.5,2],gridspec_kw={'width_ratios':[1,2]})
            print(f'Saving {histname}')
            print(f'calculated from {len(df)} mRNAs, of which {correct_orf_count} ({100*correct_orf_count/len(df):.1f}%) predict the true CDS, {len(perfect_matches)} ({100 *len(perfect_matches) / len(df):.1f}%) had a perfect protein decoding, % align id (mean={mean:.2f},std={std:.2f})')
            sns.set_style(style='white',rc={'font.family' : ['Helvetica']})
            sns.histplot(data=df,x=id_label,stat='count',bins=np.linspace(0,100,21),ax=ax1,color='red',alpha=0.7) 
            ax1.set_xlabel('Pred. align. w/ true protein (% ID)')
            sns.despine() 
            sns.histplot(data=perfect_matches,x='Protein length',stat='count',bins=np.linspace(0,400,21),ax=ax2,color='red',alpha=0.7,label='exact\npreds') 
            test_df['Protein length'] = [len(x) for x in test_df['Protein'].tolist()]
            coding = test_df[test_df['Type'] == '<PC>']
            sns.histplot(data=coding,x='Protein length', stat='count',element='step',bins=np.linspace(0,400,21),fill=False,ax=ax2,color='black',linestyle='--',label='all') 
            ax2.set_xlabel('Protein length (aa)')
            sns.despine() 
            plt.tight_layout()
            f.legend(loc="upper right", bbox_to_anchor=(0.58, 1.0),borderpad=0.25)
            plt.savefig(histname)
            plt.close()

if __name__ == "__main__":

    args,unknown_args = parse_config()
    setup_fonts() 
    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    prefix = args.test_prefix.replace('test','test_RNA')
    best_preds = os.path.join(args.best_BIO_DIR,f'{prefix}_full_preds.txt')
    if 'parse' in unknown_args:
        mode = 'parse'
    elif 'align' in unknown_args:
        mode = 'align'
    else:
        raise ValueError("mode must be \'align\' or \'parse\'")
    
    parse(best_preds,test_file,mode=mode)
