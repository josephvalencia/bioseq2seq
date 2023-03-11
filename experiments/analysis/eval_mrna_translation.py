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

def parse_needle_results(entry,save_dir):

    match_tscript = re.search('ID: (.*)',entry[0])

    is_coding = lambda x : x.startswith('NM') or x.startswith('XM')
    if match_tscript and is_coding(match_tscript.group(1)):
        tscript = match_tscript.group(1)
        needle_file = f'{save_dir}/{tscript}.needle'
        if os.path.exists(needle_file): 
            with open(needle_file,'r') as inFile:
                data = inFile.read()
            identity_pattern = "# Identity:(\s*)(\d*)\/(\d*) \((.*)%\)\n"
            name_pattern = "# 1: (.*)\n"
            id_matches = re.findall(identity_pattern,data)
            name_matches = re.findall(name_pattern,data)
            scores = [(y,int(x[-2]),float(x[-1])) for x,y in zip(id_matches,name_matches)]
            scores = sorted(scores,key = lambda x : x[1],reverse=True) 
            scores = sorted(scores,key = lambda x : x[2],reverse=True) 
            return scores[0]

def align_proteins(entry,save_dir,test_df):

    match_tscript = re.search('ID: (.*)',entry[0])
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
            
            # original seq
            seq = test_df.loc[tscript]['Protein']
            record = make_record(tscript,seq) 
            true_file = f'{save_dir}/{tscript}_true_PROTEIN.fa'
            with open(true_file,'w') as outfile:
                SeqIO.write([record],outfile,"fasta")
           
            cmd_format = f"needle -asequence {predicted_file} -bsequence {true_file} -outfile {save_dir}/{tscript}.needle -gapopen 10 -gapextend 0.5 -brief"
            print(cmd_format)

def make_record(id,seq):
    return SeqRecord(Seq(seq),id=id,description='')

def parse(filename,testfile,mode='parse'):

    records = []
    parent = os.path.split(filename)[0]
    save_dir = os.path.join(parent,'translations')
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir) 
   
    test_df = pd.read_csv(testfile,sep='\t').set_index('ID')
    id_label =  'align_id'
    with open(filename) as inFile:
        lines = inFile.readlines()
        storage = []
        for i in range(0,len(lines),6):
            entry = lines[i:i+6]
            if mode == 'align': 
                align_proteins(entry,save_dir,test_df)
            elif mode == 'parse':
                result = parse_needle_results(entry,save_dir)
                if result: 
                    entry = {'peptide' : result[0], 'Protein length': result[1], id_label : result[2]}
                    storage.append(entry)
            else:
                raise ValueError("mode must be align or parse")
        
        if mode == 'parse': 
            df = pd.DataFrame(storage)
            perfect_matches = df[df[id_label] == 100.0]
            valid = perfect_matches['peptide'].tolist() 
            histname = os.path.join(parent,'mRNA_translation_hist.svg')
            f, (ax1,ax2) = plt.subplots(1,2,figsize=[7.5,2],gridspec_kw={'width_ratios':[1,2]})
            print(f'Saving {histname} calculated from {len(df)} mRNAs, of which {len(perfect_matches)} = {100 *len(perfect_matches) / len(df):.1f} % had a perfect protein decoding')
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
