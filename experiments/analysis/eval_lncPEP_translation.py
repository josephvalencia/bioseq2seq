import sys,re,os
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import getLongestORF, getFirstORF, parse_config,setup_fonts

def parse_needle_results(entry,save_dir):

    match_tscript = re.search('ID: (.*)',entry[0])
    is_noncoding = lambda x : x.startswith('NR') or x.startswith('XR')

    if match_tscript:
        tscript = match_tscript.group(1)
        needle_file = f'{save_dir}/{tscript}.needle'
        if os.path.exists(needle_file): 
            with open(needle_file,'r') as inFile:
                data = inFile.read()
            identity_pattern = "# Identity:(\s*)(\d*)\/(\d*) \((.*)%\)\n"
            name_pattern = "# 2: (.*)\n"
            id_matches = re.findall(identity_pattern,data)
            name_matches = re.findall(name_pattern,data)
            scores = [(y,int(x[-2]),float(x[-1])) for x,y in zip(id_matches,name_matches)]
            scores = sorted(scores,key = lambda x : x[1],reverse=True) 
            scores = sorted(scores,key = lambda x : x[2],reverse=True) 
            return scores[0]

def align_peptides(entry,save_dir,test_df):

    match_tscript = re.search('ID: (.*)',entry[0])
    is_noncoding = lambda x : x.startswith('NR') or x.startswith('XR')
   
    if match_tscript:
        tscript = match_tscript.group(1) 
        results = []
        for i,line in enumerate(entry):
            match_pred = re.search('PRED: <PC>(.*) SCORE: (.*)',line)
            if match_pred:
                peptide = make_record(f'{tscript}.peptide.{i}',match_pred.group(1))
                results.append(peptide) 
        
        peptide_file = f'{save_dir}/{tscript}.micropeptides.fa'
        with open(peptide_file,'w') as outfile:
            SeqIO.write(results,outfile,"fasta")
     
        # original seq
        if tscript in test_df.index: 
            seq = test_df.loc[tscript]['Protein']
            record = make_record(tscript,seq) 
            true_file = f'{save_dir}/{tscript}_true_PROTEIN.fa'
            with open(true_file,'w') as outfile:
                SeqIO.write([record],outfile,"fasta")
            
            cmd_format = f"needle -asequence {true_file} -bsequence {peptide_file} -outfile {save_dir}/{tscript}.needle -gapopen 10 -gapextend 0.5 -brief"
            print(cmd_format)

def make_record(id,rna):
    return SeqRecord(Seq(rna),id=id,description='')

def parse(filename,testfile,mode='getorf'):

    records = []
    parent = os.path.split(filename)[0]
    save_dir = os.path.join(parent,'micropeptides')
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
                align_peptides(entry,save_dir,test_df)
            elif mode == 'parse':
                result = parse_needle_results(entry,save_dir)
                if result: 
                    entry = {'peptide' : result[0], 'Peptide length': result[1], id_label : result[2]}
                    storage.append(entry)
            else:
                raise ValueError("mode must be \'align\' or \'parse\'")
       
    if mode == 'parse':
        df = pd.DataFrame(storage)
        
        longest = []
        for x in test_df['RNA'].tolist():
            s,e = getLongestORF(x)
            translated = Seq(x[s:e]).translate()
            longest.append(str(translated)[:-1])
        
        test_df['longest_ORF_translation'] = longest 
        test_df['peptide_is_longest'] = test_df['Protein'] == test_df['longest_ORF_translation'] 
        test_df = test_df.reset_index()
        test_df = test_df[['ID','Protein','peptide_is_longest']]
        df['ID'] = [x.split('.peptide')[0] for x in df['peptide']] 
        combined = test_df.merge(df,on='ID')
        print(combined)

if __name__ == "__main__":

    args,unknown_args = parse_config()
    setup_fonts() 
    test_file = os.path.join(args.data_dir,'lnc_PEP.csv')
    best_preds = os.path.join('translation',f'lnc_PEP_RNAs_preds_with_translations.txt')
    
    if 'parse' in unknown_args:
        mode = 'parse'
    elif 'align' in unknown_args:
        mode = 'align'
    else:
        raise ValueError("mode must be \'align\' or \'parse\'")

    parse(best_preds,test_file,mode=mode)
