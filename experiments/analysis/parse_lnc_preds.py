import sys,re,os
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def needle_align_id(entry,save_dir):

    #cmd_format = "/home/bb/valejose/EMBOSS-6.6.0/emboss/needle -asequence {} -bsequence {} -gapopen {} -gapextend {} -sprotein -brief -stdout -auto"
    match_tscript = re.search('ID: (.*)',entry[0])
    is_noncoding = lambda x : x.startswith('NR') or x.startswith('XR')

    if match_tscript and is_noncoding(match_tscript.group(1)):
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

def parse_peptides(entry,save_dir,test_df):

    match_tscript = re.search('ID: (.*)',entry[0])
    is_noncoding = lambda x : x.startswith('NR') or x.startswith('XR')
   
    if match_tscript and is_noncoding(match_tscript.group(1)):
        tscript = match_tscript.group(1) 
        results = []
        for i,line in enumerate(entry[1:]):
            match_pred = re.search('PRED: <PC>(.*) SCORE: (.*)',line)
            if match_pred:
                results.append(make_record(f'{tscript}.peptide.{i}',match_pred.group(1)))
        
        peptide_file = f'{save_dir}/{tscript}_micropeptides.fa'
        with open(peptide_file,'w') as outfile:
            SeqIO.write(results,outfile,"fasta")
     
        # original seq
        seq = test_df.loc[tscript]['RNA']
        record = make_record(tscript,seq) 
        RNA_file = f'{save_dir}/{tscript}_RNA.fa'
        with open(RNA_file,'w') as outfile:
            SeqIO.write([record],outfile,"fasta")
       
        # 3-frame translation
        small_ORF_file = f'{save_dir}/{tscript}_ORF_translations.fa'
        cmd_args = ['getorf','-sequence',RNA_file,'-outseq',small_ORF_file,'-find', '1','-noreverse']
        subprocess.run(cmd_args)

        cmd_format = f"/home/bb/valejose/EMBOSS-6.6.0/emboss/needle -asequence {peptide_file} -bsequence {small_ORF_file} -outfile {save_dir}/{tscript}.needle -gapopen 10 -gapextend 0.5 -brief"
        print(cmd_format)

def make_record(id,rna):
    return SeqRecord(Seq(rna),id=id,description='')

def save_valid_sORFs(save_dir,valid):

    valid_list = []   
    for peptide in valid:
        tscript = peptide.split('.peptide')[0] 
        # save valid
        with open(f'{save_dir}/micropeptides/{tscript}_micropeptides.fa','r') as inFile:
            for record in SeqIO.parse(inFile,'fasta'):
                if record.id == peptide:
                    valid_list.append(record)

    # save valid
    with open(f'{save_dir}/validated_lncRNA_micropeptides.fa','w') as outFile:
        SeqIO.write(valid_list,outFile,'fasta')

def parse(filename,testfile,mode='getorf'):

    records = []
    parent = os.path.split(filename)[0]
    save_dir = os.path.join(parent,'micropeptides')
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir) 
   
    test_df = pd.read_csv(testfile,sep='\t').set_index('ID')
    with open(filename) as inFile:
        lines = inFile.readlines()
        storage = []
        for i in range(0,len(lines),6):
            entry = lines[i:i+6]
            if mode == 'getorf': 
                parse_peptides(entry,save_dir,test_df)
            elif mode == 'needle':
                result = needle_align_id(entry,save_dir)
                if result: 
                    entry = {'peptide' : result[0], 'match length': result[1], '% align id' : result[2]}
                    storage.append(entry)
            else:
                raise ValueError("mode must be getorf or needle")
        
        if mode == 'needle': 
            df = pd.DataFrame(storage)
            perfect_matches = df[df['% align id'] == 100.0]
            valid = perfect_matches['peptide'].tolist() 
            save_valid_sORFs(parent,valid)
            histname = os.path.join(parent,'micropeptide_hist.svg')
            f, (ax1,ax2) = plt.subplots(1,2,figsize=[9,4])
            print(f'Saving {histname} calculated from {len(df)} lncRNAs, of which {len(perfect_matches)} had a perfect sORF')
            sns.histplot(data=df,x='% align id',stat='count',bins=np.linspace(0,100,20),ax=ax1) 
            sns.histplot(data=perfect_matches,x='match length',stat='count',bins=np.linspace(0,100,20),ax=ax2) 
            plt.tight_layout()
            plt.savefig(histname)
            plt.close()

if __name__ == "__main__":

    testfile = "data/mammalian_200-1200_test_nonredundant_80.csv"
    parse(sys.argv[1],testfile,mode=sys.argv[2])
