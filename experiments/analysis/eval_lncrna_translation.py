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

    #match_tscript = re.search('ID: (.*)',entry[0])
    match_tscript = re.search('ID: (.*) coding_prob : ([0|1]\.\d*)',entry[0])
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

def align_peptides(entry,save_dir,test_df):

    #match_tscript = re.search('ID: (.*)',entry[0])
    match_tscript = re.search('ID: (.*) coding_prob : ([0|1]\.\d*)',entry[0])
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

        cmd_format = f"needle -asequence {peptide_file} -bsequence {small_ORF_file} -outfile {save_dir}/{tscript}.needle -gapopen 10 -gapextend 0.5 -brief"
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
        perfect_matches = df[df[id_label] == 100.0]
        valid = perfect_matches['peptide'].tolist() 
        
        peptide_lens = []
        alt_peptide_lens = []
        for x in test_df['RNA'].tolist():
            s,e = getLongestORF(x)
            peptide_lens.append((e-s) / 3)
            s_alt,e_alt = getFirstORF(x)
            alt_peptide_lens.append((e_alt-s_alt) / 3)

        test_df['Peptide length'] = peptide_lens
        test_df['First peptide length'] = alt_peptide_lens
        
        noncoding = test_df[test_df['Type'] == '<NC>']
        max_len = noncoding['Peptide length'].max() 
        histname = os.path.join(parent,'micropeptide_hist.svg')
        f, (ax1,ax2) = plt.subplots(1,2,figsize=[7.5,2],gridspec_kw={'width_ratios':[1,2]})
        sns.set_style(style='white',rc={'font.family' : ['Helvetica']})
        print(f'Saving {histname} calculated from {len(df)} lncRNAs, of which {len(perfect_matches)} = {100 *len(perfect_matches) / len(df):.1f}% had a perfect sORF')
        sns.histplot(data=df,x=id_label,stat='count',bins=np.linspace(0,100,21),ax=ax1,alpha=0.7) 
        ax1.set_xlabel('Best pred. align. w/ ORF translation (% ID)')
        sns.despine() 
        sns.histplot(data=perfect_matches,
                    x='Peptide length',
                    stat='count',
                    bins=np.linspace(0,220,23),
                    ax=ax2,label='exact preds') 
        sns.histplot(data=noncoding,
                    x='Peptide length',
                    stat='count',
                    element='step',
                    fill=False,
                    bins=np.linspace(0,220,23),
                    ax=ax2,
                    color='black',
                    linestyle='--',label='all (longest ORF)') 
        sns.histplot(data=noncoding,
                    x='First peptide length',
                    stat='count',
                    element='step',
                    fill=False,
                    bins=np.linspace(0,220,23),
                    ax=ax2,
                    color='tab:orange',linewidth=2,
                    linestyle='--',label='all (first ORF)',legend=True) 
        ax2.set_xlabel('ORF length (codons)')
        sns.despine() 
        plt.tight_layout()
        f.legend(loc="upper right", bbox_to_anchor=(0.95, 1.0),borderpad=0.25,prop={"family":"Helvetica"})
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
