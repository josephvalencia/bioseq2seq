import pandas as pd
import re,sys
import os
import biotite.sequence.io.fasta as fasta
from utils import parse_config,setup_fonts
import seaborn as sns
import matplotlib.pyplot as plt

def needle_commands_from_blast(blast_file,test_name,train_name,parent):
    
    test_fasta = fasta.FastaFile.read(test_name)
    train_fasta = fasta.FastaFile.read(train_name)

    colnames = ['query', 'subject', '% identity', 'alignment length', 'mismatches', 'gap opens', \
                'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
    
    df = pd.read_csv(blast_file, sep='\t', header=None,names=colnames)
    grouped = df.groupby(['query'])
    total = []
    
    for query, group_df in grouped:
        inner = group_df.groupby(['subject'])[['query','subject','evalue']].min()
        inner = inner.sort_values(['evalue'], ascending=True).iloc[0:5]['subject'].tolist()
        print(inner) 
        query_name = f'{parent}/{query}.fa'
        subj_name = f'{parent}/{query}_hits.fa'
        subject_file = fasta.FastaFile()
        query_file = fasta.FastaFile() 
        for subject in inner:
            subject_file[subject] = train_fasta[subject]
        query_file[query] = test_fasta[query]
        subject_file.write(subj_name)
        query_file.write(query_name)
        cmd_format = f"needle -asequence {query_name} -bsequence {subj_name} -outfile {parent}/{query}.needle -gapopen 10 -gapextend 0.5 -brief"
        print(cmd_format) 

def parse_needle_results(tscript,save_dir):
    
    needle_file = f'{save_dir}/{tscript}.needle'
    if os.path.exists(needle_file): 
        with open(needle_file,'r') as inFile:
            data = inFile.read()
        identity_pattern = "# Identity:(\s*)(\d*)\/(\d*) \((.*)%\)\n"
        name_pattern = "# 2: (.*)\n"
        id_matches = re.findall(identity_pattern,data)

        name_matches = re.findall(name_pattern,data)
        scores = [(y,int(x[-2]),float(x[-1])) for x,y in zip(id_matches,name_matches)]
        sep = "_____________________________________________________________" 
        scores = sorted(scores,key = lambda x : x[1],reverse=True) 
        scores = sorted(scores,key = lambda x : x[2],reverse=True) 
        return scores[0]
    else:
        return (None,None,0.0)

def parse_all_needle(test_name):
    test_fasta = fasta.FastaFile.read(test_name)
    
    results = [] 
    for tscript in test_fasta.keys():
        coding = tscript.startswith('XM') or tscript.startswith('NM')
        train_score = parse_needle_results(tscript,'matches_train_blast')
        val_score = parse_needle_results(tscript,'matches_val_blast') 
        entry = {'ID':tscript,'coding':coding,
                 'train_match' : train_score[0],
                 'train_score':train_score[-1],
                 'val_match' : val_score[0],
                 'val_score':val_score[-1]}
        results.append(entry)
    
    df = pd.DataFrame(results)
    df['score'] = df[['train_score','val_score']].max(axis=1)
    train_removed = df[df['train_score'] <= 80]
    nonredundant = df[df['score'] <= 80]
    
    print('Full',df.groupby(['coding']).count())
    print('Test removed',train_removed.groupby(['coding']).count())
    print('Both removed',nonredundant.groupby(['coding']).count())
    df.to_csv("test_maximal_homology.csv",index=False) 
    num_bad = len(df[df['score'] > 80]) 
    print(f'Number of bad alignments: {num_bad}/{len(df)} = {num_bad/len(df):.3f}')
    #sns.histplot(data=df,x='score',hue='coding')
    plt.figure(figsize=(3.5,3))
    sns.histplot(data=df,x='score',bins=20)
    plt.xlabel('Max %id with train set')
    plt.title('CD-HIT') 
    plt.tight_layout()
    plt.savefig('full_needle_align_results.png')
    #plt.savefig('reduced_needle_align_results.png')

def run_blast(train_fasta,test_fasta,val_fasta):
    
    cmd1=f'makeblastdb -in {train_fasta} -dbtype nucl -out train_balanced_db'
    cmd2=f'makeblastdb -in {val_fasta} -dbtype nucl -out val_db'
    cmd3=f'blastn -query {test_fasta} -db train_balanced_db -out train_matches.blast -outfmt 6'
    cmd4=f'blastn -query {test_fasta} -db val_db -out val_matches.blast -outfmt 6'
    os.system(cmd1) 
    os.system(cmd2) 
    os.system(cmd3) 
    os.system(cmd4) 

def run_and_parse_blast(train_fasta,test_fasta,val_fasta):

    run_blast(train_fasta,test_fasta,val_fasta)
    
    if not os.path.isdir('matches_train_blast'):
        os.mkdir('matches_train_blast')
    needle_commands_from_blast('train_matches.blast',test_fasta,train_fasta,'matches_train_blast')
    
    if not os.path.isdir('matches_val_blast'):
        os.mkdir('matches_val_blast')
    needle_commands_from_blast('val_matches.blast',test_fasta,val_fasta,'matches_val_blast')

if __name__ == "__main__":

    if sys.argv[1] == 'blast': 
        run_and_parse_blast(*sys.argv[2:])
    elif sys.argv[1] == 'needle':
        parse_all_needle(sys.argv[2])
    else:
        raise ValueError("mode must be \'blast\' or \'needle\'")

