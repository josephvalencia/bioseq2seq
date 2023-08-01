from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import os,sys
from parse_blast import parse_needle_results
import biotite.sequence.io.fasta as fasta

def to_fasta(df,name,column='RNA'):

    ids = df['ID'].tolist()
    sequences  = df[column].tolist()
    records = [make_record(name,seq) for name,seq in zip(ids,sequences)]
    SeqIO.write(records, name, "fasta")

def make_record(id,rna):
    return SeqRecord(Seq(rna),id=id,description='')

def parse_nonredundant_transcripts(filtered_fa):
    '''Ingest FASTA output from CD-HIT'''

    subset = []
    with open(filtered_fa) as inFile:
        for record in SeqIO.parse(inFile,'fasta'):
            subset.append(record.id)
    return subset

def needle_commands_from_blast(blast_file,test_name,train_name,parent):
    
    test_fasta = {x.id : str(x.seq) for x in SeqIO.parse(test_name,'fasta')} 
    reference_fasta = {x.id : str(x.seq) for x in SeqIO.parse(train_name,'fasta')} 
    colnames = ['query', 'subject', '% identity', 'alignment length', 'mismatches', 'gap opens', \
                'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
    
    df = pd.read_csv(blast_file, sep='\t', header=None,names=colnames)
    grouped = df.groupby(['query'])
    total = []
    
    for query, group_df in grouped:
        inner = group_df.groupby(['subject'])[['query','subject','evalue']].min()
        inner = inner.sort_values(['evalue'], ascending=True).iloc[0:5]['subject'].tolist()
        query_name = f'{parent}/{query}.fa'
        subj_name = f'{parent}/{query}_hits.fa'
        subject_file = fasta.FastaFile()
        query_file = fasta.FastaFile() 
        for subject in inner:
            subject_file[subject] = reference_fasta[subject]
        query_file[query] = test_fasta[query]
        subject_file.write(subj_name)
        query_file.write(query_name)
        cmd_format = f"needle -asequence {query_name} -bsequence {subj_name} -outfile {parent}/{query}.needle -gapopen 10 -gapextend 0.5 -brief"
        print(cmd_format) 

def parse_all_needle(reduced_rna_fa,train_csv,balance):
   
    # discarding BLAST+NEEDLE hits
    train_fasta = fasta.FastaFile.read(reduced_rna_fa)
    filtered = [] 
    for tscript in train_fasta.keys():
        coding = tscript.startswith('XM') or tscript.startswith('NM')
        lncPEP_score = parse_needle_results(tscript,'matches_lncPEP_blast')
        if lncPEP_score[-1] <= 80:
            filtered.append(tscript)
 
    # filter original csv 
    df = pd.read_csv(train_csv,sep='\t').set_index('ID')
    df_reduced = df.loc[filtered]
    if balance == 'true' or balance == 'True': 
        # rebalance by class
        groups = df_reduced.groupby('Type')
        print(groups.count()) 
        shorter = min([len(df) for group_name,df in groups])
        df_reconstructed = pd.concat([df.sample(n=shorter) for group_name,df in groups])
        df_reduced = df_reconstructed.sample(frac=1.0)
        print(df_reduced.groupby('Type').count())
   
    df_reduced = df_reduced.reset_index()
    # save files
    reduced_csv = train_csv.replace('.csv','_no_lncPEP.csv')
    reduced_rna_fa = train_csv.replace('.csv','_RNA_no_lncPEP.fa')
    reduced_protein_fa = train_csv.replace('.csv','_PROTEIN_no_lncPEP.fa')
    df_reduced.to_csv(reduced_csv,sep='\t',index=False)
    to_fasta(df_reduced,reduced_protein_fa,column='Protein')
    to_fasta(df_reduced,reduced_rna_fa,column='RNA')

def cdhit_and_blast(lncPEP_fa,train_fa):

    reduced_rna_fa = train_fa+'.reduced'
   
    cmd = f'cd-hit-est-2d -i {lncPEP_fa} -i2 {train_fa} -c 0.80 -n 5 -M 16000 -T 8 -o {reduced_rna_fa}'
    os.system(cmd) 

    cmd1=f'makeblastdb -in {lncPEP_fa} -dbtype nucl -out lncPEP_db'
    cmd2=f'blastn -query {reduced_rna_fa} -db lncPEP_db -out lncPEP_matches.blast -outfmt 6'
    os.system(cmd1)
    os.system(cmd2)
    
    if not os.path.isdir('matches_lncPEP_blast'):
        os.mkdir('matches_lncPEP_blast')
    needle_commands_from_blast('lncPEP_matches.blast',reduced_rna_fa,lncPEP_fa,'matches_lncPEP_blast')

if __name__ == "__main__":

    if sys.argv[1] == 'blast': 
        cdhit_and_blast(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == 'needle':
        parse_all_needle(sys.argv[2],sys.argv[3],sys.argv[4])
    else:
        raise ValueError("mode must be \'blast\' or \'needle\'")
