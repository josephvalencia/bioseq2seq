import re,os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from utils import getLongestORF, parse_config,setup_fonts

def parse_needle_results(entry,save_dir):

    match_tscript = re.search('ID: (.*)',entry[0])
    match_pred = re.search('PRED: (<PC>|<NC>)',entry[1])
    match_score = re.search('PRED: <PC>(.*) SCORE: (.*)',' '.join(entry))
    is_noncoding = lambda x : x.startswith('NR') or x.startswith('XR')
    coding_prob = float(match_score.group(2)) if match_score is not None else 0.0
    coding_prob = round(coding_prob *100,2)

    if match_tscript and match_pred:
        tscript = match_tscript.group(1)
        pred_coding = match_pred.group(1) == '<PC>'
        needle_file = f'{save_dir}/{tscript}.needle'
        if os.path.exists(needle_file): 
            with open(needle_file,'r') as inFile:
                data = inFile.read()
            identity_pattern = "# Identity:(\s*)(\d*)\/(\d*) \((.*)%\)\n"
            name_pattern = "# 2: (.*)\n"
            id_matches = re.findall(identity_pattern,data)
            name_matches = re.findall(name_pattern,data)
            scores = [(y,int(x[-2]),float(x[-1])) for x,y in zip(id_matches,name_matches)]
            first_found = None 
            for x in sorted(scores,key = lambda x : x[0]):
                print(x) 
                if x[2] >= 90:
                    first_found = x[0] 
                    break
            scores = sorted(scores,key = lambda x : x[1],reverse=True) 
            scores = sorted(scores,key = lambda x : x[2],reverse=True) 
            return pred_coding,first_found,scores[0]
    else:
        return None

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
    test_df['true_peptide_len'] = [len(x) for x in test_df['Protein'].tolist()]
    with open(filename) as inFile:
        lines = inFile.readlines()
        storage = []
        for i in range(0,len(lines),6):
            entry = lines[i:i+6]
            if mode == 'align': 
                align_peptides(entry,save_dir,test_df)
            elif mode == 'parse':
                parsed = parse_needle_results(entry,save_dir)
                if parsed:
                    coding_pred,first_found,result = parsed
                    entry = {'peptide' : result[0], 'Peptide length': result[1],
                            'best_align_id' : result[2],'bioseq2seq_pred' : coding_pred,'bioseq2seq_first' :first_found}
                    storage.append(entry)
            else:
                raise ValueError("mode must be \'align\' or \'parse\'")
       
    if mode == 'parse':
        
        # check overlap of lncPEP with training data according to ID 
        id_list = []
        with open('validated_lncPEP_ids.txt','r') as outFile:
            id_list = [x.rstrip().split('.')[0] for x in outFile.readlines()]
        df = pd.read_csv('data/mammalian_200-1200_train_balanced.csv',sep="\t")
        overlap = df.loc[df['ID'].str.startswith(tuple(id_list))]
        id_match = set(overlap['ID'].tolist())
       
        # check overlap of lncPEP with training data according to CD-hit
        cdhit_nonredundant = []
        for record in SeqIO.parse('possible_hits','fasta'):
            cdhit_nonredundant.append(record.id)
        
        samba_results = pd.read_csv('rnasamba_new.tsv',sep='\t')
        samba_results['ID'] = [x.split(' ')[0] for x in samba_results['sequence_name']]
        samba_results = samba_results.set_index('ID') 
        df = pd.DataFrame(storage)
        longest = []
        lens = [] 
        samba = [] 
        for tscript,x in zip(test_df.index.tolist(),test_df['RNA'].tolist()):
            s,e = getLongestORF(x)
            translated = Seq(x[s:e]).translate()
            longest.append(str(translated)[:-1])
            lens.append(len(x))
            pred = samba_results.loc[tscript,'classification']
            mark = pred == 'coding'
            samba.append(mark) 
       
        test_df['rnasamba_pred'] = samba
        test_df['longest_ORF_translation'] = longest 
        test_df['tscript_len'] = lens 
        test_df['peptide_is_longest'] = test_df['Protein'] == test_df['longest_ORF_translation']
        
        test_df = test_df.reset_index()
        print(test_df[['ID','Protein','longest_ORF_translation','peptide_is_longest']])
        test_df = test_df[['ID','Protein','peptide_is_longest','true_peptide_len','tscript_len','rnasamba_pred']]
        df['ID'] = [x.split('.peptide')[0] for x in df['peptide']] 
        df['best_beam'] = [x.split('.peptide.')[1] for x in df['peptide']] 
        df['first_beam'] = [x.split('.peptide.')[1] if x is not None else '9999' for x in df['bioseq2seq_first']] 
        # add  needle alignment results 
        combined = test_df.merge(df,on='ID')
        
        in_training_set = (combined['ID'].isin(id_match)) | (~combined['ID'].isin(cdhit_nonredundant)) 
        unique = combined.loc[~in_training_set]
        similar = combined.loc[in_training_set]
        print('in training set') 
        print_latex_table(similar)
        print('not in training set') 
        print_latex_table(unique)

def print_latex_table(df): 

    '''
    df = df[['ID','tscript_len','true_peptide_len','peptide_is_longest','rnasamba_pred',
        'bioseq2seq_pred','first_beam','best_beam','best_align_id']] 
    df = df.sort_values(by=['best_beam','true_peptide_len'],ascending=[True,False])
    '''

    df = df[['ID','tscript_len','true_peptide_len','peptide_is_longest','rnasamba_pred','bioseq2seq_pred','first_beam']] 
    df = df.sort_values(by=['first_beam','true_peptide_len'],ascending=[True,False])
    table = df.to_latex(index=False,column_format='l'*len(df.columns))
    table = re.sub("&(\s*)","& ",table)
    table = table.replace('True',r'\cmark')
    table = table.replace('False',r'\xmark')
    table = table.replace('9999',r'\xmark')
    print(table)

if __name__ == "__main__":

    args,unknown_args = parse_config()
    setup_fonts() 
    test_file = os.path.join(args.data_dir,'lnc_PEP.csv')
    best_preds = os.path.join('translation',f'lnc_PEP_RNA_preds.txt')
    
    if 'parse' in unknown_args:
        mode = 'parse'
    elif 'align' in unknown_args:
        mode = 'align'
    else:
        raise ValueError("mode must be \'align\' or \'parse\'")

    parse(best_preds,test_file,mode=mode)
