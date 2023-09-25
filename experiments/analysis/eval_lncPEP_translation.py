import re,os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
from utils import getLongestORF, parse_config,setup_fonts
import subprocess 

def parse_needle_results(entry,save_dir):

    #match_tscript = re.search('ID: (.*)',entry[0])
    match_tscript = re.search('ID: (.*) coding_prob : ([0|1]\.\d*)',entry[0])
    match_pred = re.search('PRED: (<PC>|<NC>)',entry[1])
    match_score = re.search('PRED: <PC>(.*) SCORE: (.*)',' '.join(entry))
    is_noncoding = lambda x : x.startswith('NR') or x.startswith('XR')
    coding_prob = float(match_score.group(2)) if match_score is not None else 0.0
    coding_prob = round(coding_prob *100,2)

    if match_tscript and match_pred:
        tscript = match_tscript.group(1)
        pred_coding = match_pred.group(1) == '<PC>'
        
        # alignment of all beams against 3-frame translation
        needle_file = f'{save_dir}/{tscript}_ALL.needle'
        if os.path.exists(needle_file): 
            with open(needle_file,'r') as inFile:
                data = inFile.read()
            max_results,top_beam_max_result = max_identity(data)
        
        needle_file = f'{save_dir}/{tscript}.needle'
        if os.path.exists(needle_file): 
            with open(needle_file,'r') as inFile:
                data = inFile.read()
            
            # all beams aligned with ground truth 
            identity_pattern = "# Identity:(\s*)(\d*)\/(\d*) \((.*)%\)\n"
            name_pattern = "# 2: (.*)\n"
            id_matches = re.findall(identity_pattern,data)
            name_matches = re.findall(name_pattern,data)
            scores = [(y,int(x[-2]),float(x[-1])) for x,y in zip(id_matches,name_matches)]
            
            # whether the top beam has higher identity with ground truth than all the other ORFs 
            top_beam_with_ground_truth = scores[0] 
            correct_ORF = False
            if top_beam_max_result is not None: 
                correct_ORF = top_beam_with_ground_truth[-1] >= top_beam_max_result[-1]

            # identify the first ORF exceeding 90% alignment with ground truth
            first_found = None 
            for x in sorted(scores,key = lambda x : x[0]):
                if x[2] >= 90:
                    first_found = x[0] 
                    break
            scores = sorted(scores,key = lambda x : x[1],reverse=True) 
            scores = sorted(scores,key = lambda x : x[2],reverse=True) 
            msa_list = list(AlignIO.parse(needle_file, "emboss"))
            alignment = msa_list[0]
            return pred_coding,alignment,first_found,scores[0],correct_ORF
    else:
        return None

def max_identity(data):
    '''From alignment of all beams against 3-fram translation, return top scoring detection overall as well as the top scoring detection from ONLY the top beam ''' 
    
    identity_pattern = "# Identity:(\s*)(\d*)\/(\d*) \((.*)%\)\n"
    name_pattern = "# 1: (.*)\n"
    id_matches = re.findall(identity_pattern,data)
    name_matches = re.findall(name_pattern,data)
    scores = [(y,int(x[-2]),float(x[-1])) for x,y in zip(id_matches,name_matches)]
    best_scores = [x for x in scores if x[0].endswith('peptide.1')] 
    
    def len_then_id(score_list): 
        score_list = sorted(score_list,key = lambda x : x[1],reverse=True) 
        score_list = sorted(score_list,key = lambda x : x[2],reverse=True)
        return score_list
    
    scores = len_then_id(scores)
    best = None 
    if len(best_scores) > 1:
        best = len_then_id(best_scores)[0]
    
    return scores[0],best

def align_peptides(entry,save_dir,test_df):

    #match_tscript = re.search('ID: (.*)',entry[0])
    match_tscript = re.search('ID: (.*) coding_prob : ([0|1]\.\d*)',entry[0])
    is_noncoding = lambda x : x.startswith('NR') or x.startswith('XR')
    
    if match_tscript:
        tscript = match_tscript.group(1) 
        results = []
        
        ''' 
        match_pred = re.search('PRED: <PC>(.*) SCORE: (.*)',entry[1])
        if match_pred:
            peptide = make_record(f'{tscript}.peptide.{0}',match_pred.group(1))
        '''

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

            orf_cmd = f"needle -asequence {peptide_file} -bsequence {small_ORF_file} -outfile {save_dir}/{tscript}_ALL.needle -gapopen 10 -gapextend 0.5 -brief"
            print(orf_cmd)
            cmd_format = f"needle -asequence {true_file} -bsequence {peptide_file} -outfile {save_dir}/{tscript}.needle -gapopen 10 -gapextend 0.5 -brief"
            print(cmd_format)

def make_record(id,rna):
    return SeqRecord(Seq(rna),id=id,description='')

def parse(filename,testfile,model_name,mode='getorf'):

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
                    coding_pred,alignment,first_found,result,correct_ORF = parsed
                    #print(coding_pred,result[2],alignment)
                    entry = {'peptide' : result[0], 'Peptide length': result[1],
                            'best_align_id' : result[2],'pred' : coding_pred,'bioseq2seq_first' :first_found, 'correct_ORF' : correct_ORF}
                    storage.append(entry)
            else:
                raise ValueError("mode must be \'align\' or \'parse\'")
       
    if mode == 'parse':
        print(filename)     
        # check overlap of lncPEP with training data according to ID 
        id_list = []
        with open('validated_lncPEP_ids.txt','r') as outFile:
            id_list = [x.rstrip().split('.')[0] for x in outFile.readlines()]
        df = pd.read_csv('data/mammalian_200-1200_train_balanced.csv',sep="\t")
        overlap = df.loc[df['ID'].str.startswith(tuple(id_list))]
        id_match = set(overlap['ID'].tolist())
        test_df = attach_external(test_df)        
        
        df = pd.DataFrame(storage)
        df['ID'] = [x.split('.peptide')[0] for x in df['peptide']] 
        df['best_beam'] = [x.split('.peptide.')[1] for x in df['peptide']] 
        df['first_beam'] = [x.split('.peptide.')[1] if x is not None else '9999' for x in df['bioseq2seq_first']] 
        # add needle alignment results 
        combined = test_df.merge(df,on='ID')
        if model_name != 'RNAsamba_lncPEP': 
            print(combined[['ID','peptide_is_longest','best_beam','first_beam','pred','correct_ORF']]) 
        return summary(combined,model_name)

def parse_start(filename,testfile,model_name):

    test_df = pd.read_csv(testfile,sep='\t').set_index('ID')
    test_df['true_peptide_len'] = [len(x) for x in test_df['Protein'].tolist()]
    df = pd.read_csv(filename,sep='\t')
    
    df = df.rename(columns={'tscript' : 'ID'})
    
    test_df = attach_external(test_df)
    combined = test_df.merge(df,on='ID') 
    print('START',combined)
    peps = []
    for tscript,rna,prot,s in zip(combined['ID'],combined['RNA'],combined['Protein'],combined['start']):
        translated = str(Seq(rna[s:]).translate())
        implied_peptide = translated.split('*')[0]
        peps.append(implied_peptide)
    
    combined['implied_peptides'] = peps
    combined['pred'] = combined['start'] != (combined['tscript_len'] -1) 
    combined['correct_ORF'] = combined['Protein'] == combined['implied_peptides']
    return summary(combined,model_name)

def attach_external(test_df):

    longest = []
    lens = [] 
    for tscript,x in zip(test_df.index.tolist(),test_df['RNA'].tolist()):
        s,e = getLongestORF(x)
        translated = Seq(x[s:e]).translate()
        longest.append(str(translated)[:-1])
        lens.append(len(x))
    
    test_df['longest_ORF_translation'] = longest 
    test_df['tscript_len'] = lens 
    test_df = test_df.reset_index() 
    test_df['peptide_is_longest'] = test_df['Protein'] == test_df['longest_ORF_translation']
    test_df = test_df[['ID','Protein','peptide_is_longest','true_peptide_len','tscript_len','RNA']]
    return test_df

def eval_samba(rnasamba_output,testfile,model_name):

    test_df = pd.read_csv(testfile,sep='\t')#.set_index('ID')
    test_df['true_peptide_len'] = [len(x) for x in test_df['Protein'].tolist()]
    samba_results = pd.read_csv(rnasamba_output,sep='\t')
    samba_results['ID'] = [x.split(' ')[0] for x in samba_results['sequence_name']]
    test_df = attach_external(test_df)
    combined = test_df.merge(samba_results,on='ID')
    combined['pred'] = [x == 'coding' for x in samba_results['classification']] 
    return summary(combined,model_name)

def summary(df,model_name):

    is_longest = df['peptide_is_longest'] 
    
    pos_count = len(df[df['pred']])
    pos_count_not_longest = len(df[(df['pred'] ) & (~is_longest)])
    pos_count_longest = len(df[(df['pred'] ) & (is_longest)])
   

    orf_count = len(df[df['correct_ORF']]) if 'correct_ORF' in df.columns else 0 
    orf_count_not_longest = len(df[(df['correct_ORF'] ) & (~is_longest)]) if 'correct_ORF' in df.columns else 0  
    orf_count_longest = len(df[(df['correct_ORF'] ) & (is_longest)]) if 'correct_ORF' in df.columns else 0 
    
    N = len(df) 
    n_not_longest = len(df[~is_longest])
    n_longest = len(df[is_longest])
    
    #return {'model' : model_name, f'Pred. coding (n={len(df)})' : f'{pos_count} ({100*pos_count/len(df):.1f}%)',
    return {'Model' : model_name, f'Pred. coding (n={n_longest})' : f'{pos_count_longest} ({100*pos_count_longest/n_longest:.1f}%)',
            f'Pred. ORF (n={n_longest})' : f'{orf_count_longest} ({100*orf_count_longest/n_longest:.1f}%)',
            f'Pred. coding (not longest ORF) (n={n_not_longest})' : f'{pos_count_not_longest} ({100*pos_count_not_longest/n_not_longest:.1f}%)',
            f'Pred. ORF (not longest ORF) (n={n_not_longest})' : f'{orf_count_not_longest} ({100*orf_count_not_longest/n_not_longest:.1f}%)'}

def print_latex_table(df): 


    df = df[['ID','tscript_len','true_peptide_len','peptide_is_longest','pred','first_beam']] 
    df = df.sort_values(by=['first_beam','true_peptide_len'],ascending=[True,False])
    table = df.to_latex(index=False,column_format='l'*len(df.columns))
    table = re.sub("&(\s*)","& ",table)
    table = table.replace('True',r'\cmark')
    table = table.replace('False',r'\xmark')
    table = table.replace('9999',r'\xmark')
    print(table)

def get_model_names(filename):
    
    model_list = [] 
    with open(filename) as inFile:
        model_list += [x.rstrip().replace('/','').replace('.pt','') for x in inFile.readlines()]
    return model_list

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
    test_file = os.path.join(args.data_dir,'lnc_PEP.csv')
    models = get_model_names('lncPEP_models.txt')
    
    if 'parse' in unknown_args:
        mode = 'parse'
    elif 'align' in unknown_args:
        mode = 'align'
    else:
        raise ValueError("mode must be \'align\' or \'parse\'")

    results = [] 
    for model in models:
        if 'seq2start' in model or 'CDS' in model:
            best_preds = os.path.join('experiments/output',os.path.join(model,f'lnc_PEP_RNA_preds.txt'))
            if mode == 'align':
                print('align is unnecessary for a seq2start model, skipping')
            else:
                results.append(parse_start(best_preds,test_file,model))
        else:
            best_preds = os.path.join('experiments/output',os.path.join(model,f'lnc_PEP_RNA_full_preds.txt'))
            results.append(parse(best_preds,test_file,model,mode=mode))

    if mode == 'parse':
        print('test_samba_lncPEP.tsv')
        results.append(eval_samba('test_samba_lncPEP.tsv',test_file,'RNAsamba_lncPEP'))
        comparison = pd.DataFrame(results)
        short_string = comparison['Model'].str.extract(r'([^\/]*)_(lncPEP)')
        comparison['Model'] = short_string[0].apply(rename)
        table = comparison.to_latex(index=False,column_format='l'*len(comparison.columns))
        print(table)
