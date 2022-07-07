#from bioseq2seq.evaluate.evaluator import Evaluator
import sys,re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef
from scipy.stats import chisquare, ks_2samp, mannwhitneyu
from os import listdir
from os.path import isfile, join
import seaborn as sns

def findPositionProbability(position_x, base):
    '''Calculate the Position probablity of a base in codon'''
    coding_probs = []
    code_prob = 0
    if (base == "A"):
        coding_probs = [.22, .20, .34, .45, .68, .58, .93, .84, .68, .94]
    if (base == "C"):
        coding_probs = [.23, .30, .33, .51, .48, .66, .81, .70, .70, .80]
    if (base == "G"):
        coding_probs = [.08, .08, .16, .27, .48, .53, .64, .74, .88, .90]
    if (base == "T"):
        coding_probs = [.09, .09, .20, .54, .44, .69, .68, .91, .97, .97]

    if (position_x >= 0 and position_x < 1.1):
        code_prob = coding_probs[0]
    elif (position_x >= 1.1 and position_x < 1.2):
        code_prob = coding_probs[1]
    elif (position_x >= 1.2 and position_x < 1.3):
        code_prob = coding_probs[2]
    elif (position_x >= 1.3 and position_x < 1.4):
        code_prob = coding_probs[3]
    elif (position_x >= 1.4 and position_x < 1.5):
        code_prob = coding_probs[4]
    elif (position_x >= 1.5 and position_x < 1.6):
        code_prob = coding_probs[5]
    elif (position_x >= 1.6 and position_x < 1.7):
        code_prob = coding_probs[6]
    elif (position_x >= 1.7 and position_x < 1.8):
        code_prob = coding_probs[7]
    elif (position_x >= 1.8 and position_x < 1.9):
        code_prob = coding_probs[8]
    elif (position_x > 1.9):
        code_prob = coding_probs[9]
    return code_prob


def findContentProbability(position_x, base):
    ''' Find the composition probablity of base in codon '''
    coding_probs = []
    code_prob = 0
    if (base == "A"):
        coding_probs = [.21, .81, .65, .67, .49, .62, .55, .44, .49, .28]
    if (base == "C"):
        coding_probs = [.31, .39, .44, .43, .59, .59, .64, .51, .64, .82]
    if (base == "G"):
        coding_probs = [.29, .33, .41, .41, .73, .64, .64, .47, .54, .40]
    if (base == "T"):
        coding_probs = [.58, .51, .69, .56, .75, .55, .40, .39, .24, .28]

    if (position_x >= 0 and position_x < .17):
        code_prob = coding_probs[0]
    elif (position_x >= .17 and position_x < .19):
        code_prob = coding_probs[1]
    elif (position_x >= .19 and position_x < .21):
        code_prob = coding_probs[2]
    elif (position_x >= .21 and position_x < .23):
        code_prob = coding_probs[3]
    elif (position_x >= .23 and position_x < .25):
        code_prob = coding_probs[4]
    elif (position_x >= .25 and position_x < .27):
        code_prob = coding_probs[5]
    elif (position_x >= .27 and position_x < .29):
        code_prob = coding_probs[6]
    elif (position_x >= .29 and position_x < .31):
        code_prob = coding_probs[7]
    elif (position_x >= .31 and position_x < .33):
        code_prob = coding_probs[8]
    elif (position_x > .33):
        code_prob = coding_probs[9]
    return code_prob


def ficketTestcode(seq):
    ''' The driver function '''
    baseOne = [0, 0, 0, 0]
    baseTwo = [0, 0, 0, 0]
    baseThree = [0, 0, 0, 0]
    seq.upper()

    for pos_1, pos_2, pos_3 in zip(range(0, len(seq), 3), range(1, len(seq), 3), range(2, len(seq), 3)):

        # Base one
        if (seq[pos_1] == "A"):
            baseOne[0] = baseOne[0] + 1
        elif (seq[pos_1] == "C"):
            baseOne[1] = baseOne[1] + 1
        elif (seq[pos_1] == "G"):
            baseOne[2] = baseOne[2] + 1
        elif (seq[pos_1] == "T"):
            baseOne[3] = baseOne[3] + 1

        # Base two
        if (seq[pos_2] == "A"):
            baseTwo[0] = baseTwo[0] + 1
        elif (seq[pos_2] == "C"):
            baseTwo[1] = baseTwo[1] + 1
        elif (seq[pos_2] == "G"):
            baseTwo[2] = baseTwo[2] + 1
        elif (seq[pos_2] == "T"):
            baseTwo[3] = baseTwo[3] + 1

        # Base two
        if (seq[pos_3] == "A"):
            baseThree[0] = baseThree[0] + 1
        elif (seq[pos_3] == "C"):
            baseThree[1] = baseThree[1] + 1
        elif (seq[pos_3] == "G"):
            baseThree[2] = baseThree[2] + 1
        elif (seq[pos_3] == "T"):
            baseThree[3] = baseThree[3] + 1

    position_A = max(baseOne[0], baseTwo[0], baseThree[0]) / (min(baseOne[0], baseTwo[0], baseThree[0]) + 1)
    position_C = max(baseOne[1], baseTwo[1], baseThree[1]) / (min(baseOne[1], baseTwo[1], baseThree[1]) + 1)
    position_G = max(baseOne[2], baseTwo[2], baseThree[2]) / (min(baseOne[1], baseTwo[2], baseThree[2]) + 1)
    position_T = max(baseOne[3], baseTwo[3], baseThree[3]) / (min(baseOne[3], baseTwo[3], baseThree[3]) + 1)

    content_A = (baseOne[0] + baseTwo[0] + baseThree[0]) / len(seq)
    content_C = (baseOne[1] + baseTwo[1] + baseThree[1]) / len(seq)
    content_G = (baseOne[2] + baseTwo[2] + baseThree[2]) / len(seq)
    content_T = (baseOne[3] + baseTwo[3] + baseThree[3]) / len(seq)

    position_A_prob = findPositionProbability(position_A, "A")
    position_C_prob = findPositionProbability(position_C, "C")
    position_G_prob = findPositionProbability(position_G, "G")
    position_T_prob = findPositionProbability(position_T, "T")

    content_A_prob = findContentProbability(content_A, "A")
    content_C_prob = findContentProbability(content_C, "C")
    content_G_prob = findContentProbability(content_G, "G")
    content_T_prob = findContentProbability(content_T, "T")
    ficket_score = position_A_prob * .26 + content_A_prob * .11 + position_C_prob * .18 + content_C_prob * .12 + position_G_prob * .31 + content_G_prob * .15 + position_T_prob * .33 + content_T_prob * .14
    return ficket_score

def count_GC(rna):
    
    gc = set(['G','C'])
    return sum([1 for c in rna if c in gc]) / len(rna) 

def get_CDS_loc(cds,rna):
    
    # use provided CDS for mRNA 
    if cds != "-1": 
        splits = cds.split(":")
        clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
        splits = [clean(x) for x in splits]
        start,end = tuple([int(x) for x in splits])
    # impute longest ORF as CDS for lncRNA
    else:
        start,end = getLongestORF(rna)
    
    return start,end

def getLongestORF(mRNA):
    ORF_start = -1
    ORF_end = -1
    longestORF = 0
    for startMatch in re.finditer('ATG',mRNA):
        remaining = mRNA[startMatch.start():]
        if re.search('TGA|TAG|TAA',remaining):
            for stopMatch in re.finditer('TGA|TAG|TAA',remaining):
                ORF = remaining[0:stopMatch.end()]
                if len(ORF) % 3 == 0:
                    if len(ORF) > longestORF:
                        ORF_start = startMatch.start()
                        ORF_end = startMatch.start()+stopMatch.end()
                        longestORF = len(ORF)
                    break
    return ORF_start,ORF_end

def get_all_stops(mRNA):
    
    return [m.start() for m in re.finditer('TAG|TGA|TAA', mRNA)]

def parse_predictions(record):
    
    transcript = record[0].split('ID: ')[1]
    pred_match  = re.search('PRED: (<PC>|<NC>)(\S*)?',record[1])
    
    if pred_match is not None:
        pred_class = pred_match.group(1)
        pred_peptide = pred_match.group(2)
    else:
        raise ValueError('Bad format')
   
    is_coding = lambda x : x.startswith('XM') or x.startswith('NM')
    gold_class = '<PC>' if is_coding(transcript) else '<NC>' 

    entry = {'transcript' : transcript,
            'pred_class' : pred_class, 
            'pred_seq': pred_peptide,
            'gold_class' : gold_class,
            'gold_seq' : ''}
    return entry

def has_upstream_inframe_stop(start,stop_locs):

    start_frame = start % 3
    for s in stop_locs:
        if s < start:
            if s % 3 == start_frame:
                return True,s
    return False,-1

def count_upstream_inframe_stops(df):

    ids = df.index.tolist()
    seqs = df['RNA'].tolist()
    cds = df['CDS'].tolist()

    all_cds_list = [get_CDS_loc(c,s)[0] for c,s in zip(cds,seqs)]
    all_stops_list = [get_all_stops(x) for x in seqs]

    upstream_list = [has_upstream_inframe_stop(a,b) for a,b in zip(all_cds_list,all_stops_list)]
    n_upstream_stops = 0
    for a,b in zip(ids,upstream_list):
        if b[0]:
            n_upstream_stops+=1
    return n_upstream_stops, len(upstream_list) - n_upstream_stops


def analysis_pipeline(groupA_df,groupB_df,names,test='KS'):

    if test == 'KS':
        stat_name = 'Two-sample KS'
    else:
        stat_name = 'Mann-Whitney U'


    obs_stops,obs_no_stops = count_upstream_inframe_stops(groupA_df)
    exp_stops,exp_no_stops = count_upstream_inframe_stops(groupB_df)
    calc_stops = (exp_stops)/(exp_stops+exp_no_stops)*(obs_stops+obs_no_stops)
    calc_no_stops = obs_stops+obs_no_stops - calc_stops
    chisq,p = chisquare(f_obs=[obs_stops,obs_no_stops],f_exp=[calc_stops,calc_no_stops])
    print(f'Upstream in-frame stop codon {names[0]} ({obs_stops}/{len(groupA_df)}={obs_stops/len(groupA_df):.3f}) vs {names[1]} ({exp_stops}/{len(groupB_df)}={exp_stops/len(groupB_df):.3f}) (Chi-square) p={p:.3e}')

    groupA_gc = groupA_df['GC_content'].to_numpy()
    groupA_gc_mean = np.mean(groupA_gc)
    groupA_gc_std = np.std(groupA_gc)
    groupB_gc = groupB_df['GC_content'].to_numpy()
    groupB_gc_mean = np.mean(groupB_gc)
    groupB_gc_std = np.std(groupB_gc)
   
    if test == 'KS':
        stat,p = ks_2samp(groupA_gc,groupB_gc)
    else:
        stat,p = mannwhitneyu(groupA_gc,groupB_gc)
    print(f'GC content {names[0]} ({groupA_gc_mean:.2f}+-{groupA_gc_std:.2f}) vs {names[1]} ({groupB_gc_mean:.2f}+-{groupB_gc_std:.2f}) ({stat_name}) p={p:.2e}')

    groupA_fickett = groupA_df['Fickett'].to_numpy()
    groupA_fickett_mean = np.mean(groupA_fickett)
    groupA_fickett_std = np.std(groupA_fickett)
    groupB_fickett = groupB_df['Fickett'].to_numpy()
    groupB_fickett_mean = np.mean(groupB_fickett)
    groupB_fickett_std = np.std(groupB_fickett)
    
    if stat ==  'KS': 
        stat,p = ks_2samp(groupA_fickett,groupB_fickett)
    else:
        stat,p = mannwhitneyu(groupA_fickett,groupB_fickett)
    print(f'Fickett TESTCODE {names[0]} ({groupA_fickett_mean:.2f}+-{groupA_fickett_std:.2f}) vs {names[1]} ({groupB_fickett_mean:.2f}+-{groupB_fickett_std:.2f}) ({stat_name}) p={p:.2e}')

    groupA_cds = groupA_df['CDS_len'].to_numpy()
    groupA_cds_mean = np.mean(groupA_cds)
    groupA_cds_std = np.std(groupA_cds)
    groupB_cds = groupB_df['CDS_len'].to_numpy()
    groupB_cds_mean = np.mean(groupB_cds)
    groupB_cds_std = np.std(groupB_cds)
    
    if test == 'KS': 
        stat,p = ks_2samp(groupA_cds,groupB_cds)
    else: 
        stat,p = mannwhitneyu(groupA_cds,groupB_cds,use_continuity=False)
    print(f'CDS length {names[0]} ({groupA_cds_mean:.2f}+-{groupA_cds_std:.2f}) vs {names[1]} ({groupB_cds_mean:.2f}+-{groupB_cds_std:.2f}) ({stat_name}) p={p:.2e}')

    groupA_rna = groupA_df['RNA_len'].to_numpy()
    groupA_rna_mean = np.mean(groupA_rna)
    groupA_rna_std = np.std(groupA_rna)
    groupB_rna = groupB_df['RNA_len'].to_numpy()
    groupB_rna_mean = np.mean(groupB_rna)
    groupB_rna_std = np.std(groupB_rna)
    
    if test == 'KS':
        stat,p = ks_2samp(groupA_rna,groupB_rna)
    else:
        stat,p = mannwhitneyu(groupA_rna,groupB_rna,use_continuity=False)
    print(f'RNA length {names[0]} ({groupA_rna_mean:.2f}+-{groupA_rna_std:.2f}) vs {names[1]} ({groupB_rna_mean:.2f}+-{groupB_rna_std:.2f}) ({stat_name}) p={p:.2e}')

    groupA_cds_cov = groupA_df['CDS_coverage'].to_numpy()
    groupA_cds_cov_mean = np.mean(groupA_cds_cov)
    groupA_cds_cov_std = np.std(groupA_cds_cov)
    groupB_cds_cov = groupB_df['CDS_coverage'].to_numpy()
    groupB_cds_cov_mean = np.mean(groupB_cds_cov)
    groupB_cds_cov_std = np.std(groupB_cds_cov)
    
    if test == 'KS':
        stat,p = ks_2samp(groupA_cds_cov,groupB_cds_cov)
    else:
        stat,p = mannwhitneyu(groupA_cds_cov,groupB_cds_cov)
    print(f'CDS coverage {names[0]} ({groupA_cds_cov_mean:.2f}+-{groupA_cds_cov_std:.2f}) vs {names[1]} ({groupB_cds_cov_mean:.2f}+-{groupB_cds_cov_std:.2f}) ({stat_name}) p={p:.2e}')

def evaluate(pred_file):

    storage = []
    with open(pred_file,"r") as inFile:
        lines = inFile.read().split("\n")
        for i in range(0,len(lines)-6,6):
            entry = parse_predictions(lines[i:i+6])
            storage.append(entry)
    df = pd.DataFrame(storage)

    preds = df['pred_class'].to_numpy() 
    gt = df['gold_class'].to_numpy()
    tscripts = df['transcript'].to_numpy()
    pos_preds = preds == '<PC>'
    pos_gt = gt == '<PC>'

    accuracy = (preds == gt).sum() / len(df)

    separator = '_____________________________________________________________________'
    TP = tscripts[pos_preds & pos_gt]
    FP = tscripts[pos_preds & ~pos_gt]
    TN  = tscripts[~pos_preds & ~pos_gt]
    FN  = tscripts[~pos_preds & pos_gt]
    #print(f'{separator}\nPerformance Summary\n{separator}')
    #print(f'TP = {TP.shape[0]}  FP = {FP.shape[0]}\nTN = {TN.shape[0]} FN = {FN.shape[0]}')
    #print(f'accuracy = {accuracy}')
    preds = pos_preds.astype(int)
    gt = pos_gt.astype(int)
    mcc = matthews_corrcoef(gt,preds)
    f1 = f1_score(gt,preds)
    #print(f'MCC = {mcc}, F1 = {f1}')
    name = pred_file.split('.')[0]
    return {'trial' : name , 'validation performance-accuracy' : accuracy, 'validation performance-F1' : f1 , 'validation performance-MCC' : mcc}


all_model_performances = []

onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
all_preds = [f for f in onlyfiles if f.endswith('preds')]
print(all_preds)

all_results = []
for f in all_preds:
    results = evaluate(f)
    model = 'EDC' if results['trial'].startswith('EDC') else 'bioseq2seq'
    results['model'] = model
    all_results.append(results)

results_df = pd.DataFrame(all_results)
results_df['id'] = results_df.index
long_df = pd.wide_to_long(results_df,stubnames='validation performance',i='id',j='metric',suffix=r'\w+\d?',sep='-')
long_df = long_df.reset_index()
print(long_df)
sns.set_theme(style='whitegrid')
ax = sns.boxplot(data=long_df,x='metric',y='validation performance',hue='model',width=0.6,palette='Set2')
plt.xlabel('')
plt.savefig('boxplot_comparison.svg')
plt.close()
'''
val_file = 'new_data/mammalian_200-1200_val_nonredundant_80.csv'
val_df = pd.read_csv(val_file,sep='\t').set_index('ID')

# add columns of interest
val_df['annotation_status'] = ['confirmed' if x.startswith('N') else 'putative' for x in val_df.index.tolist()]
val_df['RNA_len'] = [len(x) for x in val_df['RNA'].tolist()]

ORF_lens = []
for c,r,p in zip(val_df['Type'].tolist(),val_df['RNA'].tolist(), val_df['Protein'].tolist()):
    if c == '<NC>':
        s,e = getLongestORF(r)
        length = e-s
    else:
        length = 3*(len(p)+1)
    ORF_lens.append(length)

val_df['CDS_len'] = ORF_lens
val_df['CDS_coverage'] = val_df['CDS_len'] / val_df['RNA_len']
cds_loc = [get_CDS_loc(c,r) for c,r in zip(val_df['CDS'].tolist(),val_df['RNA'].tolist())]
val_df['GC_content'] = [count_GC(x) for x in val_df['RNA'].tolist()] 
val_df['Fickett'] = [ficketTestcode(x) for x in val_df['RNA'].tolist()] 
#val_df['GC_content'] = [count_GC(x[s:e]) if e-s > 0 else 0.0 for x,(s,e) in zip(val_df['RNA'].tolist(),cds_loc)] 

correct_df = val_df.loc[TN.tolist()+TP.tolist()]
fn_df = val_df.loc[FN]
fp_df = val_df.loc[FP]
tp_df = val_df.loc[TP]
tn_df = val_df.loc[TN]

#print('FP')
#print(fp_df[fp_df['annotation_status'] == 'confirmed'])


#print('FN')
#print(fn_df[fn_df['annotation_status'] == 'confirmed'])

print(f'{separator}\n False Negatives vs True Positives\n{separator}')
analysis_pipeline(fn_df,tp_df,('FN','TP'),test='MW')
print(f'{separator}\n False Positives vs True Negatives \n{separator}')
analysis_pipeline(fp_df,tn_df,('FP','TN'),test='MW')
print(f'{separator}\n False Positives vs False Negatives \n{separator}')
analysis_pipeline(fp_df,fn_df,('FP','FN'),test='MW')
print(f'{separator}\n False Positives vs True Positives \n{separator}')
analysis_pipeline(fp_df,tp_df,('FP','TP'),test='MW')
print(f'{separator}\n False Negatives vs True Negatives \n{separator}')
analysis_pipeline(fn_df,tn_df,('FN','TN'),test='MW')
'''
'''
best_scores, best_n_scores = evaluator.calculate_stats(all_preds,all_golds,all_ids,log_all=True)
for k,v in best_scores.items():
    vals = np.asarray(v)
    if  vals.size > 1:
        mean = np.mean(vals)
        std = np.std(vals)
        print("{} -  mean : {} std : {}".format(k,mean,std))
    else:
        print("{} - {}".format(k,vals))
'''