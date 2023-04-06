import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support,confusion_matrix,matthews_corrcoef, accuracy_score
import sys

def eval_cpc2(results_file,dataset):
    
    col_names = ['ID','transcript_length','peptide_length','Fickett_score','pI','ORF_integrity','coding_probability','label']
    df = pd.read_csv(results_file,sep='\t',names=col_names)

    is_coding_gt =  lambda x:  1 if x.startswith('XM') or x.startswith('NM') else 0
    is_coding_pred =  lambda x:  1 if x == "coding" else 0
    gt = [is_coding_gt(x) for x in df['ID'].tolist()]
    predictions = [ is_coding_pred(x) for x in df['label'].tolist()]

    return calculate_metrics('CPC2',dataset,gt,predictions)

def eval_rnasamba(results_file,dataset):
    
    gt = []
    predictions = []
    pred_probs = []

    with open(results_file) as inFile:
        for l in inFile.readlines()[1:]:
            fields = l.split()
            transcript = fields[0]
            pred = fields[-1]
            pred_prob = float(fields[-2])
            true_class = 1 if transcript.startswith('XM') or transcript.startswith('NM') else 0
            pred_class = 1 if pred == "coding" else 0
            gt.append(true_class)
            predictions.append(pred_class)
            pred_probs.append(pred_prob)
    
    return calculate_metrics('rnasamba',dataset,gt,predictions)

def eval_cpat(results_file,no_orfs_file,dataset):

    df = pd.read_csv(results_file,sep='\t')
    df = df[['seq_ID','Coding_prob']]

    with open(no_orfs_file) as inFile:
        no_orf_list = inFile.readlines()
    no_orf_df = pd.DataFrame([{'seq_ID' : x.rstrip() , 'Coding_prob' :0.0} for x in no_orf_list])
    df = pd.concat([df,no_orf_df])

    # found using two ROC plot
    threshold = 0.44
    is_coding_gt =  lambda x:  1 if x.startswith('XM') or x.startswith('NM') else 0
    is_coding_pred =  lambda x:  1 if x > threshold  else 0
    gt = [is_coding_gt(x) for x in df['seq_ID'].tolist()]
    predictions = [is_coding_pred(x) for x in df['Coding_prob'].tolist()]
    
    return calculate_metrics('CPAT',dataset,gt,predictions)

def calculate_metrics(model,dataset,ground_truth,predictions):

    accuracy = accuracy_score(ground_truth,predictions)
    tn, fp, fn, tp = confusion_matrix(ground_truth,predictions).ravel()
    specificity = safe_divide(tn,tn+fp)
    precision,recall,fscore,support = precision_recall_fscore_support(ground_truth,predictions,average='binary')
    mcc = matthews_corrcoef(ground_truth,predictions)
    metrics = {'model' : model, 'dataset' : dataset, 'accuracy' : accuracy, 'F1' : fscore, 'recall' : recall, 'precision' : precision, 'specificity' :specificity, 'MCC' : mcc}
    return metrics

def safe_divide(num,denom):

    if denom > 0:
        return float(num)/ denom
    else:
        return 0.0

if __name__ == "__main__":

    storage = []
    parent_dir = sys.argv[1]    
    
    for i in range(1,6):
        samba = f'{parent_dir}/rnasamba/test_rnasamba_mammalian_{i}.tsv'
        storage.append(eval_rnasamba(samba,i)) 
    
    cpc = f'{parent_dir}/CPC2/test_cpc2_mammalian.txt'
    storage.append(eval_cpc2(cpc,'mammalian')) 
    cpat_a = f'{parent_dir}/CPAT/test_cpat_mammalian.ORF_prob.best.tsv'
    cpat_b = f'{parent_dir}/CPAT/test_cpat_mammalian.no_ORF.txt'
    storage.append(eval_cpat(cpat_a,cpat_b,'mammalian'))
    df = pd.DataFrame(storage)
    by_type_mean = df.groupby('model').mean()
    by_type_std = df.groupby('model').std()
    for_latex_mean = by_type_mean.applymap('{:.3f}'.format)
    for_latex_std = by_type_std.applymap('{:.3f}'.format)
    for_latex = for_latex_mean.add(' $\pm$ ').add(for_latex_std)
    col_format = 'c'*len(for_latex.columns)
    table = for_latex.style.to_latex(column_format=f'|{col_format}|',hrules=True)
    print(table)
    df.to_csv('competitors_test_results.csv',sep='\t',index=False)

