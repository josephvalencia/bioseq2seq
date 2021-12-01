import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support,confusion_matrix,matthews_corrcoef
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

    tn, fp, fn, tp = confusion_matrix(ground_truth,predictions).ravel()
    specificity = safe_divide(tn,tn+fp)
    precision,recall,fscore,support = precision_recall_fscore_support(ground_truth,predictions,average='binary')
    mcc = matthews_corrcoef(ground_truth,predictions)
    metrics = {'model' : model, 'dataset' : dataset, 'F1' : fscore, 'recall' : recall,'specificity' : specificity ,'precision' : precision ,'MCC' : mcc}
    return metrics

def safe_divide(num,denom):

    if denom > 0:
        return float(num)/ denom
    else:
        return 0.0

if __name__ == "__main__":
  
    '''
    storage = []
    for d in ['mammalian_1k','mammalian_1k-2k','zebrafish_1k']:
        samba = f'test_rnasamba_{d}.tsv'
        storage.append(eval_rnasamba(samba,d)) 
        cpc = f'test_cpc2_{d}.txt'
        storage.append(eval_cpc2(cpc,d)) 
        cpat_a = f'test_cpat_{d}.ORF_prob.best.tsv'
        cpat_b = f'test_cpat_{d}.no_ORF.txt'
        storage.append(eval_cpat(cpat_a,cpat_b,d))

    df = pd.DataFrame(storage)
    df.to_csv('competitors_test_results.csv',sep='\t',index=False)
    '''

    results = eval_rnasamba('mammalian_200-1200_rnasamba_test.tsv','new_test')
    print(results)
