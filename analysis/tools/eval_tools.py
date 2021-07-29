import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support,confusion_matrix
import sys

def eval_cpc2(results_file):

    df = pd.read_csv(results_file,sep='\t')
    df = df.set_index('#ID')

    is_coding_gt =  lambda x:  1 if x.startswith('XM') or x.startswith('NM') else 0
    is_coding_pred =  lambda x:  1 if x == "coding" else 0
    gt = [is_coding_gt(x) for x in df['#ID'].tolist()]
    predictions = [ is_coding_pred(x) for x in df['label'].tolist()]
    f1_results = f1_score(gt,predictions)
    print(f1_results)

def eval_rnasamba(results_file):
    
    gt = []
    predictions = []
    pred_probs = []

    with open(results_file) as inFile:
        for l in inFile:
            fields = l.split()
            transcript = fields[0]
            pred = fields[-1]
            pred_prob = float(fields[-2])
            true_class = 1 if transcript.startswith('XM') or transcript.startswith('NM') else 0
            pred_class = 1 if pred == "coding" else 0
            gt.append(true_class)
            predictions.append(pred_class)
            pred_probs.append(pred_prob)
    
    multi_results = precision_recall_fscore_support(gt,predictions,average='binary')
    tn, fp, fn, tp = confusion_matrix(gt,predictions).ravel()
    print(f'TN={tn} FP={fp}\nFN={fn} TP={tp}')
    roc_auc = roc_auc_score(gt,pred_probs)
    print(multi_results)

def eval_cpat(results_file,no_orfs_file):

    df = pd.read_csv(results_file,sep='\t')
    df = df[['seq_ID','Coding_prob']]

    with open(no_orfs_file) as inFile:
        no_orf_list = inFile.readlines()
    no_orf_df = pd.DataFrame([{'seq_ID' : x.rstrip() , 'Coding_prob' :0.0} for x in no_orf_list])
    df = pd.concat([df,no_orf_df])

    threshold = 0.44
    is_coding_gt =  lambda x:  1 if x.startswith('XM') or x.startswith('NM') else 0
    is_coding_pred =  lambda x:  1 if x > threshold  else 0
    gt = [is_coding_gt(x) for x in df['seq_ID'].tolist()]
    predictions = [is_coding_pred(x) for x in df['Coding_prob'].tolist()]
    f1_results = f1_score(gt,predictions)
    print(f1_results)

if __name__ == "__main__":
    
    eval_rnasamba(sys.argv[1])
    #eval_cpc2(sys.argv[1])
    #eval_cpat(sys.argv[1],sys.argv[2])

