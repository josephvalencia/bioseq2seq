import pandas as pd
from sklearn.metrics import auc, precision_recall_fscore_support,confusion_matrix,matthews_corrcoef, accuracy_score,roc_auc_score,average_precision_score,precision_recall_curve
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from utils import parse_config,build_output_dir,setup_fonts,palette_by_model 

def coarse_bin_fn(score):
    if score >=0 and score <= 40:
        return '[0-40]'
    elif score > 40 and score <= 60:
        return '(40-60]'        
    elif score > 60 and score <= 80:
        return '(60-80]'
    elif score > 80 and score <= 100:
        return '(80-100)'

def eval_cpc2(results_file,dataset,gt_df):
    
    col_names = ['ID','transcript_length','peptide_length','Fickett_score','pI','ORF_integrity','coding_probability','label']
    df = pd.read_csv(results_file,sep='\t',names=col_names)

    homology = pd.read_csv("test_maximal_homology.csv")
    df = df.merge(homology,on='ID') 
    df['bin'] = df['score'].apply(coarse_bin_fn)
    df = drop_missing_start(df,gt_df)
    results = []
    def true_and_pred(sub_df):
        gt = [1 if x.startswith('XM') or x.startswith('NM') else 0 for x in sub_df['ID'].tolist()]
        predictions = [1 if x == 'coding' else 0 for x in sub_df['label'].tolist()]
        probs = [float(x) for x in sub_df['coding_probability'].tolist()] 
        return gt,predictions,probs
    for bin,sub_df in df.groupby('bin'):
        gt,predictions,probs = true_and_pred(sub_df)
        results.append(calculate_metrics('CPC2',dataset,gt,predictions,probs,bin))
    lesser = df[df['score'] <= 80]
    gt,predictions,probs = true_and_pred(lesser)
    results.append(calculate_metrics('CPC2',dataset,gt,predictions,probs,'<=80'))
    greater = df[df['score'] > 80]
    gt,predictions,probs = true_and_pred(greater)
    results.append(calculate_metrics('CPC2',dataset,gt,predictions,probs,'>80'))

    return results

def eval_rnasamba(results_file,dataset,gt_df):
    results = []
    df = pd.read_csv(results_file,sep='\t')
    homology = pd.read_csv("test_maximal_homology.csv")
    df = df.merge(homology,left_on='sequence_name',right_on='ID')
    
    df = drop_missing_start(df,gt_df)

    df['bin'] = df['score'].apply(coarse_bin_fn)
    def true_and_pred(sub_df):
        gt = [1 if x.startswith('XM') or x.startswith('NM') else 0 for x in sub_df['sequence_name'].tolist()]
        probs = [float(x) for x in sub_df['coding_score'].tolist()] 
        predictions = [1 if x == 'coding' else 0 for x in sub_df['classification'].tolist()]
        return gt,predictions,probs
    for bin,sub_df in df.groupby('bin'):
        gt,predictions,probs = true_and_pred(sub_df) 
        results.append(calculate_metrics('RNAsamba',dataset,gt,predictions,probs,bin))
    lesser = df[df['score'] <= 80]
    gt,predictions,probs = true_and_pred(lesser)
    results.append(calculate_metrics('RNAsamba',dataset,gt,predictions,probs,'<=80'))
    greater = df[df['score'] > 80]
    gt,predictions,probs = true_and_pred(greater)
    results.append(calculate_metrics('RNAsamba',dataset,gt,predictions,probs,'>80'))
    return results

def eval_cpat(results_file,no_orfs_file,dataset,gt_df):

    df = pd.read_csv(results_file,sep='\t')
    df = df[['seq_ID','Coding_prob']]
    homology = pd.read_csv("test_maximal_homology.csv")

    with open(no_orfs_file) as inFile:
        no_orf_list = inFile.readlines()
    no_orf_df = pd.DataFrame([{'seq_ID' : x.rstrip() , 'Coding_prob' :0.0} for x in no_orf_list])
    df = pd.concat([df,no_orf_df])
    df = df.merge(homology,left_on='seq_ID',right_on='ID') 
    df['bin'] = df['score'].apply(coarse_bin_fn)
    df = drop_missing_start(df,gt_df)

    results = []
    # found using two ROC plot
    def true_and_pred(sub_df):
        threshold = 0.44
        gt = [1 if x.startswith('XM') or x.startswith('NM') else 0 for x in sub_df['seq_ID'].tolist()]
        probs = [float(x) for x in sub_df['Coding_prob'].tolist()] 
        predictions = [1 if x > threshold else 0 for x in probs]
        return gt,predictions,probs
    for bin,sub_df in df.groupby('bin'):
        gt,predictions,probs = true_and_pred(sub_df) 
        results.append(calculate_metrics('CPAT',dataset,gt,predictions,probs,bin))
    lesser = df[df['score'] <= 80]
    gt,predictions,probs = true_and_pred(lesser)
    results.append(calculate_metrics('CPAT',dataset,gt,predictions,probs,'<=80'))
    greater = df[df['score'] > 80]
    gt,predictions,probs = true_and_pred(greater)
    results.append(calculate_metrics('CPAT',dataset,gt,predictions,probs,'>80'))
    return results

def calculate_metrics(model,dataset,ground_truth,predictions,probs,bin):

    accuracy = accuracy_score(ground_truth,predictions)
    tn, fp, fn, tp = confusion_matrix(ground_truth,predictions).ravel()
    specificity = safe_divide(tn,tn+fp)
    precision,recall,fscore,support = precision_recall_fscore_support(ground_truth,predictions,average='binary')
    mcc = matthews_corrcoef(ground_truth,predictions)
    auroc = roc_auc_score(ground_truth,probs)
    auprc = average_precision_score(ground_truth,probs)
    pr,rc,thresh = precision_recall_curve(ground_truth,probs)
    alt = auc(rc,pr)
    plt.plot(rc,pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)
    plt.savefig(f'{model}_AUPRC.svg')
    plt.close()
    metrics = {'model' : model, 'bin' : bin, 'support' : len(ground_truth), 'dataset' : dataset, 'accuracy' : accuracy,\
            'F1' : fscore, 'recall' : recall, 'precision' : precision, 'specificity' :specificity, 'MCC' : mcc,\
            'AUROC' : auroc, 'AUPRC' : auprc}
    return metrics

def drop_missing_start(df,gt_df):
    
    df = df.set_index('ID')
    # a few examples are <PC> but with missing start codon annotation, omit 
    df = df.drop(gt_df[gt_df['CDS'].str.startswith('<0')]['ID'])
    df = df.reset_index() 
    return df

def safe_divide(num,denom):

    if denom > 0:
        return float(num)/ denom
    else:
        return 0.0

def plot_by_homology(df,output_dir):

    #coarse = (df['bin'] == '<=80') | (df['bin'] == '>80')
    coarse = (df['bin'] == '<=80')
    coarse_df = df.loc[coarse]
    bin_df = df.loc[~coarse]

    # binary separation at 80% homology
    #coarse_df['model'] = coarse_df['model'].add(coarse_df['bin']).add(' n=(').add(coarse_df['support'].astype(str)).add(')')
    cols = ['model','F1','recall','precision','MCC','AUROC','AUPRC']
    #cols = ['model', 'F1','recall','precision','MCC','AUPRC']
    print(coarse_df[['model','support','F1']]) 
    coarse_df = coarse_df[cols]
    by_type_mean = coarse_df.groupby('model').mean()
    by_type_std = coarse_df.groupby('model').std()
    print('Coarse Binning') 
    for_latex_mean = by_type_mean.applymap('{:.3f}'.format)
    for_latex_std = by_type_std.applymap('{:.3f}'.format)
    for_latex = for_latex_mean.add(' $\pm$ ').add(for_latex_std).reset_index().set_index('model')
    col_format = 'c'*len(for_latex.columns)
    table = for_latex.style.to_latex(column_format=f'|{col_format}|',hrules=True)
    table  = table.replace('<=80','$\leq80$').replace('>80','$>80$').replace('$\pm$ nan','') 
    table = table.replace('EDC_eq','EDC (small)').replace('EDC$','EDC (large)$') 
    print(table) 
    def rename(x):
        if x == 'rnasamba':
            return 'RNAsamba'
        elif x == 'EDC_eq':
            return 'EDC (small)'
        elif x == 'EDC':
            return 'EDC (large)'
        else:
            return x

    bin_df['Model'] = bin_df['model'].apply(rename) #[rename(x) for x in bin_df['model'].tolist()]
    order =['[0-40]', '(40-60]', '(60-80]'] #, '(80-100)']
    pbm = palette_by_model() 
    sns.set_style('ticks') 
    
    def plot_metric_homology(metric): 
        plt.figure(figsize=(6,3))
        sns.pointplot(data=bin_df,x='bin',y=metric,hue='Model',palette=pbm,order=order,errorbar='sd')
        sns.move_legend(plt.gca(),loc='upper left',bbox_to_anchor=(1,1))
        plt.xlabel('Max % id with train set')
        sns.despine() 
        plt.tight_layout()
        plt.savefig(f'{output_dir}/homology_binning_{metric}.svg')
        plt.close()

    plot_metric_homology('F1')
    plot_metric_homology('accuracy')
    plot_metric_homology('MCC')

if __name__ == "__main__":

    args,unknown_args = parse_config()
    setup_fonts()
    storage = []
    parent_dir = args.tools_dir 
    output_dir = build_output_dir(args)
    
    test_file = 'data/mammalian_200-1200_test_nonredundant_80.csv'
    gt_df = pd.read_csv(test_file,sep='\t')

    for i in range(1,6):
        samba = f'{parent_dir}/rnasamba/test_rnasamba_mammalian_{i}.tsv'
        storage.extend(eval_rnasamba(samba,i,gt_df)) 
    
    #storage.extend(eval_rnasamba('test_partial_length.tsv','partial',gt_df)) 
    #storage.extend(eval_rnasamba('test_full_length.tsv','full',gt_df))
    df = pd.DataFrame(storage)

    cpc = f'{parent_dir}/CPC2/test_cpc2_mammalian.txt'
    storage.extend(eval_cpc2(cpc,'mammalian',gt_df)) 
    cpat_a = f'{parent_dir}/CPAT/test_cpat_mammalian.ORF_prob.best.tsv'
    cpat_b = f'{parent_dir}/CPAT/test_cpat_mammalian.no_ORF.txt'
    storage.extend(eval_cpat(cpat_a,cpat_b,'mammalian',gt_df))
    df = pd.DataFrame(storage)
    ours = pd.read_csv('our_models_homology_binning.csv')
    df = pd.concat([df,ours])

    plot_by_homology(df,output_dir)

