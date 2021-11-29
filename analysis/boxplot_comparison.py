import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import re

def dataset_boxplot(best_bioseq2seq,best_EDC,competitors,consensus_bioseq2seq):

    sns.set_style('whitegrid')
   
    # best bioseq2seq 
    best_bioseq2seq_df = pd.read_csv(best_bioseq2seq,sep='\t')
    data = 'test_([\w\-]*).preds' 
    best_bioseq2seq_df['dataset'] = [ re.search(data,x).group(1) for x in best_bioseq2seq_df['model'].tolist()]
    best_bioseq2seq_df['model'] = ['bioseq2seq'] * len(best_bioseq2seq_df)
   
    # best EDC  
    best_EDC_df = pd.read_csv(best_EDC,sep='\t')
    best_EDC_df['dataset'] = [ re.search(data,x).group(1) for x in best_EDC_df['model'].tolist()]
    best_EDC_df['model'] = ['EDC'] * len(best_bioseq2seq_df)
    
    competitors_df = pd.read_csv(competitors,sep='\t')
    consensus_bioseq2seq_df = pd.read_csv(consensus_bioseq2seq,sep='\t')
    consensus_bioseq2seq_df['model'] = ['bioseq2seq (consensus)'] * len(consensus_bioseq2seq_df)
    consensus_bioseq2seq_df['dataset'] = ['mammalian_1k'] * len(consensus_bioseq2seq_df)
     
    total = pd.concat([best_bioseq2seq_df,best_EDC_df,competitors_df,consensus_bioseq2seq_df])
    total = total[['dataset','model','F1','recall','precision','specificity','MCC']]
    total = total.sort_values(by=['dataset','model'])
    print(total.to_latex(index=False,float_format="{:0.3f}".format))
    
    long_df = pd.melt(total,id_vars=['model','dataset'],var_name = 'metric')
    
    aspect_ratio = 2
    size = 8.5

    # metric as the row 
    g1 = sns.FacetGrid(long_df,row='metric',hue='model',aspect = aspect_ratio)
    g1.map_dataframe(sns.stripplot,x='dataset',y='value',s=size,order=['mammalian_1k','mammalian_1k-2k','zebrafish_1k']) 
    g1.add_legend()
    
    for ax in g1.axes.flat:
        title = ax.get_title()
        metric = title.split(' = ')[1]
        ax.set_title('')
        ax.set_ylabel(metric)
    
    name = 'trials_by_metric_plot.svg'
    plt.savefig(name)
    plt.close()

    # dataset as the row
    g2 = sns.FacetGrid(long_df,row='dataset',hue='model',aspect = aspect_ratio)
    g2.map_dataframe(sns.stripplot,x='metric',y='value',s=size,order=['F1','recall','precision','specificity']) 
    g2.add_legend()
    for ax in g2.axes.flat:
        title = ax.get_title()
        metric = title.split(' = ')[1]
        ax.set_title('')
        ax.set_ylabel(metric)
    name = 'trials_by_dataset_plot.svg'
    plt.savefig(name)
    plt.close()

def model_boxplot(f_bio_ensemble,f_bio_top,f_EDC_top):

    sns.set_style('whitegrid')
    
    df_bio_ensemble = modified_df(f_bio_ensemble,'bioseq2seq\n(ensemble)')
    df_bio_top = modified_df(f_bio_top,'bioseq2seq\n(top beam)')
    df_EDC_top = modified_df(f_EDC_top,'EDC') 
    df_all = pd.concat([df_bio_ensemble,df_bio_top,df_EDC_top])
    long_df = pd.melt(df_all,id_vars=['model','replicate'],var_name = 'metric')
    
    #g = sns.FacetGrid(long_df,col='metric',aspect=aspect_ratio,sharey=False) 
    #g.map_dataframe(sns.boxplot,x='model',y='value',hue='model')#,boxprops={'facecolor':'None'})
    #g.map_dataframe(sns.stripplot,x='model',y='value',hue='model')
    
    plt.figure(figsize=(10,6))
    g = sns.boxplot(data=long_df,x='metric',y='value',hue='model',order=['F1','recall','precision','specificity','MCC'],palette='Set3')

    add_points = False
    if add_points: 
        plt.setp(g.artists, edgecolor = 'gray', facecolor='w')
        g = sns.stripplot(data=long_df,x='metric',y='value', hue='model',order=['F1','recall','precision','specificity','MCC'],jitter=True,dodge=True, marker='o', palette="Set2")
        handles, labels = g.get_legend_handles_labels()
        l = plt.legend(handles[0:3], labels[0:3],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    else:
        l = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    sns.despine()
    plt.ylabel('mammalian_1k performance')
    plt.xlabel('')
    ticks = [round(0.65+0.05*x,2) for x in range(8)]
    plt.yticks(ticks=ticks,labels=ticks)
    plt.tight_layout()
    plt.savefig('model_replicate_results.svg')
    plt.close()

def modified_df(filename,model_name):
    
    df = pd.read_csv(filename,sep='\t').rename(columns={'model' : 'old_model'})
    df['model'] = [model_name] * len(df)
    df['replicate'] = [find_num(x) for x in df['old_model']]
    return df.drop(columns=['old_model'])

def find_num(x):
    match = re.search('(\d)_test.preds',x)
    if match is not None:
        return int(match.group(1))
    else:
        return -1

if __name__ == "__main__":
    
    model_boxplot('bioseq2seq_test_ensemble_results.csv','bioseq2seq_test_top_beam_results.csv','EDC_test_top_beam_results.csv')
    dataset_boxplot('bioseq2seq_test_ALL_ensemble_results.csv','EDC_test_ALL_top_beam_results.csv','competitors_test_results.csv','bioseq2seq_test_ensemble_multi_replicate_results.csv')    
