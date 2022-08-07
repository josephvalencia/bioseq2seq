import pandas as pd
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(filename):

    df = pd.read_csv(filename,sep='\t',index_col=[0])
    df['diff_original'] = (df['optimized'] - df['original']) / df['original']
    df['diff_mean'] = (df['optimized'] - df['mean_sampled']) / df['mean_sampled']
    df['diff_best'] = (df['optimized'] - df['best_sampled']) / df['best_sampled']
    df['alt_id'] = df.index
    df = pd.wide_to_long(df,stubnames='diff',i='alt_id',j='reference',sep='_',suffix=".*").reset_index()
    print(df)
    ax = sns.violinplot(data=df,y="diff",x='reference')
    savename = filename.split('.rank')[0] 
    plt.savefig(f'figs/{savename}_violin.svg')
    plt.close()

if __name__ == "__main__":
    
    files = [f for f in os.listdir('.') if 'PC.EG' in f]
    for f in files: 
        evaluate(f)
