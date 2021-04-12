import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def partition(val):
    
    if val >= 0.0 and val < 0.4:
        return 0
    elif val >= 0.4 and val < 0.6:
        return 2
    elif val >= 0.6 and val < 0.8:
        return 3
    elif val >= 0.8 and val <= 1.0:
        return 4
    else:
        return 5

def violin_by_bin(alignments_file,similarity_file):
    
    align_df = pd.read_csv(alignments_file,sep="\t",names=["tscript","gold_match"]).set_index("tscript")
    similarity_df = pd.read_csv(similarity_file,sep="\t",names=["tscript","train_match"]).set_index("tscript")

    merged = similarity_df.join(align_df)
    merged = merged[pd.notna(merged.gold_match)]
    merged['partition'] = [partition(v) for v in merged['train_match'].values.tolist()]
    counts = merged.groupby('partition').count()['train_match'].tolist()
    group = merged.groupby('partition')
    print(group.describe())
    pal = sns.color_palette("muted")
    ax = sns.violinplot(data=merged,x='partition',y='gold_match',cut=0,width=1,palette=pal)
    labels =  ['[0.0,0.4)','[0.4,0.6)','[0.6,0.8)','[0.8,1.0]']
    labels = [l+'\nN = {}'.format(n) for l,n in zip(labels,counts)]
    ax.set_xticklabels(labels)
    ax.set_ylabel('% Identity predicted vs true protein')
    ax.set_xlabel( '% Identity closest match in train set')
    plt.tight_layout()
    plt.savefig('train_test_similarity.svg')
    plt.close()

if __name__ == "__main__":
    
    violin_by_bin(sys.argv[1],sys.argv[2])
