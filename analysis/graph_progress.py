import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import  matplotlib.font_manager
# Supp Fig 1 

prefix = "accuracy_plots"
sns.set_theme(style="whitegrid")

for m in ['bioseq2seq','EDC']:
    storage = []
    plt.figure(figsize=(12,7))
    for i in range(1,6):
        f = '{}/{}_{}_val_accuracy.csv'.format(prefix,m,i)
        df_model = pd.read_csv(f)
        df_model['Step'] = [2500 * (j+1) for j in range(len(df_model))]
        df_model['Model'] = [m for _ in range(len(df_model))]
        df_model['Trial'] = [i for _ in range(len(df_model))]
        df_model = df_model[df_model['Step'] < 152500]
        storage.append(df_model)
    df = pd.concat(storage)
    ax = sns.lineplot(data=df,x='Step',y='Value',hue='Trial',palette="tab10")
    plt.ylabel('Validation accuracy')
    plt.xlabel('{} training step'.format(m))
    plt.savefig('val_accuracy_progress_{}.svg'.format(m))
    plt.close()
