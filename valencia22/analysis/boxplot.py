import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

df = pd.read_csv('results.csv')
g = sns.boxplot(data=df,x="model",y="accuracy",order = ['EDC-small','EDC-large','bioseq2seq'],palette='Set2',width=0.40)
plt.rcParams["axes.labelsize"] = 24
plt.savefig('results_boxplot.png')
plt.close()
