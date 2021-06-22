import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

#sns.set_style('darkgrid')
plt.style.use('ggplot')
df = pd.read_csv(sys.argv[1])
plt.figure(figsize=(4,6))
ax = sns.boxplot(data=df,x="model",y="F1",order=["EDC","bioseq2seq"])
plt.xlabel('Network architecture')
plt.ylabel('F1-score in transcript classification task')
plt.tight_layout()
plt.savefig('model_comparison.svg')
plt.close()
