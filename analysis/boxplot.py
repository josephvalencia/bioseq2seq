import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')

ED_classify = [0.871,0.843,0.870,0.856]
seq2seq = [0.939,0.933,0.945,0.953,0.948]

storage = []

for s in ED_classify:
    entry = {'Network architecture':"EDC", "F1" : s}
    storage.append(entry)
for s in seq2seq:
    entry = {'Network architecture':"bioseq2seq", "F1" : s}
    storage.append(entry)

df = pd.DataFrame(storage)
print(df)
plt.figure(figsize=(3,2.5))
ax = sns.boxplot(data=df,x="Network architecture",y="F1",width=0.20,linewidth=1.0)
plt.tight_layout()
plt.savefig("model_boxplot.pdf")
plt.close()
