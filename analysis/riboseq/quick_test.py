from sklearn.metrics import f1_score
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1],sep='\t',names=['transcript','pred','tgt','pc_prob','nc_prob'])

tgt = df['tgt'].values
pred = df['pred'].values
f1 = f1_score(tgt,pred,pos_label=24)
print('F1',f1)

pc_prob = df['pc_prob'].values.mean()
nc_prob = df['nc_prob'].values.mean()
print(pc_prob,nc_prob)
