from bioseq2seq.bin.evaluator import Evaluator
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

pred_file = sys.argv[1]
mode = sys.argv[2]
subset = None

if len(sys.argv) > 3:
    sub_file = sys.argv[3]
    subset = set()
    with open(sub_file) as inFile:
        for l in inFile:
            subset.add(l.rstrip())

if mode == "classify":
    evaluator = Evaluator(mode=mode,best_of=1,k=0,full_align=False,exact_match=False)
elif mode == "combined":
    evaluator = Evaluator(mode=mode,best_of=1,k=8,full_align=True,exact_match=True)

with open(pred_file,"r") as inFile:
    lines = inFile.read().split("\n")
all_ids = []
all_golds = []
all_preds = []

for i in range(0,len(lines)-8,8):
    id = lines[i].split("ID: ")[1].rstrip()
    if subset is not None and id not in subset:
        continue
    preds = [x.rstrip().split("PRED: ") for x in lines[i+2:i+6]]
    preds = [x[1] if len(x) ==2 else "?" for x in preds]
    gold = lines[i+6].rstrip().split("GOLD: ")[1]
    all_ids.append(id)
    all_golds.append(gold)
    all_preds.append(preds)

best_scores, best_n_scores = evaluator.calculate_stats(all_preds,all_golds,all_ids,log_all=True)
for k,v in best_scores.items():
    vals = np.asarray(v)
    if  vals.size > 1:
        mean = np.mean(vals)
        std = np.std(vals)
        print("{} -  mean : {} std : {}".format(k,mean,std))
    else:
        print("{} - {}".format(k,vals))
