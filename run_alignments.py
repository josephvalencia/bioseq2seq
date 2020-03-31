from bioseq2seq.bin.evaluator import Evaluator
import sys
import numpy as np
import matplotlib.pyplot as plt

evaluator = Evaluator(best_of = 4,full_align=True,exact_match=True)

with open(sys.argv[1],"r") as inFile:
    lines = inFile.read().split("\n")
print(len(lines))
all_ids = []
all_golds = []
all_preds = []

for i in range(0,len(lines),8):

    id = lines[i].split("ID: ")[1].rstrip()
    print(i)
    preds = [x.rstrip().split("PRED: ")[1] for x in lines[i+2:i+6]]
    print(preds)

    gold = lines[i+6].rstrip().split("GOLD: ")[1]
    all_ids.append(id)
    all_golds.append(gold)
    all_preds.append(preds)

best_scores, best_n_scores = evaluator.calculate_stats(all_preds,all_golds,all_ids,log_all=True)
