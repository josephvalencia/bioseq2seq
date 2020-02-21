from bioseq2seq.bio.evaluator import Evaluator
import sys
import numpy as np
import matplotlib.pyplot as plt

evaluator = Evaluator(best_of = 4,full_align=True,exact_match=True)

with open(sys.argv[1],"r") as inFile:
    lines = inFile.readlines()

all_ids = []
all_golds = []
all_preds = []

for i in range(0,len(lines),7):

    id = lines[i].split("ID: ")[1].rstrip()
    preds = [x.rstrip().split("PRED: ")[1] for x in lines[i+1:i+5]]
    gold = lines[i+5].rstrip().split("GOLD: ")[1]
    all_ids.append(id)
    all_golds.append(gold)
    all_preds.append(preds)

best_scores, best_n_scores = evaluator.calculate_stats(all_preds,all_golds,all_ids,log_all=True)