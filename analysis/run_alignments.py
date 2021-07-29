from bioseq2seq.bin.evaluator import Evaluator
import sys,re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

def parse_predictions(record):
    
    transcript = record[0].split('ID: ')[1]
    src = record[1].split('RNA: ')[1]
    pc_score = np.exp(float(record[6].split('PC_SCORE: ')[1]))
    preds = record[2:6]
    gold_match  = re.search('GOLD: (<PC>|<NC>)(\S*)?',record[-2])

    pred_list = []
    for p in preds:
        pred_match  = re.search('PRED: (<PC>|<NC>)(\S*)?',p)
        if pred_match is not None:
            pred_class = pred_match.group(1)
            pred_peptide = pred_match.group(2)
            pred_list.append(pred_class+pred_peptide)
        else:
            pred_list.append('?')
    if gold_match is not None:
        gold_class = gold_match.group(1)
        gold_peptide = gold_match.group(2)
        gold = gold_class+gold_peptide
    else:
        print('Uh oh')
        gold = 'NC>'

    return pred_list,gold,transcript

pred_file = sys.argv[1]
mode = sys.argv[2]

if mode == "classify":
    evaluator = Evaluator(mode=mode,best_of=1,k=0,full_align=False,exact_match=False)
elif mode == "combined":
    evaluator = Evaluator(mode=mode,best_of=1,k=8,full_align=True,exact_match=True)

with open(pred_file,"r") as inFile:
    lines = inFile.read().split("\n")
all_ids = []
all_golds = []
all_preds = []

'''
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
'''
    
for i in range(0,len(lines)-9,9):
    vals = parse_predictions(lines[i:i+9])
    if vals is not None:
        pred,gold,transcript = vals 
        all_preds.append(pred)
        all_golds.append(gold)
        all_ids.append(transcript)

best_scores, best_n_scores = evaluator.calculate_stats(all_preds,all_golds,all_ids,log_all=True)
for k,v in best_scores.items():
    vals = np.asarray(v)
    if  vals.size > 1:
        mean = np.mean(vals)
        std = np.std(vals)
        print("{} -  mean : {} std : {}".format(k,mean,std))
    else:
        print("{} - {}".format(k,vals))
