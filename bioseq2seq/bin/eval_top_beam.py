#!/usr/bin/env python
import argparse
import re
import numpy as np
from math import ceil
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score,precision_recall_fscore_support,pairwise_distances,confusion_matrix,matthews_corrcoef
from bioseq2seq.evaluate.ensemble import Vote, Prediction, prediction_from_record
from bioseq2seq.evaluate.evaluator import Evaluator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    """ Parse required and optional configuration arguments"""

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--tune_ensemble",action="store_true",help="Calculate optimal decision thresholds." )

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--output_name","--o", default = "translation",help = "Name of file for saving predicted translations")
    parser.add_argument("--top_n", default = 3,type=int,help = "The number of highest model thresholds to save when --tune_ensemble is used.")
    parser.add_argument("--models_file", "--m",help="File of paths to models")
    
    return parser.parse_args()

def evaluate(args):

    with open(args.models_file) as inFile:
        model_paths = [x.rstrip() for x in inFile.readlines()]

    # classification functions
    is_coding_gt = lambda x : 1 if x.startswith('NM') or x.startswith('XM') else 0
    is_coding_pred = lambda x : 1 if x == '<PC>' else 0 
    is_coding_thresh = lambda x,t : 1 if x > t else 0 
    is_coding_majority = lambda x,y : 1 if x >= ceil(0.5*y) else 0

    consensus_probs = defaultdict(list)
    ensemble_storage = defaultdict(lambda : defaultdict(float))
    gt = []
    performance = []
    collect_GT = True 
    
    for f in model_paths: 
        best_preds = []
        gt = []
        with open(f) as inFile:
            lines = [x.rstrip() for x in inFile.readlines()]
            for i in range(0,len(lines),9):
                record = lines[i:i+9]
                pred = prediction_from_record(record,f)
                ensemble_storage[pred.transcript][f] = pred
                composite_vote = pred.prob_from_beam_energies()
                consensus_probs[f].append(composite_vote)
                best_preds.append(is_coding_pred(pred.votes[0].pred_class))
                if collect_GT:
                    gt.append(is_coding_gt(pred.transcript))
            # after first pass, no need to collect ground truth
            #collect_GT = False
            metrics = calculate_metrics(f,gt,best_preds)
            performance.append(metrics)
    
    output_file = args.output_name + '_top_beam_results.csv'
    df = pd.DataFrame(performance)
    df.to_csv(output_file,index=False,sep="\t")

    if args.tune_ensemble:
        # test different thresholds
        storage = []
        for f in model_paths:
            for thresh in [0.01*x for x in range(1,100)]:
                preds = [is_coding_thresh(x,thresh) for x in consensus_probs[f]]
                entry = calculate_metrics(f,gt,preds)
                entry['threshold'] = thresh
                storage.append(entry)
        df = pd.DataFrame(storage)
        maxes = df.loc[df.groupby(['model'])['F1'].idxmax()]
        top_maxes = maxes.sort_values(by='F1',ascending=False).iloc[:args.top_n]
        ensemble_thresholds = top_maxes.reset_index()[['model','threshold']] 
        ensemble_thresholds['model'] = [m.replace('val','test') for m in ensemble_thresholds['model'].tolist()]
        ensemble_file = f'{args.output_name}_top{args.top_n}.ensemble'
        ensemble_thresholds.to_csv(ensemble_file,index=False,sep=',')

def calculate_metrics(model,ground_truth,predictions):

    tn, fp, fn, tp = confusion_matrix(ground_truth,predictions).ravel()
    specificity = safe_divide(tn,tn+fp)
    precision,recall,fscore,support = precision_recall_fscore_support(ground_truth,predictions,average='binary')
    mcc = matthews_corrcoef(ground_truth,predictions)
    metrics = {'model' : model, 'F1' : fscore, 'recall' : recall,'specificity' : specificity ,'precision' : precision , 'MCC' : mcc}
    return metrics

def safe_divide(num,denom):

    if denom > 0:
        return float(num) / denom
    else:
        return 0.0

if __name__ == "__main__":

    args = parse_args()
    evaluate(args)
