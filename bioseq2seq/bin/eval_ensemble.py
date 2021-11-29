#!/usr/bin/env python
import argparse
import re
import numpy as np
from math import ceil
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score,precision_recall_fscore_support,pairwise_distances,confusion_matrix,matthews_corrcoef
from bioseq2seq.evaluate.ensemble import Vote, Prediction, prediction_from_record
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    """ Parse required and optional configuration arguments"""

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--consensus",action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--output_name","--o", default = "translation",help = "Name of file for saving predicted translations")
    parser.add_argument("--ensemble_file", "--m",help="File of paths to models")
    return parser.parse_args()

def ensemble_predict(args):

    ensemble_df = pd.read_csv(args.ensemble_file)
    model_paths = ensemble_df['model'].tolist()
    decision_vals = ensemble_df['threshold'].tolist()

    # classification functions
    is_coding_gt = lambda x : 1 if x.startswith('NM') or x.startswith('XM') else 0
    is_coding_pred = lambda x : 1 if x == '<PC>' else 0 
    is_coding_thresh = lambda x,t : 1 if x > t else 0 
    is_coding_majority = lambda x,y : 1 if x >= ceil(0.5*y) else 0

    performance = []

    for f,t in zip(model_paths,decision_vals): 
        best_preds = []
        with open(f) as inFile:
            lines = [x.rstrip() for x in inFile.readlines()]
            gt = []
            preds = []
            for i in range(0,len(lines),9):
                record = lines[i:i+9]
                pred = prediction_from_record(record,f)
                composite_vote = pred.prob_from_beam_energies()
                thresholded_vote = is_coding_thresh(composite_vote,t) 
                preds.append(thresholded_vote)
                true_class = is_coding_gt(pred.transcript)
                gt.append(true_class)
                 
        metrics = calculate_metrics(f,gt,preds)
        performance.append(metrics)
    
    performance_df = pd.DataFrame(performance)

    output_file = args.output_name +'_ensemble_results.csv'
    performance_df.to_csv(output_file,index=False,sep='\t')

def calculate_metrics(model,ground_truth,predictions):

    tn, fp, fn, tp = confusion_matrix(ground_truth,predictions).ravel()
    specificity = safe_divide(tn,tn+fp)
    precision,recall,fscore,support = precision_recall_fscore_support(ground_truth,predictions,average='binary')
    mcc = matthews_corrcoef(ground_truth,predictions)
    metrics = {'model' : model, 'F1' : fscore, 'recall' : recall,'specificity' : specificity ,'precision' : precision , 'MCC' : mcc}
    return metrics

def safe_divide(num,denom):

    if denom > 0:
        return float(num)/ denom
    else:
        return 0.0
if __name__ == "__main__":

    args = parse_args()
    ensemble_predict(args)
