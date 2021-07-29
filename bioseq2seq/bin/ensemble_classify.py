#!/usr/bin/env python
import argparse
import re
import numpy as np
from math import ceil
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score,precision_recall_fscore_support,pairwise_distances,confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    """ Parse required and optional configuration arguments"""

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--verbose",action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--output_name","--o", default = "translation",help = "Name of file for saving predicted translations")
    parser.add_argument("--models_file", "--m",help="File of paths to models")
    return parser.parse_args()

class Vote:
    def __init__(self,pred_class,pred_seq,pred_score):
        self.pred_class = pred_class
        self.pred_seq = pred_seq
        self.pred_score = pred_score
    
    def __str__(self):
        if self.pred_class == '<PC>':
            excerpt = self.pred_seq[:20]+'...' if len(self.pred_seq) > 20 else self.pred_seq
        else:
            excerpt = None

        return 'Vote({},{},{})'.format(self.pred_class,excerpt,self.pred_score)

class Prediction:
    def __init__(self,model_name,transcript,transcript_len,pc_prob):
        self.transcript = transcript
        self.transcript_len = transcript_len
        self.model_name = model_name
        self.pc_prob = pc_prob 
        self.votes = []

    def add(self,prediction):
        self.votes.append(prediction)

def ensemble_predict(args):

    with open(args.models_file) as inFile:
        model_paths = [x.rstrip() for x in inFile.readlines()]

    # classification functions
    is_coding_gt = lambda x : 1 if x.startswith('NM') or x.startswith('XM') else 0
    is_coding_pred = lambda x : 1 if x == '<PC>' else 0 
    is_coding_thresh = lambda x,t : 1 if x > t else 0 
    is_coding_majority = lambda x,y : 1 if x >= ceil(0.5*y) else 0

    prior = {}
    lengths = {}
    gt = []
    consensus_probs = defaultdict(list)
    ensemble_storage = defaultdict(lambda : defaultdict(float))
    save_gt = True
    gt_count = 0 
    for f in model_paths: 
        best_preds = []
        with open(f) as inFile:
            lines = [x.rstrip() for x in inFile.readlines()]
            for i in range(0,len(lines),9):
                pred = parse_predictions(lines[i:i+9],f)
                ensemble_storage[pred.transcript][f] = pred
                composite_vote = prob_from_beam_energies(pred) 
                consensus_probs[f].append(composite_vote)
                best_preds.append(is_coding_pred(pred.votes[0].pred_class))
                if save_gt:
                    gt.append(is_coding_gt(pred.transcript))
                    gt_count+=1
                if pred.transcript not in lengths:
                    lengths[pred.transcript] = pred.transcript_len
            # after first pass, no need to save gold
            save_gt = False
        precision,recall,f1,support = precision_recall_fscore_support(gt,best_preds,average='binary')
        print(f,precision,recall,f1)

    '''
    # test different thresholds
    storage = []
    for f in model_paths:
        for thresh in [0.01*x for x in range(1,100)]:
        #for thresh in [0.05*x for x in range(1,20)]:
            preds = [is_coding_thresh(x,thresh) for x in consensus_probs[f]]
            precision,recall,f1,support = precision_recall_fscore_support(gt,preds,average='binary')
            entry = {'model' : f , 'threshold' : thresh , 'precision' : precision , 'recall' : recall , 'F1' : f1}
            prior[f] = 1
            storage.append(entry)
    df = pd.DataFrame(storage)
    maxes = df.loc[df.groupby(['model'])['F1'].idxmax()]
    print(maxes)
    quit() 
    df.to_csv('threshold_trials.csv',index=False) 
    sns.lineplot(data=df,x='threshold',y='F1',hue='model')
    plt.savefig('threshold_trials.svg')
    '''

    for f in model_paths:
        prior[f] = 1
    total_weight = sum(list(prior.values()))
    
    gt = []
    consensus_preds = []
    pc_votes = []
    nc_votes = []
    
    # no length penalty 
    decision_vals = [0.88,0.86,0.78,0.52,0.52]
    #decision_vals = [0.85,0.85,0.75,0.55,0.50]
    
    # wu norm
    #decision_vals = [0.77,0.82,0.74,0.59,0.66]
    #decision_vals = [0.80,0.80,0.75,0.60,0.65]
    
    for tscript, entries in ensemble_storage.items():
        true_class = is_coding_gt(tscript)
        gt.append(true_class)
        #Find weighted average
        consensus_prob = 0
        votes = []
        for model,prediction in entries.items():
            composite_vote = prob_from_beam_energies(prediction) 
            votes.append(composite_vote)
            # count classes on conditional on most probable beam 
            first_guess = prediction.votes[0]
            if first_guess == '<PC>':
                num_pc = sum([is_coding_pred(x.pred_class) for x in prediction.votes])
                pc_votes.append(num_pc)
            elif first_guess == '<NC>':
                num_pc = sum([is_coding_pred(x.pred_class) for x in prediction.votes])
                nc_votes.append(num_pc)
            # PC prob based on initial PC weight or energy-based score ?
            #consensus_prob+= prediction.pc_prob*prior[model]
            consensus_prob += composite_vote*prior[model]
        
        consensus_prob /= total_weight
        #vote_tally = sum([is_coding_thresh(x,t) for x,t in zip(votes,decision_vals)])
        vote_tally = sum([is_coding_thresh(x,0.6) for x,t in zip(votes,decision_vals)])
        
        #consensus_class = is_coding_thresh(consensus_prob,0.50) 
        consensus_class = is_coding_majority(vote_tally,len(votes))
        
        consensus_preds.append(consensus_class)
        
        if consensus_class != true_class:
            if consensus_class == 0 and true_class == 1:
                error_type = 'FN'
            else:
                error_type = 'FP'
            diff = vote_tally if error_type == 'FN' else 3-vote_tally
            '''
            if error_type =='FN' :
                print(f'Transcript = {tscript} len = {lengths[tscript]} ({error_type}), Pr(consensus) = {consensus_prob}')
                print(f'Votes = {votes}, Good votes = {diff}')
                models_tmp = ensemble_storage[tscript]
                for m,p in models_tmp.items():
                    print(p.pc_prob)
                    for v in p.votes:
                        print(v.pred_class,np.exp(float(v.pred_score)))
            ''' 
    
    #print(f'PC = {np.nanmean(pc_votes)}+-{np.nanstd(pc_votes)} , NC = {np.nanmean(nc_votes)}+-{np.nanstd(nc_votes)}') 
    multi_results = precision_recall_fscore_support(gt,consensus_preds,average='binary')
    tn, fp, fn, tp = confusion_matrix(gt,consensus_preds).ravel()
    print(f'TN={tn} FP={fp}\nFN={fn} TP={tp}')
    print('Consensus results',multi_results)

def prob_from_beam_energies(prediction):
    numerator = 0
    denominator = 0
    for i,v in enumerate(prediction.votes):
        denominator += np.exp(float(v.pred_score))
        if v.pred_class == '<PC>':
            numerator += np.exp(float(v.pred_score))
    composite_vote = numerator / denominator
    return composite_vote 

def parse_predictions(record,model):
    
    transcript = record[0].split('ID: ')[1]
    src = record[1].split('RNA: ')[1]
    pc_score = np.exp(float(record[6].split('PC_SCORE: ')[1]))
    preds = record[2:6]

    container = Prediction(model,transcript,len(src),pc_score)
    for p in preds:
        pred_match  = re.search('PRED: (<PC>|<NC>)(\S*)?',p)
        score_match = re.search('SCORE: (\S*)',p)
        if score_match is not None and pred_match is not None:
            pred_class = pred_match.group(1)
            pred_peptide = pred_match.group(2)
            score = score_match.group(1)
            vote = Vote(pred_class,pred_peptide,score)
            container.add(vote)
    return container

if __name__ == "__main__":

    args = parse_args()
    ensemble_predict(args)
