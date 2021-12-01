import numpy as np
import re

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

    def prob_from_beam_energies(self):
        
        numerator = 0
        denominator = 0
        for i,v in enumerate(self.votes):
            denominator += np.exp(float(v.pred_score))
            if v.pred_class == '<PC>':
                numerator += np.exp(float(v.pred_score))
        
        if denominator > 0 :
            composite_vote = numerator / denominator
        else:
            composite_vote = 0
        
        return composite_vote 

def prediction_from_record(record,model):
    
    transcript = record[0].split('ID: ')[1]
    src = record[1].split('RNA: ')[1]
    pc_score = np.exp(float(record[6].split('PC_SCORE: ')[1]))
    preds = record[2:6]

    container = Prediction(model,transcript,len(src),pc_score)
   
    for p in preds:
        pred_match  = re.search('PRED: (<PC>|<NC>)(\S*)?',p)
        score_match = re.search('SCORE: (\S*)',p)
        
        if score_match is not None:
            score = score_match.group(1)
            if pred_match is not None:
                pred_class = pred_match.group(1)
                pred_peptide = pred_match.group(2)
                vote = Vote(pred_class,pred_peptide,score)
                container.add(vote)
            else:
                pred_match  = re.search('PRED: (\S*)?',p)
                if pred_match is not None:
                    pred_peptide = pred_match.group(1)
                    if len(pred_peptide):
                        pred_class = '<PC>'
                    else:
                        pred_class = '<NC>'
                    vote = Vote(pred_class,pred_peptide,score)
                    container.add(vote)

    return container

