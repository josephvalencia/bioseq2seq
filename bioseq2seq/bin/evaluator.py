import re
import shlex
import subprocess
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

class Evaluator:

    def __init__(self, mode = "combined", k=8, best_of=1,exact_match=True, full_align=False):
        
        self.mode = mode
        self.k = k
        self.full_align = full_align
        self.exact_match = exact_match
        
        self.best_of_n = best_of
        self.best_stats = defaultdict(lambda : [0.0])
        self.best_n_stats  = defaultdict(float) if best_of > 1 else None

    def calculate_stats(self,preds,golds,names,log_all=False):

        pred_labels,preds = self.separate_preds(preds)
        true_labels,golds = self.separate_gold(golds)
  
        total_items = len(golds)
        pc_count = 0
        
        if self.mode == "translate":
            for name,candidates,gold in zip(names,preds,golds):
                if self.full_align:
                    self.update_alignment(name,candidates,gold)
                if self.exact_match:
                    self.update_exact_match(name,candidates,gold)
                if self.k > 1:
                    self.update_kmer_stats(name,candidates,gold)

        elif self.mode == "combined":
            for name,truth,candidates,gold in zip(names,true_labels,preds,golds):
                if truth == 1:
                    if self.full_align:
                        self.update_alignment(name,candidates,gold)
                    if self.exact_match:
                        self.update_exact_match(name,candidates,gold)
                    if self.k > 1:
                        self.update_kmer_stats(name,candidates,gold)
                    pc_count +=1
                    

        if self.exact_match:
            self.best_stats['exact_match_rate'] = np.mean(self.best_stats['exact_match_rate']) 

        if self.mode == "classify" or self.mode == "combined":
            self.calculate_F1(true_labels,pred_labels)
            self.calculate_classification_metrics(true_labels,pred_labels)
            
        return self.best_stats,self.best_n_stats

    def calculate_F1(self,true_labels,pred_labels):
        
        best_preds = [x[0] for x in pred_labels]
        f1 = f1_score(true_labels,best_preds) 
        self.best_stats['F1'] = f1
    
    def calculate_classification_metrics(self,true_labels,pred_labels):

        best_preds = [x[0] for x in pred_labels]
        tn, fp, fn, tp = confusion_matrix(true_labels,best_preds).ravel()
        self.best_stats['recall'] = self.divide(tp,tp+fn)
        self.best_stats['specificity'] = self.divide(tn,tn+fp)
        self.best_stats['precision'] = self.divide(tp,tp+fp)

    def divide(self,num,denom):

        if denom > 0:
            return float(num)/ denom
        else:
            return 0.0

    def update_alignment(self,name,candidates,gold):
      
        align = lambda a,b : self.emboss_needle(a,b) if a != "" and b != "" else (0.0,1)
        alignments = [align("".join(c),gold) for c in candidates[:self.best_of_n]]
        align_pcts = [self.divide(a,b) for a,b in alignments]
        print("{}\t{}".format(name,align_pcts[0]))
        self.update_helper("needle_align_id",align_pcts,name=name)

    def update_exact_match(self,name,candidates,gold):

        perfect_matches = [self.perfect_match(c,gold) for c in candidates[:self.best_of_n]]
        self.update_helper("exact_match_rate",perfect_matches)

    def update_kmer_stats(self,name,candidates,gold):
        
        kmer_stats = lambda a,b: self.kmer_overlap_scores(a,b,self.k) if a != "" and b != "" else (0.0,1,1) 
        kmer_results = [kmer_stats(c,gold) for c in candidates[:self.best_of_n]]

        recalls = [self.divide(tp,ref_len) for tp,ref_len,query_len in kmer_results]
        precisions = [self.divide(tp,query_len) for tp,ref_len,query_len in kmer_results]

        self.update_helper("kmer_recall",recalls) # log=True,name=name,candidates=candidates,gold=gold
        self.update_helper("kmer_precision",precisions)
        
    def update_helper(self,metric,scores, log = False,name=None,candidates=None,gold=None):

        self.best_stats[metric].append(scores[0])

        if self.best_n_stats is None:
            if log and name is not None and candidates is not None and gold is not None:
                msg = "NAME: {} \nBEST: {} \n SCORE({}) : {} \nGOLD: {}\n"
                print(msg.format(name, candidates[0],metric,scores[0],gold))
        else:
            max_loc = np.argmax(scores) 
            max_score = scores[max_loc]
            self.best_n_stats[metrix].append(max_score)

            if log and name is not None and candidates is not None and gold is not None:
                msg = "NAME: {}\nBEST: {} SCORE({}): {}\nBEST_N: {} SCORE({}) {}\nGOLD: {}\n"
                print(msg.format(name,candidates[0],metric,scores[0],candidates[max_loc],metric,scores[max_loc],gold))
 
    def __get_int_label__(self,label):
        """ Convert text based classed label to integer label"""
        
        if label == "<NC>":
            return 0
        elif label == "<PC>":
            return 1
        else:
            return 0

    def __decouple__(self,combined):

        splits = re.match("(<\w*>)(\w*)",combined)

        if not splits is None:
            label = self.__get_int_label__(splits.group(1))
            protein = splits.group(2)
        else:
            label = 0
            protein = combined

        return label,protein

    def separate_gold(self,original):

        labels = []
        proteins = []

        for x in original:
            label,protein = self.__decouple__(x)
            labels.append(label)
            proteins.append(protein)

        return labels,proteins

    def separate_preds(self,original):

        labels = []
        proteins = []

        for best_n in original:
            n_labels = []
            n_proteins = []
            
            for x in best_n:
                label,protein = self.__decouple__(x)
                n_labels.append(label)
                n_proteins.append(protein)

            labels.append(n_labels)
            proteins.append(n_proteins)

        return labels,proteins

    def perfect_match(self,pred,gold):
        match =  1 if pred == gold else 0
        return match

    def kmer_overlap_scores(self,query,reference,k):
        '''Calculates recall and precision of all words of length k.
        query: query sequence (string or BioSeq)
        reference: reference sequence (string or BioSeq)
        k : size of word (int)
        returns - recall,precision,f1 (tuple(float,float,float))'''

        q_kmer_list = [query[i:i+k] for i in range(len(query)-k+1)]
        r_kmer_list = [reference[i:i+k] for i in range(len(reference)-k+1)]

        q_counts = Counter(q_kmer_list)
        r_counts = Counter(r_kmer_list)
        tp = 0

        for key,val in r_counts.items():
            query_count = q_counts[key] if key in q_counts else 0
            count = min(val,query_count)
            tp += count

        return tp, len(reference)-k+1, len(query)-k+1

    def emboss_needle(self,seqa,seqb):
        '''Calculate Needleman-Wunsch global alignment percentage identity using needle from EMBOSS package.
        See http://www.bioinformatics.nl/cgi-bin/emboss/help/needle '''

        disallowed = ['','?','<unk>']

        if seqa not in disallowed and seqb not in disallowed:
            #cmd_format = "/home/bb/valejose/EMBOSS-6.6.0/emboss/needle -asequence {} -bsequence {} -gapopen {} -gapextend {} -sprotein -brief -stdout -auto"
            cmd_format = "needle -asequence {} -bsequence {} -gapopen {} -gapextend {} -sprotein -brief -stdout -auto"
            cmd = cmd_format.format("asis::"+seqa,"asis::"+seqb,10,0.5)

            response = subprocess.check_output(shlex.split(cmd),universal_newlines=True)
            identity_pattern = "# Identity:(\s*)(\d*\/\d*)(.*)\n"
            match, total = re.search(identity_pattern,response).group(2).split('/')
            return int(match), int(total)
        else:
            return 0, 1.0

    def emboss_getorf(self,nucleotide_seq):
        '''Find longest Open Reading Frame (START-STOP) using getorf from needle package.
        See http://bioinf-hpc.ibun.unal.edu.co/cgi-bin/emboss/help/getorf'''

        cmd = "getorf -sequence={} -find=1 -noreverse -stdout -auto".format(nucleotide_seq)
        response = subprocess.check_output(shlex.split(cmd),universal_newlines=True)
        split_pattern = r'>_.*\n'

        # ORFs sorted by size descending
        orfs = sorted(re.split(split_pattern,response),key = lambda x : len(x),reverse = True)
        return [x.replace("\\n",'') for x in orfs]
