import re
import os
import sys
import shlex
import subprocess

import numpy as np
import pandas as pd

from collections import Counter, defaultdict

class Evaluator:

    def __init__(self, k=15, best_of = 1, full_align = False, exact_match = False):

        self.k = k
        self.best_of_n = best_of
        self.full_align = full_align
        self.exact_match = exact_match

    def calculate_stats(self,preds,golds,names=None,log_all =False):

        best_stats = defaultdict(int)
        best_n_stats = defaultdict(int)

        total_items = len(golds)

        divide = lambda n,d : float(n) / d if d > 0 else 0.0

        if self.best_of_n > 1: # report best of n and best of 1
            idx = 0
            for candidates, gold in zip(preds,golds):

                if self.full_align:

                    alignments= [self.emboss_needle("".join(c),gold) for c in candidates]
                    align_pcts = [divide(a,b) for a,b in alignments]
                    pct_align_best = align_pcts[0]
                    best_stats['avg_align_id'] += pct_align_best
                    best_n_loc = np.argmax(align_pcts)
                    pct_align_best_n = align_pcts[best_n_loc]
                    best_n_stats['avg_align_id'] += pct_align_best_n

                    if log_all and names is not None:
                        msg = "NAME: {}\nBEST: {} PCT: {}\nBEST_N: {} PCT {}\nGOLD: {}\n"
                        print(msg.format(names[idx],candidates[0],pct_align_best,candidates[best_n_loc],pct_align_best_n,gold))
                    elif log_all:
                        msg = "BEST: {} PCT: {}\nBEST_N: {} PCT {}\nGOLD: {}\n"
                        print(msg.format(candidates[0],pct_align_best,candidates[best_n_loc],pct_align_best_n,gold))


                if self.exact_match:

                    perfect_matches = [self.perfect_match(c,gold) for c in candidates]
                    best_stats['exact_match_rate'] += perfect_matches[0]
                    best_n_stats['exact_match_rate'] += max(perfect_matches)

                if self.k >1:

                    kmer_results = [self.kmer_overlap_scores(c,gold,self.k) for c in candidates]

                    tp,ref_len,query_len = kmer_results[0]
                    best_stats['avg_kmer_recall'] += divide(tp,ref_len)
                    best_stats['avg_kmer_precision'] += divide(tp,query_len)

                    tp,ref_len,query_len = max(kmer_results, key = lambda x: divide(x[0],x[1]))
                    best_n_stats['avg_kmer_recall'] += divide(tp,ref_len)

                    tp,ref_len,query_len = max(kmer_results, key = lambda x: divide(x[0],x[2]))
                    best_n_stats['avg_kmer_precision'] += divide(tp,query_len)

                idx+=1

            # average
            for k in best_stats.keys():
                best_stats[k] /= total_items

            for k in best_n_stats.keys():
                best_n_stats[k] /= total_items

            return best_stats, best_n_stats

        else: # report only best of 1

            for candidate,gold in zip(preds,golds):

                if self.full_align:
                    best_match,best_total = self.emboss_needle(candidate,gold)
                    best_stats['avg_align_id'] += divide(best_match,best_total)
                if self.perfect_match:
                    perfect_match = self.perfect_match(candidate,gold)
                    best_stats['exact_match_rate'] += perfect_match
                if self.k >1:
                    tp,ref_len,query_len = self.kmer_overlap_scores(candidate,gold,self.k)
                    best_stats['avg_kmer_recall'] += divide(tp,ref_len)
                    best_stats['avg_kmer_precision'] += divide(tp,query_len)
            # average
            for k in best_stats.keys():
                best_stats[k] /= total_items

            return (best_stats,)

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

        return tp, len(reference) - k+1, len(query) - k+1

    def emboss_needle(self,seqa,seqb):

        '''Calculate Needleman-Wunsch global alignment percentage identity using needle from EMBOSS package.
        See http://www.bioinformatics.nl/cgi-bin/emboss/help/needle '''

        cmd_format = "/home/bb/valejose/EMBOSS-6.6.0/emboss/needle -asequence {} -bsequence {} -gapopen {} -gapextend {} -sprotein -brief -stdout -auto"
        cmd = cmd_format.format("asis::"+seqa,"asis::"+seqb,10,0.5)

        response = subprocess.check_output(shlex.split(cmd),universal_newlines=True)

        identity_pattern = "# Identity:(\s*)(\d*\/\d*)(.*)\n"
        match, total = re.search(identity_pattern,response).group(2).split('/')

        return int(match), int(total)

    def emboss_getorf(self,nucleotide_seq):

        '''Find longest Open Reading Frame (START-STOP) using getorf from needle package.
        See http://bioinf-hpc.ibun.unal.edu.co/cgi-bin/emboss/help/getorf'''

        cmd = "getorf -sequence={} -find=1 -noreverse -stdout -auto".format(nucleotide_seq)
        response = subprocess.check_output(shlex.split(cmd),universal_newlines=True)
        print(response)
        split_pattern = r'>_.*\n'

        # ORFs sorted by size descending
        orfs = sorted(re.split(split_pattern,response),key = lambda x : len(x),reverse = True)

        return [x.replace("\\n",'') for x in orfs]
