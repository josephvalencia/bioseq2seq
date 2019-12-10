import re
import os
import sys
import shlex
import subprocess

import numpy as np
import pandas as pd

from Bio import Align
from Bio.Seq import Seq
from Bio.Emboss.Applications import NeedleCommandline

from collections import Counter

def modified_hamming(first,second):

    longer = first if len(first) >= len(second) else second
    shorter = second if longer == first else first
    return

def kmer_overlap_scores(query,reference,k):

    '''Calculates recall and precision of all words of length k.

       query: query sequence (string or BioSeq)
       reference: reference sequence (string or BioSeq)
       k : size of word (int)

       returns - recall,precision,f1 (tuple(float,float,float))
    '''

    q_kmer_list = [query[i:i+k] for i in range(len(query)-k+1)]
    r_kmer_list = [reference[i:i+k] for i in range(len(reference)-k+1)]

    q_counts = Counter(q_kmer_list)
    r_counts = Counter(r_kmer_list)

    tp = 0

    for key,val in r_counts.items():

        query_count = q_counts[key] if key in q_counts else 0
        count = min(val,query_count)
        tp += count

    try:
        recall = tp/(len(reference) - k+1)
        precision = tp/(len(query)-k+1)
        f1 = 2*(precision*recall)/(precision+recall)

        return recall,precision,f1

    except ZeroDivisionError:
        return 0.0,0.0,0.0

def emboss_needle(seqa, seqb):

    '''Calculate Needleman-Wunsch global alignment percentage identity using needle from EMBOSS package. See http://www.bioinformatics.nl/cgi-bin/emboss/help/needle
    '''

    pred_tmp = "pred.tmp"
    gold_tmp = "gold.tmp"
    out = "alignment.tmp"

    with open(pred_tmp,'w') as tempFile:
        tempFile.write(seqa)

    with open(gold_tmp,'w') as tempFile:
        tempFile.write(seqb)

        cmd = NeedleCommandline(asequence = "plain::"+pred_tmp, bsequence = "plain::"+gold_tmp,gapopen = 10, gapextend = 0.5,outfile = out,sprotein=True)
    try:
        stdout,stderrf = cmd()
    except Exception:
        pass

    response = open(out,'r').read()

    identity_pattern = "# Identity:(\s*)(\d*\/\d*)(.*)\n"
    identity_line = re.search(identity_pattern,response).group(2).split('/')

    try:
        os.remove(pred_tmp)
        os.remove(gold_tmp)
    except OSError:
        pass

    return float(identity_line[0]) / float(identity_line[1])

def emboss_getorf(nucleotide_seq,out):

    '''Find longest Open Reading Frame (START-STOP) using getorf from needle package. See http://bioinf-hpc.ibun.unal.edu.co/cgi-bin/emboss/help/getorf'''

    cmd = "getorf -sequence={} -outseq={} -find=1 -noreverse".format(nucleotide_seq,out)
    subprocess.call(shlex.split(cmd))

    response = open(out,'r').read()
    split_pattern = r'>_.*\n'

    orfs  = sorted(re.split(split_pattern,response),key = lambda x : len(x),reverse = True) # ORFs sorted by size descending
    return [x.replace("\\n",'') for x in orfs]

if __name__=="__main__":

    data = pd.read_csv("Fa/translations.map")

    id_list = data['ID'].tolist()
    gold = data['Protein'].tolist()

    predictions = [x.replace("\s","")for x in sys.stdin.readlines()]

    for idx,i,g,p in zip(range(len(id_list)),id_list,gold,predictions):

        print(idx)
        root = "/home/other/valejose/deep-translate/Fa/"

        gold_tmp = root+i+"tmp_gold.fa"
        pred_tmp = root+i +"tmp_pred.fa"
        out_tmp = root+i+"tmp.out"

        with open(pred_tmp,'w') as tempFile:
            tempFile.write(p)
        with open(gold_tmp,'w') as tempFile:
            tempFile.write(g)

        identity_score = emboss_needle(pred_tmp,gold_tmp,out_tmp)
        recall,precision,f1, = kmer_overlap_scores(g,i,3)
        print("Identity: {}\n3-mer Recall: {}\n3-mer Precision: {}\n3-mer F1: {}".format(identity_score,recall,precision,f1))

        os.remove(gold_tmp)
        os.remove(pred_tmp)
        os.remove(out_tmp)

        #translation = emboss_getorf("/home/other/valejose/deep-translate/temp3.fa","out.orf")
        #print("Predicted Protein: {}".format(translation))









