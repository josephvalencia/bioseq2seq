import sys,re,os
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import getLongestORF,parse_config, setup_fonts
from tqdm import tqdm

def parse(testfile,filename):

    gt = pd.read_csv(filename,sep='\t')
    pred = pd.read_csv(testfile,sep='\t')
    pred = pred.rename(columns={'tscript' : 'ID'})
    pred = pred.astype({'start':'int'})
    starts = [x.split(':')[0] for x in gt['CDS'].tolist()]
    lens = [len(x) for x in gt['RNA'].tolist()]

    cds_start = [] 
    for s,l in zip(starts,lens):
        if s != "-1":
            if s == "<0":
                cds_start.append(int(l)-1)
            else:
                cds_start.append(int(s))
        else:
            cds_start.append(int(l)-1)
    
    gt['true_start'] = cds_start

    combined = gt.merge(pred,on='ID')
    
    pc = combined[combined['Type'] == '<PC>'] 
    homology = pd.read_csv("test_maximal_homology.csv")
    pc = pc.merge(homology,on='ID') 
    reduced = pc['score'] <=80
    pc = pc.loc[reduced]
    correct = 0
    for pred,true in zip(pc['start'],pc['true_start']):
        if pred == true:
            correct +=1

    print(f'calculated from {len(pc)} mRNAs, of which {correct} ({100*correct/len(pc):.1f}%) predict the true CDS')

if __name__ == "__main__":

    parse(sys.argv[1],sys.argv[2])
