import sys,random
import json
import os,re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr,kendalltau,ttest_ind
from scipy.stats import entropy
from collections import defaultdict
from IPython.display import Image

from Bio.Data import CodonTable
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio import SeqIO

from bin.batcher import train_test_val_split
#from captum.attr import visualization as viz
from interpretation import visualizer as viz


def calculate_entropy(saved_file,df,tgt_field,mode):
    
    storage = []
    temp_idx = 0

    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"

            id = fields[id_field]
            array = fields[tgt_field]
            seq = df.loc[id,'RNA']
            tscript_type = df.loc[id,'Type']

            curr_entropy = entropy(array,base=2)
            uniform = np.log2(len(seq))

            information_gain =  uniform - curr_entropy
            storage.append(information_gain)

    print("head: {} , mean information gain {}".format(tgt_field,sum(storage)/len(storage)))

if __name__ == "__main__":

    plt.style.use('ggplot')
    
    # ingest stored data
    data_file = "../Fa/refseq_combined_cds.csv.gz"
    dataframe = pd.read_csv(data_file,sep="\t",compression = "gzip")
    df_train,df_test,df_val = train_test_val_split(dataframe,1000,65)
    df_val = df_val.set_index("ID")
   
    print("ED_classifier\n")
    for l in range(4):
        layer = "results/best_ED_classify/best_ED_classify_layer"+str(l)+".enc_dec_attns"
        for h in range(8):
            tgt_head = "layer{}head{}".format(l,h)
            calculate_entropy(layer,df_val,tgt_head,"attn")
   
    print("______________________________________\n seq2seq\n") 
    for l in range(4):
        layer = "results/best_seq2seq/best_seq2seq_layer"+str(l)+".enc_dec_attns"
        for h in range(8):
            tgt_head = "layer{}head{}".format(l,h)
            calculate_entropy(layer,df_val,tgt_head,"attn")

