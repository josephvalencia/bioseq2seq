import pyfiglet
import argparse
import os
import pandas as pd
import torch
import sys
import random

from bioseq2seq.inputters import TextDataReader,get_fields
from bioseq2seq.inputters.text_dataset import TextMultiField
from bioseq2seq.translate import Translator, GNMTGlobalScorer
from models import make_transformer_model, make_loss_function

from torch.optim import Adam
from bioseq2seq.utils.optimizers import Optimizer
from bioseq2seq.bio.batcher import train_test_val_split, filter_by_length

def start_message():

    width = os.get_terminal_size().columns
    bar = "-"*width+"\n"

    centered = lambda x : x.center(width)

    welcome = "BioSeq2Seq"
    welcome = pyfiglet.figlet_format(welcome)

    print(welcome+"\n")

    info = {"Author":"Joseph Valencia"\
            ,"Date": "11/08/2019",\
            "Version":"1.0.0",
            "License": "Apache 2.0"}

    for k,v in info.items():

        formatted = k+": "+v
        print(formatted)

    print("\n")

def parse_args():

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--verbose",action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--stats-output","--st",help = "Name of file for saving evaluation statistics")
    parser.add_argument("--translation-output","--o",help = "Name of file for saving predicted translations")
    parser.add_argument("--checkpoint", "--c",help="ONMT checkpoint")

    # translate optional args
    parser.add_argument("--decoding-strategy","--d",default = "greedy",choices = ["beam","greedy"])
    parser.add_argument("--beam_size","--b",type = int, default = 4)
    parser.add_argument("--alpha","--a",type = float, default = 1.0)

    return parser.parse_args()

def restore_model(checkpoint,machine):

    model = make_transformer_model()
    model.load_state_dict(checkpoint['model'],strict = False)
    model.generator.load_state_dict(checkpoint['generator'])
    model.to(device = machine)

    return model

def build_vocab(fields,src,tgt):

    src = TextMultiField('src',fields['src'],[])
    tgt = TextMultiField('tgt',fields['tgt'],[])

    text_fields = get_fields(src_data_type ='text', n_src_feats = 0, n_tgt_feats = 0, pad ="<pad>", eos ="<eos>", bos ="<sos>")
    text_fields['src'] = src
    text_fields['tgt'] = tgt

    return text_fields

def translate_from_checkpoint(args):

    #machine = torch.device('cuda:0')
    machine = "cpu"

    checkpoint = torch.load(args.checkpoint,map_location = machine)

    random_seed = 65
    random.seed(random_seed)
    state = random.getstate()

    data = pd.read_csv(args.input)
    train,test,dev = train_test_val_split(data,1000,random_seed) # replicate splits

    protein = [x for x in dev['Protein'].tolist()]
    rna = [x for x in dev['RNA'].tolist()]

    model = restore_model(checkpoint,machine)
    text_fields = build_vocab(checkpoint['vocab'],rna,protein)

    translate(args,model,text_fields,rna,protein)

def translate(args,model,text_fields,rna,protein):

    beam_scorer = GNMTGlobalScorer(alpha = args.alpha, beta = 0.0, length_penalty = "avg" , coverage_penalty = "none")

    out_file = open("translations.out",'w')

    MAX_LEN = 500

    translator = Translator(model,
                            gpu = -1,
                            src_reader = TextDataReader(),
                            tgt_reader = TextDataReader(),
                            fields = text_fields,
                            beam_size = args.beam_size,
                            n_best = 1,
                            global_scorer = beam_scorer,
                            out_file = out_file,
                            verbose = True,
                            max_length = MAX_LEN)

    scores, predictions = translator.translate(src = rna, tgt = protein,batch_size = 4)

    out_file.close()

if __name__ == "__main__":

    args = parse_args()
    start_message()
    translate_from_checkpoint(args)
