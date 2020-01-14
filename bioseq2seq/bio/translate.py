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
from models import EncoderDecoder, make_transformer_model, make_loss_function

from torch.optim import Adam
from bioseq2seq.utils.optimizers import Optimizer

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
    parser.add_argument("--beam","--b",type = int, default = 4)
    parser.add_argument("--alpha","--a",type=float,default = 0.5)

    return parser.parse_args()

def define_vocab():
    pass

def translate(args):

    random.seed(65)
    state = random.getstate()

    data = pd.read_csv(args.input,index_col = 0)
    train,test,dev = dataset_from_csv(data,1000,random_state) # obtain splits

    #data = data[data.Protein.str.len() < 1000]

    protein = [x[:5000] for x in data['Protein'].tolist()]
    rna = [x[:5000] for x in data['RNA'].tolist()]

    src_reader = TextDataReader()
    tgt_reader = TextDataReader()

    model = make_transformer_model()

    machine = torch.device('cuda:0')

    checkpoint = torch.load(args.checkpoint,map_location = machine)

    model.load_state_dict(checkpoint['model'],strict = False)
    model.generator.load_state_dict(checkpoint['generator'])
    model.to(device = machine)

    total_params = sum(p.numel() for p in model.parameters())
    print("TOTAL # PARAMS: {} ".format(total_params))

    loss_computer = make_loss_function(device = machine, generator = model.generator)

    adam = Adam(model.parameters())
    optim = Optimizer(adam, learning_rate = 1e-3)

    optim.load_state_dict(checkpoint['optim'])

    fields = checkpoint['vocab']

    src = TextMultiField('src',fields['src'],[])
    tgt = TextMultiField('tgt',fields['tgt'],[])

    google_scorer = GNMTGlobalScorer(alpha = args.alpha, beta = 0.0, length_penalty = "avg" , coverage_penalty = "none")

    text_fields = get_fields(src_data_type ='text', n_src_feats = 0, n_tgt_feats = 0, pad ="<pad>", eos ="<eos>", bos ="<sos>")
    text_fields['src'] = src
    text_fields['tgt'] = tgt

    out_file = open("translations.out",'w')

    max_len = max([len(x) for x in protein])+100

    print("MAX_LEN: "+str(max_len))

    translator = Translator(model,gpu = 0,src_reader = src_reader,tgt_reader = tgt_reader,\
                            fields = text_fields, beam_size = 8, n_best = 6,\
                            global_scorer = google_scorer,out_file = out_file,verbose = True,max_length = max_len)

    scores, predictions = translator.translate(src = rna, tgt = protein,batch_size = 4)
    out_file.close()

if __name__ == "__main__":

    args = parse_args()
    start_message()
    translate(args)
