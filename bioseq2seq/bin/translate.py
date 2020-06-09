#!/usr/bin/env python
import argparse
import pandas as pd
import torch
import random
import time

from bioseq2seq.inputters import TextDataReader,get_fields
from bioseq2seq.inputters.text_dataset import TextMultiField
from bioseq2seq.translate import Translator, GNMTGlobalScorer
from bioseq2seq.bin.models import make_transformer_model

from torchtext.data import RawField
from bioseq2seq.bin.batcher import train_test_val_split
from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition,train_test_val_split
from bioseq2seq.bin.batcher import train_test_val_split


def parse_args():
    """ Parse required and optional configuration arguments"""

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--verbose",action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--output_name","--o", default = "translation",help = "Name of file for saving predicted translations")
    parser.add_argument("--checkpoint", "--c",help="ONMT checkpoint (.pt)")
    parser.add_argument("--mode",default = "translate",help="translate|classify|combined")
    # translate optional args
    parser.add_argument("--beam_size","--b",type = int, default = 8, help ="Beam size for decoding")
    parser.add_argument("--n_best", type = int, default = 4, help = "Number of beams to wait for")
    parser.add_argument("--alpha","--a",type = float, default = 1.0)

    return parser.parse_args()

def restore_transformer_model(checkpoint,machine):
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''

    model = make_transformer_model()
    model.load_state_dict(checkpoint['model'],strict=False)
    model.generator.load_state_dict(checkpoint['generator'])
    model.to(device = machine)

    return model

def make_vocab(fields,src,tgt):
    """ Map torchtext.Vocab and raw source/target text as required by bioseq.translate.Translator
    Args:
        fields (torchtext.Vocab): to apply to data
        src (list(str)): input data
        tgt (list(str)): output data
    Returns:
        fields (dict)
    """

    src = TextMultiField('src',fields['src'],[])
    tgt = TextMultiField('tgt',fields['tgt'],[])

    text_fields = get_fields(src_data_type ='text',
                             n_src_feats = 0,
                             n_tgt_feats = 0,
                             pad ="<pad>",
                             eos ="<eos>",
                             bos ="<sos>")

    text_fields['src'] = src
    text_fields['tgt'] = tgt

    return text_fields

def arrange_data_by_mode(df, mode):

    if args.mode == "translate":
        # in translate-only mode, only protein coding are considered
        df = df[df['Type'] == "<PC>"]
        protein = df['Protein'].tolist()
    elif args.mode == "combined":
        # identify and translate coding, identify non coding
        protein = (df['Type'] + df['Protein']).tolist()
    elif args.mode == "classify":
        # regular binary classifiation coding/noncoding
        protein = df['Type'].tolist()

    ids = df['ID'].tolist() 
    rna = df['RNA'].tolist()
    cds = df['CDS'].tolist()
    
    return protein,ids,rna,cds

def translate_from_transformer_checkpt(args,device):

    random_seed = 65
    random.seed(random_seed)
    state = random.getstate()

    data = pd.read_csv(args.input,sep="\t")
    data["CDS"] = ["-1" for _ in range(data.shape[0])]

    # replicate splits
    train,test,dev = train_test_val_split(data,1000,random_seed)

    #train,test,val = dataset_from_df(df_train.copy(),df_test.copy(),df_val.copy(),mode=args.mode)
    protein,ids,rna,cds = arrange_data_by_mode(dev,args.mode)

    checkpoint = torch.load(args.checkpoint,map_location = device)
    saved_params = checkpoint['opt']

    vocab = checkpoint['vocab']
    print(vocab['tgt'].vocab.stoi)
    print(vocab['src'].vocab.stoi)

    if not saved_params is None:
        model_name = ""
        print("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
        for k,v in vars(saved_params).items():
            print(k,v)

    model = restore_transformer_model(checkpoint,device)
    text_fields = make_vocab(checkpoint['vocab'],rna,protein)

    translate(model,text_fields,rna,protein,ids,cds,device,beam_size=args.beam_size,n_best=args.n_best,save_preds=True,save_attn=True,file_prefix=args.output_name)

def translate(model,text_fields,rna,protein,ids,cds,device,beam_size = 8,
                    n_best = 4,save_preds=False,save_attn=False,file_prefix= "temp"):
    
    """ Translate raw data
    Args:
        model (bioseq2seq.translate.NMTModel): Encoder-Decoder + generator for translation
        text_fields (dict): returned by make_vocab()
        rna (list(str)): raw src (RNA) data
        protein (list(str)): raw tgt (Protein) data
        ids (list(str)): GENCODE ids
        args (argparse.Namespace | dict): config arguments
        device (torch.device | str): device for translation.
    """
    # global scorer for beam decoding
    beam_scorer = GNMTGlobalScorer(alpha = 1.0,
                                   beta = 0.0,
                                   length_penalty = "avg" ,
                                   coverage_penalty = "none")

    MAX_LEN = 500

    translator = Translator(model,
                            device = device,
                            src_reader = TextDataReader(),
                            tgt_reader = TextDataReader(),
                            file_prefix = file_prefix,
                            fields = text_fields,
                            beam_size = beam_size,
                            n_best = n_best,
                            global_scorer = beam_scorer,
                            verbose = True,
                            max_length = MAX_LEN)

    predictions, golds, scores = translator.translate(src = rna,
                                                      tgt = protein,
                                                      names = ids,
                                                      cds = cds,
                                                      batch_size = 4,
                                                      save_attn = save_attn,
                                                      save_preds = save_preds)
    
    return predictions,golds,scores

if __name__ == "__main__":

    args = parse_args()
    machine = "cuda:0"
    translate_from_transformer_checkpt(args,machine)