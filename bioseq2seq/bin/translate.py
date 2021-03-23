#!/usr/bin/env python
import argparse
import pandas as pd
import torch
import random
import time

from bioseq2seq.inputters import TextDataReader,get_fields
from bioseq2seq.inputters.text_dataset import TextMultiField
from bioseq2seq.translate import Translator, GNMTGlobalScorer
from bioseq2seq.bin.models import make_transformer_seq2seq
from bioseq2seq.modules.embeddings import PositionalEncoding

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
    parser.add_argument("--dataset",default="validation",help="train|validation|test")
    parser.add_argument("--rank",default=0)
    
    # translate optional args
    parser.add_argument("--beam_size","--b",type = int, default = 8, help ="Beam size for decoding")
    parser.add_argument("--n_best", type = int, default = 4, help = "Number of beams to wait for")
    parser.add_argument("--alpha","--a",type = float, default = 1.0)
    parser.add_argument("--attn_save_layer", type = int,default=0)
    
    return parser.parse_args()

def restore_transformer_model(checkpoint,machine,opts):
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''

    #model = make_transformer_model(n_enc=opts.n_enc_layers,n_dec=opts.n_dec_layers,model_dim=opts.model_dim,max_rel_pos=opts.max_rel_pos)
    model = make_transformer_seq2seq(n_enc=4,n_dec=4,model_dim=128,max_rel_pos=10)
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
    elif args.mode == "ED_classify":
        # regular binary classifiation coding/noncoding
        protein = df['Type'].tolist()

    ids = df['ID'].tolist() 
    rna = df['RNA'].tolist()
    cds = df['CDS'].tolist()
    
    return protein,ids,rna,cds

def translate_from_transformer_checkpt(args,use_splits=False):

    random_seed = 65
    random.seed(random_seed)
    state = random.getstate()

    data = pd.read_csv(args.input,sep="\t")
    data["CDS"] = ["-1" for _ in range(data.shape[0])]

    # replicate splits
    if use_splits:
        train,test,dev = train_test_val_split(data,2000,random_seed,min_len=1000,splits=[0.0,0.9454,0.0546])
        #train,test,dev = train_test_val_split(data,1000,random_seed,splits=[0.0,0.0,1.00])
        total = len(train)+len(test)+len(dev)
        print(total,len(dev))
        if args.dataset == "validation":
            protein,ids,rna,cds = arrange_data_by_mode(dev,args.mode)
        elif args.dataset == "test":
            protein,ids,rna,cds = arrange_data_by_mode(test,args.mode)
        elif args.dataset == "train":
            protein,ids,rna,cds = arrange_data_by_mode(train,args.mode)
    else:
        protein,ids,rna,cds = arrange_data_by_mode(data,args.mode)

    device = "cuda:{}".format(args.rank)
    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
    vocab = checkpoint['vocab']
    print(vocab['tgt'].vocab.stoi)
    print(vocab['src'].vocab.stoi)

    if not options is None:
        model_name = ""
        print("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
        for k,v in vars(options).items():
            print(k,v)
 
    model = restore_transformer_model(checkpoint,device,options)
    model.eval()

    text_fields = make_vocab(checkpoint['vocab'],rna,protein)
    translate(model,text_fields,rna,protein,ids,cds,device,beam_size=args.beam_size,
            n_best=args.n_best,save_preds=True,save_attn=False,
            attn_save_layer=args.attn_save_layer,file_prefix=args.output_name)

def translate(model,text_fields,rna,protein,ids,cds,device,beam_size = 8,
                    n_best = 4,save_preds=False,save_attn=False,attn_save_layer=3,file_prefix= "temp"):
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
                                   length_penalty = "avg",
                                   coverage_penalty = "none")

    MAX_LEN = 666

    # hack to expand positional encoding
    '''
    new_pe  = PositionalEncoding(0.0,128,max_len=MAX_LEN)
    new_embedding = torch.nn.Sequential(*list(model.encoder.embeddings.make_embedding.children())[:-1])
    new_embedding.add_module('pe',new_pe)
    new_embedding.to(device)
    model.encoder.embeddings.make_embedding = new_embedding
    '''
    translator = Translator(model,
                            device = device,
                            src_reader = TextDataReader(),
                            tgt_reader = TextDataReader(),
                            file_prefix = file_prefix,
                            fields = text_fields,
                            beam_size = beam_size,
                            n_best = n_best,
                            global_scorer = beam_scorer,
                            verbose = False,
                            attn_save_layer=attn_save_layer,
                            max_length = MAX_LEN)

    predictions, golds, scores = translator.translate(src = rna,
                                                      tgt = protein,
                                                      names = ids,
                                                      cds = cds,
                                                      batch_size = 1,
                                                      save_attn = save_attn,
                                                      save_preds = save_preds,
                                                      save_scores = True)
    return predictions,golds,scores

if __name__ == "__main__":

    args = parse_args()
    translate_from_transformer_checkpt(args,use_splits=True)
