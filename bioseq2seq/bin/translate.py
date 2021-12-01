#!/usr/bin/env python
import argparse
import pandas as pd
import torch
import random
import time
import numpy as np

from bioseq2seq.inputters import TextDataReader,get_fields
from bioseq2seq.inputters.text_dataset import TextMultiField
from bioseq2seq.translate import Translator, GNMTGlobalScorer
from bioseq2seq.bin.models import make_transformer_seq2seq, Generator
from bioseq2seq.modules.embeddings import PositionalEncoding

from torchtext.data import RawField
from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset


def parse_args():
    """ Parse required and optional configuration arguments"""

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--save_SA", action="store_true")
    parser.add_argument("--save_EDA", action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--output_name","--o", default = "translation",help = "Name of file for saving predicted translations")
    parser.add_argument("--checkpoint", "--c",help="ONMT checkpoint (.pt)")
    parser.add_argument("--mode",default = "translate",help="translate|classify|combined")
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--num_gpus",type=int,default=1)

    # translate optional args
    parser.add_argument("--beam_size","--b",type = int, default = 8, help ="Beam size for decoding")
    parser.add_argument("--n_best", type = int, default = 4, help = "Number of beams to wait for")
    parser.add_argument("--alpha","--a",type = float, default = 1.0)
    parser.add_argument("--attn_save_layer", type = int,default=0,help="If --save_attn flag is used, which layer of EDA to save")
    return parser.parse_args()

def restore_transformer_model(checkpoint,machine,opts):
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''

    vocab = checkpoint['vocab'] 
    print(vocab['tgt'].vocab.stoi)
    #n_input_classes = len(vocab['src'].vocab.stoi)
    #n_output_classes = len(vocab['tgt'].vocab.stoi)
    n_input_classes = 19
    n_output_classes = 29
    
    model = make_transformer_seq2seq(n_input_classes,n_output_classes,n_enc=opts.n_enc_layers,n_dec=opts.n_dec_layers,model_dim=opts.model_dim,max_rel_pos=opts.max_rel_pos)
    #model = make_transformer_seq2seq(n_enc=4,n_dec=4,model_dim=128,max_rel_pos=10)
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

    if mode == "translate":
        # in translate-only mode, only protein coding are considered
        df = df[df['Type'] == "<PC>"]
        protein = df['Protein'].tolist()
    elif mode == "bioseq2seq":
        # identify and translate coding, identify non coding
        protein = (df['Type'] + df['Protein']).tolist()
    elif mode == "ED_classify":
        # regular binary classifiation coding/noncoding
        protein = df['Type'].tolist()

    ids = df['ID'].tolist() 
    rna = df['RNA'].tolist()
    cds = df['CDS'].tolist()
    return protein,ids,rna,cds

def exclude_transcripts(data):

    # hack to process seqs that failed on GPU
    failed = pd.read_csv('../Fa/mammalian_1k_to_2k_RNA_reduced_80_ids.txt',sep='\n',names=['ID'])
    failed = failed.set_index("ID")
    data = data.set_index("ID")
    #data = data.drop(labels=data.index.difference(failed.index))
    data = data.drop(labels=failed.index)
    data = data.reset_index()
    print(data)
    return data

def partition(df,split_ratios,random_seed):

    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    N = df.shape[0]

    # splits to cumulative percentages
    cumulative = [split_ratios[0]]
    for i in range(1,len(split_ratios) -1):
        cumulative.append(cumulative[i-1]+split_ratios[i])
    split_points = [int(round(x*N)) for x in cumulative]

    # split dataframe at split points
    return np.split(df,split_points)

def run_helper(rank,model,vocab,args):

    random_seed = 65
    random.seed(random_seed)
    state = random.getstate()
    file_prefix = args.output_name
    
    device = "cuda:{}".format(rank)
    #device = 'cpu'

    data = pd.read_csv(args.input,sep="\t")
    #data = exclude_transcripts(data)
    data["CDS"] = ["-1" for _ in range(data.shape[0])]
    
    if args.num_gpus > 1:
        file_prefix += '.rank_{}'.format(rank)

    if args.num_gpus > 0:
        # One CUDA device per process
        torch.cuda.set_device(device)
        model.cuda()
        split_ratios = [1.0/args.num_gpus] * args.num_gpus
        df_partitions = partition(data,split_ratios,random_seed)
        data = df_partitions[rank]

    protein,ids,rna,cds = arrange_data_by_mode(data,args.mode)
    text_fields = make_vocab(vocab,rna,protein)
    
    translate(model,
            text_fields,
            rna,
            protein,
            ids,
            cds,
            device,
            args.alpha,
            beam_size=args.beam_size,
            n_best=args.n_best,
            save_preds=True,
            save_SA=args.save_SA,
            save_EDA=args.save_EDA,
            attn_save_layer=args.attn_save_layer,
            file_prefix=file_prefix)

def translate(model,text_fields,rna,protein,ids,cds,device,alpha,beam_size = 8,
                    n_best = 4,save_preds=False,save_SA=False,save_EDA=False,attn_save_layer=3,file_prefix= "temp"):
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
    
    MAX_LEN = 333
    BATCH_SIZE = 1
    
    # global scorer for beam decoding
    beam_scorer = GNMTGlobalScorer(alpha = 0.6,
                                   beta = 0.0,
                                   length_penalty = "wu",
                                   coverage_penalty = "none")
    
    translator = Translator(model,
                            device = device,
                            src_reader = TextDataReader(),
                            tgt_reader = TextDataReader(),
                            file_prefix = file_prefix,
                            fields = text_fields,
                            beam_size = beam_size,
                            n_best = n_best,
                            global_scorer = beam_scorer,
                            tgt_prefix = None,
                            verbose = False,
                            attn_save_layer=attn_save_layer,
                            max_length = MAX_LEN)
    
    translator.translate(src = rna,
                          tgt = protein,
                          names = ids,
                          cds = cds,
                          batch_size = BATCH_SIZE,
                          save_SA = save_SA,
                          save_EDA= save_EDA,
                          save_preds = save_preds,
                          save_scores = False)

def translate_from_transformer_checkpt(args):

    device = 'cpu'
    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
    vocab = checkpoint['vocab']

    if not options is None:
        print(vocab['tgt'].vocab.stoi)
        print(vocab['src'].vocab.stoi)
        model_name = ""
        print("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
        for k,v in vars(options).items():
            print(k,v)
    
    model = restore_transformer_model(checkpoint,device,options)
    model.eval()

    vocab = checkpoint['vocab']
    if args.num_gpus > 1:
        print('Translating on {} GPUs'.format(args.num_gpus))
        torch.multiprocessing.spawn(run_helper, nprocs=args.num_gpus, args=(model,vocab,args))
    elif args.num_gpus > 0:
        run_helper(args.rank,model,vocab,args)
    else:
        run_helper(args.rank,model,vocab,args)

if __name__ == "__main__":

    args = parse_args()
    translate_from_transformer_checkpt(args)
