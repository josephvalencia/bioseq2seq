import os
import argparse
import pandas as pd
import torch
import random
import time
import numpy as np
from math import log, floor
from itertools import count, zip_longest

from bioseq2seq.inputters import TextDataReader,get_fields, DynamicDataset, str2reader,str2sortkey
from bioseq2seq.inputters.text_dataset import TextMultiField
from bioseq2seq.translate import Translator, GNMTGlobalScorer, TranslationBuilder
from bioseq2seq.bin.models import make_transformer_seq2seq, make_hybrid_seq2seq, Generator
from bioseq2seq.utils.logging import init_logger, logger
from bioseq2seq.bin.data_utils import iterator_from_fasta
from bioseq2seq.inputters.corpus import maybe_fastafile_open
from torchtext.data import RawField
from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset
from bioseq2seq.bin.data_utils import AttachClassLabel
from bioseq2seq.transforms.transform import Transform

class NoOp(Transform):

    def apply(self, example, is_train=False, stats=None, **kwargs):
        return example

def parse_args():
    """ Parse required and optional configuration arguments"""

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--save_SA", action="store_true")
    parser.add_argument("--save_EDA", action="store_true")

    # translate required args
    parser.add_argument("--input",help="FASTA file for translation")
    parser.add_argument("--output_name","--o", default = "translation",help = "Name of file for saving predicted translations")
    parser.add_argument("--checkpoint", "--c",help="Model checkpoint (.pt)")
    parser.add_argument("--max_tokens",type = int , default = 9000, help = "Max number of tokens in training batch")
    parser.add_argument("--mode",default = "bioseq2seq",help="bioseq2seq|EDC")
    parser.add_argument("--decode_strategy",default = "beamsearch",help="beamsearch|greedy")
    parser.add_argument("--rank",type=int,default = 0)
    parser.add_argument("--num_gpus",type=int,default = 1)

    # translate optional args
    parser.add_argument("--beam_size","--b",type = int, default = 8, help ="Beam size for decoding")
    parser.add_argument("--n_best", type = int, default = 4, help = "Number of beams to wait for")
    parser.add_argument("--max_decode_len", type = int, default = 400, help = "Maximum length of protein decoding")
    parser.add_argument("--attn_save_layer", type = int,default=0,help="If --save_attn flag is used, which layer of EDA to save")
    return parser.parse_args()

def restore_transformer_model(checkpoint,machine,opts):
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''
    
    print(opts)
    vocab_fields = checkpoint['vocab'] 
    
    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
    
    n_input_classes = len(src_vocab.stoi)
    n_output_classes = len(tgt_vocab.stoi)
    print(f'n_enc = {opts.n_enc_layers} ,n_dec={opts.n_dec_layers}, n_output_classes= {n_output_classes} ,n_input_classes ={n_input_classes}')
    n_output_classes = 28 
    print('WINDOW',opts.window_size) 
    #model = make_hybrid_seq2seq(n_input_classes,n_output_classes,n_enc=opts.n_enc_layers,n_dec=opts.n_dec_layers,\
    #                            model_dim=opts.model_dim,dim_filter=opts.filter_size,window_size=opts.window_size,dropout=opts.dropout)
    
    model = make_hybrid_seq2seq(n_input_classes,
                                    n_output_classes,
                                    n_enc=opts.n_enc_layers,
                                    n_dec=opts.n_dec_layers,
                                    fourier_type=opts.model_type,
                                    model_dim=opts.model_dim,
                                    max_rel_pos=opts.max_rel_pos,
                                    dim_filter=opts.filter_size,
                                    window_size=opts.window_size,
                                    lambd_L1=opts.lambd_L1,
                                    dropout=opts.dropout)
    #model = make_transformer_seq2seq(n_input_classes,n_output_classes,n_enc=opts.n_enc_layers,n_dec=opts.n_dec_layers,\
    #        model_dim=opts.model_dim,max_rel_pos=opts.max_rel_pos)

    model.load_state_dict(checkpoint['model'],strict=False)
    model.generator.load_state_dict(checkpoint['generator'])
    model.to(device=machine)
    return model

def prune_model(model):

    print(model.encoder.fnet.named_modules)
    '''
    parameters_to_prune = ((model.conv1, 'weight'),
                            (model.conv2, 'weight'),
                            (model.fc1, 'weight'),
                            (model.fc2, 'weight'),
                            (model.fc3, 'weight'),)

    prune.global_unstructured(parameters_to_prune,
                                pruning_method=prune.L1Unstructured,
                                amount=0.2,)
    '''

def human_format(number):
    
    units = ['','K','M','G','T','P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

def run_helper(rank,model,vocab,args):

    init_logger()
    
    scorer = GNMTGlobalScorer(alpha=0.0, 
                                beta=0.0, 
                                length_penalty="avg", 
                                coverage_penalty="none")
    
    gpu = rank if args.num_gpus > 0 else -1
    if gpu != -1:
        device = "cuda:{}".format(rank) 
        model.to(device)

    src_reader = str2reader["text"]
    tgt_reader = str2reader["text"]

    outfile = open(f'{args.output_name}.preds.rank{rank}','w')
    
    tgt_text_field = vocab['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
    
    translator = Translator(model=model, 
                            fields=vocab,
                            out_file=outfile,
                            src_reader=src_reader, 
                            tgt_reader=tgt_reader, 
                            global_scorer=scorer,
                            gpu=gpu,
                            beam_size=args.beam_size,
                            n_best=args.n_best,
                            max_length=args.max_decode_len)
    
    stride = args.num_gpus if args.num_gpus > 0 else 1
    offset = rank if stride >1 else 0

    src = []
    tscripts = []

    with maybe_fastafile_open(args.input) as fa:
        for i,record in enumerate(fa):
            if (i % stride) == offset:
                    seq = str(record.seq)
                    seq_whitespace =  ' '.join([c for c in seq])
                    src.append(seq_whitespace.encode('utf-8'))
                    tscripts.append(record.id)

    translator.translate_dynamic(src=src,
                        src_feats=None,
                        ids=tscripts,
                        transform=NoOp(opts={}),
                        batch_size=args.max_tokens,
                        batch_type="tokens")

    outfile.close()

def file_cleanup(output_name):

    os.system(f'cat {output_name}.preds.rank* > {output_name}.preds')
    os.system(f'rm {output_name}.preds.rank* ')

def translate_from_transformer_checkpt(args):

    device = 'cpu'
    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
    vocab = checkpoint['vocab']
    
    logger.info("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
    if not options is None:
        for k,v in vars(options).items():
            logger.info(k,v)
    
    model = restore_transformer_model(checkpoint,device,options)
    
    print('EMBEDDING MATRIX',model.encoder.embeddings.emb_luts[0].weight.shape)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"# trainable parameters = {human_format(num_params)}")
    model.eval()

    vocab = checkpoint['vocab']
    if args.num_gpus > 1:
        logger.info('Translating on {} GPUs'.format(args.num_gpus))
        torch.multiprocessing.spawn(run_helper, nprocs=args.num_gpus, args=(model,vocab,args))
        file_cleanup(args.output_name)
    elif args.num_gpus > 0:
        logger.info('Translating on single GPU'.format(args.num_gpus))
        run_helper(args.rank,model,vocab,args)
        file_cleanup(args.output_name)
    else:
        logger.info('Translating on single CPU'.format(args.num_gpus))
        run_helper(args.rank,model,vocab,args)
        file_cleanup(args.output_name)

if __name__ == "__main__":

    args = parse_args()
    translate_from_transformer_checkpt(args)
