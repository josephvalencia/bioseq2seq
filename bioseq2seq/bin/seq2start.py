#!/usr/bin/env python
import torch
import argparse
import pandas as pd
import random
import numpy as np
import os
import warnings
import copy

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from bioseq2seq.bin.models import restore_seq2seq_model
from bioseq2seq.bin.data_utils import iterator_from_fasta, build_standard_vocab, IterOnDevice, test_effective_batch_size
from bioseq2seq.inputters.corpus import maybe_fastafile_open
import bioseq2seq.bin.transforms as xfm
from argparse import Namespace

def parse_args():
    """ Parse required and optional configuration arguments"""

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--save_EDA",help="Whether to save encoder-decoder attention", action="store_true")

    # translate required args
    parser.add_argument("--input",help="FASTA file for translation")
    parser.add_argument("--output_name","--o", default="translation",help="Name of file for saving predicted translations")
    parser.add_argument("--checkpoint","--c",help="Model checkpoint (.pt)")
    parser.add_argument("--max_tokens",type=int,default=9000,help="Max number of tokens in prediction batch")
    parser.add_argument("--mode",default="bioseq2seq",help="Inference mode. One of bioseq2seq|EDC")
    parser.add_argument("--rank",type=int,help="Rank of process",default=0)
    parser.add_argument("--address",default =  "127.0.0.1",help = "IP address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "Port for master process in distributed training")
    parser.add_argument("--num_gpus",type=int,help="Number of available GPU machines",default=0)
    parser.add_argument("--model_type","--m", default = "LFNet", help = "Model architecture type.|Transformer|CNN|GFNet|")
    parser.add_argument("--loss_mode",default="original",help="Method of loss computation. original|pointer|weighted")

    # translate optional args
    parser.add_argument("--beam_size","--b",type=int, default=1, help ="Beam size for decoding")
    parser.add_argument("--n_best",type=int, default=1, help="Number of beam hypotheses to list")
    parser.add_argument("--max_decode_len",type=int, default=1, help="Maximum length of protein decoding")
    parser.add_argument("--attn_save_layer",type=int,default=-1,help="If --save_attn flag is used, which layer of EDA to save")
    return parser.parse_args()

def run_helper(rank,args,model,vocab,use_splits=False):
    
    random_seed = 65
    random.seed(random_seed)
    random_state = random.getstate()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    vocab_fields = build_standard_vocab()
    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    src_text_field = vocab_fields['src'].base_field
    src_vocab = src_text_field.vocab
    device = "cpu"
    
    # GPU training
    if args.num_gpus > 0:
        # One CUDA device per process
        device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(device)
        model.cuda()
    
    # multi-GPU training
    if args.num_gpus > 1:
        # configure distributed training with environmental variables
        os.environ['MASTER_ADDR'] = args.address
        os.environ['MASTER_PORT'] = args.port
        torch.distributed.init_process_group(
            backend="nccl",
            init_method= "env://",
            world_size=args.num_gpus,
            rank=rank)
   
    gpu = rank if args.num_gpus > 0 else -1
    world_size = args.num_gpus if args.num_gpus > 0 else 1
    offset = rank if world_size > 1 else 0
    
    if not os.path.isdir(args.output_name):
        os.mkdir(args.output_name)
    input_filename = os.path.split(args.input)[-1].replace('.fasta','').replace('.fa','')
    
    if world_size > 1 :
        savefile += f'.rank-{gpu}'
    tscripts = []
    with maybe_fastafile_open(args.input) as fa:
        for i,record in enumerate(fa):
            #if (i % world_size) == offset:
            tscripts.append(record.id)
   
    tscripts = np.asarray(tscripts) 
    print(f'INFERENCE MODE {args.mode}')
    valid_iter = iterator_from_fasta(src=args.input,
                                    #tgt=args.tgt_input,
                                    tgt=None,
                                    vocab_fields=vocab_fields,
                                    mode=args.mode,
                                    is_train=False,
                                    max_tokens=args.max_tokens,
                                    external_transforms={}, 
                                    rank=rank,
                                    world_size=world_size) 
    valid_iter = IterOnDevice(valid_iter,gpu)
   
    apply_softmax = False
    model.eval()

    #outfile = open(f'{args.output_name}/{input_filename}_preds.txt','w')
    storage = [] 
    for batch in tqdm(valid_iter):
        with torch.no_grad():
            src, src_lengths = batch.src
            tgt = batch.tgt
            # F-prop through the model.

            outputs,enc_attns, attns = model(src, tgt, src_lengths,
				     with_align=False)
            pointer_attn = model.generator(outputs)
            probs =  torch.exp(pointer_attn) 
            pred = pointer_attn.argmax(dim=0).cpu().tolist()
            pc_prob = probs[:-1,:].sum(dim=0).cpu().tolist()
            index = batch.indices.cpu().numpy()
            names = tscripts[index]
            for j in range(batch.batch_size):
                #outfile.write(f'{names[j]}, P(PC) = {pc_prob[j]:.4f}, start={pred[j]}\n')
                print(f'{names[j]}, P(PC) = {pc_prob[j]:.4f}, start={pred[j]}')
                entry = {'tscript' : names[j], 'coding_prob' : f'{pc_prob[j]:.4f}', 'start' : pred[j]}
                storage.append(entry)
    
    df = pd.DataFrame(storage)
    df.to_csv(f'{args.output_name}/{input_filename}_preds.txt',index=False,sep='\t')
    #outfile.close()

def run_attribution(args,device):
    
    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
    # optionally override 
    opts = vars(options)
    cmd_args = vars(args)
    overriden = set(opts).intersection(set(cmd_args))
    print('overriden args',overriden)
    opts.update(cmd_args)
    options = Namespace(**opts)
    vocab = checkpoint['vocab']
    model = restore_seq2seq_model(checkpoint,device,options)

    if not options is None:
        model_name = ""
        print("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
        for k,v in vars(options).items():
            print(k,v)
 
    if args.num_gpus > 1:
        torch.multiprocessing.spawn(run_helper, nprocs=args.num_gpus, args=(args,model,vocab))
    elif args.num_gpus > 0:
        run_helper(args.rank,args,model,vocab)
    else:
        run_helper(0,args,model,vocab)

if __name__ == "__main__": 

    warnings.filterwarnings("ignore")
    args = parse_args()
    device = "cpu"
    run_attribution(args,device)
