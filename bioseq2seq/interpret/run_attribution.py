#!/usr/bin/env python
import torch
import argparse
import pandas as pd
import random
import numpy as np
import os
import warnings
import copy
from scipy import stats , signal

from embedding import SynonymousShuffleExpectedGradients, GradientAttribution, EmbeddingMDIG, EmbeddingIG
from onehot import OneHotSalience, OneHotIntegratedGradients, OneHotMDIG, OneHotExpectedGradients
from sampler import DiscreteLangevinSampler
from ism import InSilicoMutagenesis

from torch.nn.parallel import DistributedDataParallel as DDP

from bioseq2seq.bin.models import restore_seq2seq_model
from bioseq2seq.bin.data_utils import iterator_from_fasta, build_standard_vocab, IterOnDevice, test_effective_batch_size
from bioseq2seq.inputters.corpus import maybe_fastafile_open
import bioseq2seq.bin.transforms as xfm
from argparse import Namespace

from analysis.average_attentions import plot_power_spectrum
import matplotlib.pyplot as plt

def plot_frequency_heatmap(tscript_name,attr,tgt_class,metric):

    periods = [18,9,6,5,4,3,2]
    tick_locs = [1.0 / p for p in periods]
   
    fft_window = 250
    # lower right NC spectrogram
    plt.subplot(212)
    Pxx, freqs, bins, im = plt.specgram(attr.ravel(),Fs=1.0,NFFT=fft_window,noverlap=fft_window/2,cmap='plasma',detrend = 'mean')
    bins = [int(x) for x in bins]
    plt.xlabel("Window center (nt.)")
    plt.ylabel("Period (nt.)")
    plt.colorbar(im).set_label('Attribution Power')
    plt.yticks(tick_locs,periods)
    plt.xticks(bins,bins)

    bins = [0]+bins+[attr.size]
    # upper left PC line plot
    plt.subplot(211)
    plt.plot(attr.ravel(),linewidth=1.0)
    plt.ylabel('Attribution')
    plt.xticks(bins,bins)
    
    '''
    plt.subplot(312)
    # set parameters for drawing gene
    exon_start = 50-.5
    exon_stop = 200+.5
    y = 0

    # draw gene
    plt.axhline(y, color='k', linewidth=1)
    plt.plot([exon_start, exon_stop],[y, y], color='k', linewidth=7.5, solid_capstyle='butt')
    ''' 
    
    plt.tight_layout() 
    plt.savefig(f'ps_examples/{tscript_name}_{tgt_class}_{metric}_spectrogram.svg')
    plt.close()

def print_pairwise_corrs(batch_attributions,steps):
    
    copy1 = batch_attributions[-2]
    copy2 = batch_attributions[-1]
    
    dist = np.linalg.norm(copy2 - copy1,ord='fro') / np.linalg.norm(copy2,ord='fro')
    print('% Distance between last two trials (Frobenius norm)',dist)
    
    stacked = np.stack(batch_attributions,axis=0)
    M = stacked.sum(axis=-1)
    cov = M @ M.T
    diag_term = np.linalg.inv(np.sqrt(np.diag(np.diag(cov))))
    pearson_corr = diag_term @ cov @ diag_term
   
    df = pd.DataFrame(pearson_corr,columns = steps,index=steps)
    print('pairwise Pearson corr by n_steps')
    print(df.round(3))


def add_synonymous_shuffled_to_vocab(num_copies,vocab_fields):

    for i in range(num_copies):
        shuffled_field = copy.deepcopy(vocab_fields['src'])
        seq_name = f'src_shuffled_{i}'
        shuffled_field.fields = [(seq_name,shuffled_field.base_field)] 
        vocab_fields[seq_name] = shuffled_field 

def add_point_mutations_to_vocab(vocab_fields):

    for base in ['A','G','C','T']: 
        for loc in range(-12,60):
            seq_name = f'src_{loc}->{base}'
            shuffled_field = copy.deepcopy(vocab_fields['src'])
            shuffled_field.fields = [(seq_name,shuffled_field.base_field)] 
            vocab_fields[seq_name] = shuffled_field 

def parse_args():

    """ Parse required and optional configuration arguments"""
    parser = argparse.ArgumentParser()
    
    # optional flags
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--negative_class",action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--tgt_input",default=None,help="File for translation")
    parser.add_argument("--checkpoint", "--c",help ="ONMT checkpoint (.pt)")
    parser.add_argument("--inference_mode",default ="combined")
    parser.add_argument("--attribution_mode",default="ig")
    parser.add_argument("--baseline",default="zero", help="zero|avg|A|C|G|T")
    parser.add_argument("--tgt_class",default="<PC>", help="<PC>|<NC>")
    parser.add_argument("--tgt_pos",type=int,default=1, help="token position in target")
    parser.add_argument("--max_tokens",type=int,default = 4500, help = "Max number of tokens in training batch")
    parser.add_argument("--name",default = "temp")
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "Number of GPUs to use on node")
    parser.add_argument("--address",default =  "127.0.0.1",help = "IP address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "Port for master process in distributed training")
    parser.add_argument("--mutation_prob",type=float, default=0.25 ,help = "Prob of mutation")
    
    return parser.parse_args()

def run_helper(rank,args,model,vocab,use_splits=False):
    
    random_seed = 65
    random.seed(random_seed)
    random_state = random.getstate()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    target_pos = args.tgt_pos
    vocab_fields = build_standard_vocab()
    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    src_text_field = vocab_fields['src'].base_field
    src_vocab = src_text_field.vocab
    print(tgt_vocab.stoi) 
    print(src_vocab.stoi) 
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
    
    class_name = args.tgt_class
    tgt_class = args.tgt_class
    if tgt_class == "PC" or tgt_class == "NC":
        tgt_class = f'<{tgt_class}>'
    if tgt_class == "STOP":
        tgt_class = "</s>"
    
    pathname = os.path.split(args.name)
    
    if pathname[0] == '':
        pathname = pathname[-1]
    else:
        pathname = pathname[0]
    if not os.path.isdir(pathname):
        os.mkdir(pathname)
    savefile = "{}/{}.{}.{}".format(args.name,class_name,target_pos,args.attribution_mode)
    print(savefile)
    
    tscripts = []
    with maybe_fastafile_open(args.input) as fa:
        for i,record in enumerate(fa):
            if (i % world_size) == offset:
                    tscripts.append(record.id)
    
    # set up synonymous shuffled copies
    n_samples = 512
    minibatch_size = 16
    
    xforms = {}
    if args.attribution_mode == 'EG-embed' or args.attribution_mode == 'EG':
        shuffle_options = Namespace(num_copies=n_samples,mutation_prob=args.mutation_prob)
        xforms = {'add_synonymous_mutations' : xfm.SynonymousCopies(opts=shuffle_options)}
        add_synonymous_shuffled_to_vocab(n_samples,vocab_fields)
    if args.attribution_mode == 'ISM':
        #xforms = {'add_point_mutations' : xfm.GenPointMutations(opts={})}
        #add_point_mutations_to_vocab(vocab_fields)
        pass
    print(f'INFERENCE MODE {args.inference_mode}')
    valid_iter = iterator_from_fasta(src=args.input,
                                    tgt=args.tgt_input,
                                    vocab_fields=vocab_fields,
                                    mode=args.inference_mode,
                                    is_train=False,
                                    max_tokens=args.max_tokens,
                                    external_transforms=xforms, 
                                    rank=rank,
                                    world_size=world_size) 
    valid_iter = IterOnDevice(valid_iter,gpu)
   
    apply_softmax = False
    model.eval()
    sep = '___________________________________' 
    
    print(sep) 
    if args.attribution_mode == 'grad':
        print(f'Running Saliency wrt. {tgt_class}')
        attributor = OneHotSalience(model,device,vocab,tgt_class, \
                                            softmax=apply_softmax,sample_size=n_samples,minibatch_size=minibatch_size,
                                            times_input=False,smoothgrad=False)
    elif args.attribution_mode == 'MDIG': 
        print(f'Running MDIG wrt. {tgt_class}')
        attributor = OneHotMDIG(model,device,vocab,tgt_class, \
                                            softmax=apply_softmax,sample_size=n_samples,minibatch_size=minibatch_size,
                                            times_input=False,smoothgrad=False)
    elif args.attribution_mode == 'IG': 
        print(f'Running IG wrt. {tgt_class}')
        attributor = OneHotIntegratedGradients(model,device,vocab,tgt_class, \
                                            softmax=apply_softmax,sample_size=n_samples,minibatch_size=minibatch_size,
                                            times_input=False,smoothgrad=False)
    elif args.attribution_mode == 'EG': 
        print(f'Running Expected grads (onehot) wrt. {tgt_class}')
        attributor = OneHotExpectedGradients(model,device,vocab,tgt_class, \
                                            softmax=apply_softmax,sample_size=n_samples,minibatch_size=minibatch_size,
                                            times_input=False,smoothgrad=False)
    elif args.attribution_mode == 'ISM': 
        print(f'Running ISM wrt. {tgt_class}')
        attributor = InSilicoMutagenesis(model,device,vocab,tgt_class, \
                                            softmax=apply_softmax,sample_size=n_samples,minibatch_size=minibatch_size,
                                            times_input=False,smoothgrad=False)
    elif args.attribution_mode == 'langevin':
        print(f'Running Discrete Langevin wrt. {tgt_class}')
        attributor = DiscreteLangevinSampler(model,device,vocab,tgt_class, \
                                            softmax=apply_softmax,sample_size=n_samples,minibatch_size=minibatch_size,
                                            times_input=False,smoothgrad=False)
    elif args.attribution_mode == 'MDIG-embed':
        print(f'Running MDIG (embed) wrt. {tgt_class}')
        attributor = EmbeddingMDIG(model,device,vocab,tgt_class, \
                                            softmax=apply_softmax,sample_size=n_samples,minibatch_size=minibatch_size)
    elif args.attribution_mode == 'IG-embed':
        print(f'Running IG (embed) wrt. {tgt_class}')
        attributor = EmbeddingIG(model,device,vocab,tgt_class, \
                                            softmax=apply_softmax,sample_size=n_samples,minibatch_size=minibatch_size)
    elif args.attribution_mode == 'EG-embed':
        print(f'Running Expected grads (embed) wrt. {tgt_class}')
        attributor = SynonymousShuffleExpectedGradients(model,device,vocab,tgt_class,\
                                            softmax=apply_softmax,sample_size=n_samples)
    else:
        raise ValueError(f"{args.attribution_mode} is not a valid value for --attribution_mode")

    print(sep) 
    attributor.run(savefile,valid_iter,target_pos,args.baseline,tscripts)
  
def run_attribution(args,device):
    
    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
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
