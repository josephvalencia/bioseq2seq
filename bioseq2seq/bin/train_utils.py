#!/usr/bin/env python
# currently ignoring DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`.
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import pandas as pd
import random
import torch
import functools
from math import log,floor
from datetime import datetime
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD

from bioseq2seq.utils.optimizers import Optimizer,noam_decay,AdaFactor
from bioseq2seq.trainer import Trainer
from bioseq2seq.utils.report_manager import ReportMgr
from bioseq2seq.utils.earlystopping import EarlyStopping, AccuracyScorer, ClassAccuracyScorer
from bioseq2seq.models import ModelSaver
from bioseq2seq.utils.loss import NMTLossCompute

from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition
from bioseq2seq.bin.models import make_cnn_seq2seq, make_transformer_seq2seq, make_hybrid_seq2seq, Generator


def parse_train_args():
    """Parse required and optional command-line arguments.""" 
    
    parser = argparse.ArgumentParser()
    # required args
    parser.add_argument("--train",help = "File containing RNA to Protein dataset")
    parser.add_argument("--val",help = "File containing RNA to Protein dataset")
    parser.add_argument("--save-directory","--s", default = "checkpoints/", help = "Name of directory for saving model checkpoints")
    parser.add_argument("--model_type","--m", default = "GFNet", help = "Model architecture type.|Transformer|CNN|GFNet|")
    parser.add_argument("--learning-rate","--lr", type = float, default = 1.0,help = "Optimizer learning rate")
    parser.add_argument("--lr_warmup_steps", type = int, default = 4000,help = "Warmup steps for Noam learning rate schedule")
    parser.add_argument("--max-epochs","--e", type = int, default = 100000,help = "Maximum number of training epochs" )
    parser.add_argument("--report-every",'--r', type = int, default = 750, help = "Number of iterations before calculating statistics")
    parser.add_argument("--mode", default = "bioseq2seq", help = "Training mode. EDC for binary classification. bioseq2seq for full translation")
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "Number of GPUs to use on node")
    parser.add_argument("--accum_steps", type = int, default = 4, help = "Number of batches to accumulate gradients before update")
    parser.add_argument("--max_tokens",type = int , default = 4500, help = "Max number of tokens in training batch")
    parser.add_argument("--patience", type = int, default = 5, help = "Maximum epochs without improvement")
    parser.add_argument("--address",default =  "127.0.0.1",help = "IP address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "Port for master process in distributed training")
    parser.add_argument("--n_enc_layers",type=int,default = 6,help= "Number of encoder layers")
    parser.add_argument("--class_weight",type=int,default = 1,help="Relative weight of classification label relative to others")
    parser.add_argument("--dropout",type=float,default = 0.1,help="Dropout of non-attention layers")
    parser.add_argument("--n_dec_layers",type=int,default = 6,help="Number of decoder layers")
    parser.add_argument("--model_dim",type=int,default = 64,help ="Size of hidden context embeddings")
    parser.add_argument("--max_rel_pos",type=int,default = 8,help="Max value of relative position embedding")
    parser.add_argument("--filter_size",type=int,default = 50,help="Size of GFNet filter")

    # optional flags
    parser.add_argument("--checkpoint", "--c", help = "Name of .pt model to initialize training")
    parser.add_argument("--finetune",action = "store_true", help = "Reinitialize generator")
    parser.add_argument("--verbose",action="store_true")

    # Ray Tune adds its own cmd line arguments so filter them out
    args, _ =  parser.parse_known_args()
    return args

def make_learning_rate_decay_fn(warmup_steps,model_size):
    return functools.partial(noam_decay,warmup_steps=warmup_steps,model_size=model_size)

def get_input_output_size(vocab):
    
    n_input_classes = len(vocab['src'].vocab.stoi)
    n_output_classes = len(vocab['tgt'].vocab.stoi)
    return n_input_classes, n_output_classes

def restore_model_from_args(args,vocab):
     
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''
    
    checkpoint = torch.load(args.checkpoint,map_location = "cpu")

    if vocab is None:
        vocab = checkpoint['vocab']
    n_input_classes,n_output_classes =  get_input_output_size(checkpoint)

    if args.model_type == 'Transformer':
        model = make_transformer_seq2seq(n_input_classes,
                                        n_output_classes,
                                        n_enc=args.n_enc_layers,
                                        n_dec=args.n_dec_layers,
                                        model_dim=args.model_dim,
                                        max_rel_pos=args.max_rel_pos)
    elif args.model_type == 'CNN':
        model = make_cnn_seq2seq(n_input_classes,
                                n_output_classes,
                                n_enc=args.n_enc_layers,
                                n_dec=args.n_dec_layers,
                                model_dim=args.model_dim)
    elif args.model_type == 'GFNet':
        model = make_hybrid_seq2seq(n_input_classes,
                                    n_output_classes,
                                    n_enc=args.n_enc_layers,
                                    n_dec=args.n_dec_layers,
                                    model_dim=args.model_dim,
                                    max_rel_pos=args.max_rel_pos,
                                    filter_size=args.filter_size,
                                    dropout=args.dropout)
    else:
        raise Exception('model_type must be one of Transformer, CNN, or GFNet')

    model.load_state_dict(checkpoint['model'],strict = False)
    model.generator.load_state_dict(checkpoint['generator'])
    
    return model

def build_model_from_args(args,vocab=None):

    n_input_classes,n_output_classes = get_input_output_size(vocab) 
    
    if args.model_type == 'Transformer':
        seq2seq = make_transformer_seq2seq(n_input_classes,
                                            n_output_classes,
                                            n_enc=args.n_enc_layers,
                                            n_dec=args.n_dec_layers,
                                            model_dim=args.model_dim,
                                            max_rel_pos=args.max_rel_pos,
                                            dropout=args.dropout)
    elif args.model_type == 'CNN':
        seq2seq = make_cnn_seq2seq(n_input_classes,
                                    n_output_classes,
                                    n_enc=args.n_enc_layers,
                                    n_dec=args.n_dec_layers,
                                    model_dim=args.model_dim,
                                    dropout=args.dropout)
    elif args.model_type == "GFNet":
        seq2seq = make_hybrid_seq2seq(n_input_classes,
                                        n_output_classes,
                                        n_enc=args.n_enc_layers,
                                        n_dec=args.n_dec_layers,
                                        model_dim=args.model_dim,
                                        max_rel_pos=args.max_rel_pos,
                                        dim_filter=args.filter_size,
                                        dropout=args.dropout)
    
    return seq2seq

def replace_generator(seq2seq,model_dim,n_output_classes):
    
    # initialize` new output layer
    generator = Generator(model_dim,n_output_classes)
    for p in generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    seq2seq.generator = generator

def human_format(number):
    
    units = ['','K','M','G','T','P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

def print_model_size(seq2seq):
    
    num_params = sum(p.numel() for p in seq2seq.parameters() if p.requires_grad)
    print(f"# trainable parameters = {human_format(num_params)}")

def build_or_restore_model(args):

    device = "cpu"
    df_train = pd.read_csv(args.train,sep='\t')
    df_val = pd.read_csv(args.val,sep='\t')
    
    if args.checkpoint:
        # whether the generator task is different
        if args.finetune:
            print(f'Finetuning {args.model_type} model {args.checkpoint} on dataset {args.train} and task {args.mode}')
            # tokenize and numericalize to obtain vocab 
            train_dataset,val_dataset = dataset_from_df([df_train,df_val.copy()],mode=args.mode)
            seq2seq = restore_model_from_args(args,vocab=train_dataset.fields)
            replace_generator(seq2seq.model_dim)
        else:
            print(f'Resuming {args.model_type} model {args.checkpoint} on dataset {args.train} and task {args.mode}')
            # use the saved vocab and generator
            seq2seq = restore_model_from_args(args)
    else:
        print(f'Training a new {args.model_type} model on dataset {args.train} and task {args.mode}')
        # tokenize and numericalize to obtain vocab 
        train_dataset,val_dataset = dataset_from_df([df_train,df_val],mode=args.mode)
        seq2seq = build_model_from_args(args,vocab = train_dataset.fields)
        
    return seq2seq

def train_helper(rank,args,seq2seq,random_seed,tune=False):

    """ Train and validate on subset of data. In DistributedDataParallel setting, use one GPU per process.
    Args:
        rank (int): order of process in distributed training
        args (argparse.namespace): See above
        seq2seq (bioseq2seq.models.NMTModel): Encoder-Decoder + generator to train
        random_seed (int): Used for deterministic dataset partitioning
    """
    # seed to control allocation of data to devices
    random.seed(random_seed)
    random_state = random.getstate()
    # determined by GPU memory
    max_tokens_in_batch = args.max_tokens
    # load CSV datasets and convert to torchtext.Dataset
    df_train = pd.read_csv(args.train,sep='\t')
    df_val = pd.read_csv(args.val,sep='\t')
    train,val = dataset_from_df([df_train.copy(),df_val.copy()],mode=args.mode)
    device = "cpu"
    
    # GPU training
    if args.num_gpus > 0:
        # One CUDA device per process
        device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(device)
        seq2seq.cuda()
    # multi-GPU training
    if args.num_gpus > 1:
        # provide unique data to each process
        splits = [1.0/args.num_gpus for _ in range(args.num_gpus)]
        train_partitions = partition(train,split_ratios = splits,random_state = random_state)
        local_slice = train_partitions[rank]
        # iterator over training batches
        train_iterator = iterator_from_dataset(local_slice,max_tokens_in_batch,device,train=True)
        # configure distributed training with environmental variables
        os.environ['MASTER_ADDR'] = args.address
        os.environ['MASTER_PORT'] = args.port
        torch.distributed.init_process_group(
            backend="nccl",
            init_method= "env://",
            world_size=args.num_gpus,
            rank=rank)
    else:
        train_iterator = iterator_from_dataset(train,max_tokens_in_batch,device,train=True)
    
    weights = None
    # apply differential weighting for classification tokens
    if args.class_weight > 1:
        tgt_vocab = train_iterator.fields['tgt'].vocab.itos
        weight_fn = lambda x : class_weight if x == "<PC>" or x == "<NC>" else 1 
        weights = [weight_fn(c) for c in tgt_vocab]
        weights = torch.Tensor(weights).to(device) 

    # computes position-wise NLLoss
    criterion = torch.nn.NLLLoss(weight=weights,ignore_index=1,reduction='sum')
    train_loss_computer = NMTLossCompute(criterion,generator=seq2seq.generator)
    val_loss_computer = NMTLossCompute(criterion,generator=seq2seq.generator)

    # optimizes model parameters
    optimizer = Adam(params=seq2seq.parameters())
    lr_fn = make_learning_rate_decay_fn(args.lr_warmup_steps,args.model_dim)
    optim = Optimizer(optimizer,learning_rate = args.learning_rate,learning_rate_decay_fn=lr_fn,fp16=True)

    saver = None
    valid_iterator = None
    report_manager = None
    
    # Ray Tune manages early stopping externally when in tuning mode
    early_stopping = None if tune else EarlyStopping(tolerance=args.patience,scorers=[ClassAccuracyScorer(),AccuracyScorer()])

    # only rank 0 device is responsible for saving models and reporting progress
    if rank == 0:
        if tune: # use Ray Tune monitoring
            from bioseq2seq.utils.raytune_utils import RayTuneReportMgr # unconditional import is unsafe, Ray takes over exception handling
            report_manager = RayTuneReportMgr(report_every=args.report_every)
        else: # ONMT monitoring 
            report_manager = ReportMgr(report_every=args.report_every,tensorboard_writer=SummaryWriter())
        save_path =  args.save_directory + datetime.now().strftime('%b%d_%H-%M-%S')+"/"
        if not os.path.isdir(save_path):
            print("Building directory ...")
            os.mkdir(save_path)
       
        valid_iterator = iterator_from_dataset(val,max_tokens_in_batch,device,train=False)
        # controls saving model checkpoints
        saver = ModelSaver(base_path=save_path,
                           model=seq2seq,
                           model_opt=args,
                           fields=valid_iterator.fields,
                           optim=optim)

    # controls training and validation
    trainer = Trainer(seq2seq,
                    train_loss=train_loss_computer,
                    earlystopper=early_stopping,
                    valid_loss=val_loss_computer,
                    optim=optim,
                    rank=rank,
                    gpus=args.num_gpus,
                    accum_count=[args.accum_steps],
                    report_manager=report_manager,
                    model_saver=saver,
                    model_dtype='fp16')
    
    # training loop
    trainer.train(train_iter=train_iterator,
                    train_steps=args.max_epochs,
                    save_checkpoint_steps=args.report_every,
                    valid_iter=valid_iterator,
                    valid_steps=args.report_every)

def train_seq2seq(args,tune=False):
    
    seed = 65
    seq2seq = build_or_restore_model(args) 
    print_model_size(seq2seq)

    if args.num_gpus > 1:
        print("Multi-GPU training")
        torch.multiprocessing.spawn(train_helper, nprocs=args.num_gpus, join=True, args=(args,seq2seq,seed,tune))
        torch.distributed.destroy_process_group()
    elif args.num_gpus == 1:
        print("Single-GPU training")
        train_helper(0,args,seq2seq,seed,tune)
    else:
        print("CPU training")
        train_helper(0,args,seq2seq,seed,tune)
    print("Training complete")
    
