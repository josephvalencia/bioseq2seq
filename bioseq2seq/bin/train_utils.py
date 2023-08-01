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
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam,AdamW, SGD

from bioseq2seq.utils.logging import init_logger, logger
from bioseq2seq.utils.optimizers import Optimizer,noam_decay,AdaFactor
from bioseq2seq.trainer import Trainer
from bioseq2seq.utils.report_manager import ReportMgr
from bioseq2seq.utils.earlystopping import EarlyStopping, AccuracyScorer, ClassAccuracyScorer
from bioseq2seq.models import ModelSaver
from bioseq2seq.utils.loss import NMTLossCompute

#from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition
from bioseq2seq.bin.data_utils import iterator_from_fasta, build_standard_vocab, IterOnDevice, test_effective_batch_size
from bioseq2seq.bin.models import make_cnn_seq2seq,make_cnn_transformer_seq2seq, make_transformer_seq2seq 
from bioseq2seq.bin.models import make_hybrid_seq2seq, Generator,make_lfnet_cnn_seq2seq,attach_pointer_output

def parse_train_args():
    """Parse required and optional command-line arguments.""" 
    
    parser = argparse.ArgumentParser()
    # required args
    parser.add_argument("--train_src",help = "FASTA file of training set RNA")
    parser.add_argument("--train_tgt",help = "FASTA file of training set protein or class labels")
    parser.add_argument("--val_src",help = "FASTA file of validation set RNA")
    parser.add_argument("--val_tgt",help = "FASTA file of validation set protein or class labels")
    parser.add_argument("--save-directory","--s", default = "checkpoints/", help = "Name of directory for saving model checkpoints")
    parser.add_argument("--name", default = "model", help = "Name of saved model")
    parser.add_argument("--model_type","--m", default = "LFNet", help = "Model architecture type.|Transformer|CNN|GFNet|")
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
    parser.add_argument("--rank",type = int, default = 0,help = "Rank in distributed training")
    parser.add_argument("--n_enc_layers",type=int,default = 6,help= "Number of encoder layers")
    parser.add_argument("--class_weight",type=int,default = 1,help="Relative weight of classification label relative to others")
    parser.add_argument("--dropout",type=float,default = 0.1,help="Dropout of non-attention layers")
    parser.add_argument("--n_dec_layers",type=int,default = 6,help="Number of decoder layers")
    parser.add_argument("--model_dim",type=int,default = 64,help ="Size of hidden context embeddings")
    parser.add_argument("--max_rel_pos",type=int,default = 8,help="Max value of relative position embedding")
    parser.add_argument("--filter_size",type=int,default = 50,help="Size of GFNet filter")
    parser.add_argument("--window_size",type=int,default = 200,help="Size of STFNet windows")
    parser.add_argument("--lambd_L1",type=float,default = 0.5,help="Sparsity threshold")
    parser.add_argument("--random_seed",type=int,default = 65,help="Seed")
    parser.add_argument("--encoder_kernel_size",type=int,default = 3,help="Size of CNN encoder kernel")
    parser.add_argument("--encoder_dilation_factor",type=int,default = 1,help="Dilation factor of CNN encoder kernel")
    parser.add_argument("--pos_decay_rate",type=float,default=0.0,help="Exponential decay for loss_mode='weighted") 
    # optional flags
    parser.add_argument("--checkpoint", "--c", help = "Name of .pt model to initialize training")
    parser.add_argument("--finetune",action = "store_true", help = "Reinitialize generator")
    parser.add_argument("--fp16",action="store_true")
    parser.add_argument("--verbose",action="store_true")

    # Ray Tune adds its own cmd line arguments so filter them out
    args, _ =  parser.parse_known_args()
    return args

def make_learning_rate_decay_fn(warmup_steps,model_size):
    return functools.partial(noam_decay,warmup_steps=warmup_steps,model_size=model_size)

def get_input_output_size(vocab_fields):
   
    src_vocab = vocab_fields['src'].base_field.vocab
    tgt_vocab = vocab_fields['tgt'].base_field.vocab
    n_input_classes = len(src_vocab.stoi)
    n_output_classes = len(tgt_vocab.stoi)
    return n_input_classes, n_output_classes

def restore_model_from_args(args,vocab):
     
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''
    
    checkpoint = torch.load(args.checkpoint,map_location = "cpu")
    print(vocab)
    if vocab is None:
        vocab = checkpoint['vocab']
    n_input_classes,n_output_classes =  get_input_output_size(vocab)

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
                                model_dim=args.model_dim,
                                encoder_kernel_size=args.encoder_kernel_size,
                                decoder_kernel_size=args.decoder_kernel_size,
                                encoder_dilation_factor=args.encoder_dilation_factor,
                                decoder_dilation_factor=args.decoder_dilation_factor)
    elif args.model_type == 'CNN-Transformer':
        model = make_cnn_transformer_seq2seq(n_input_classes,
                                n_output_classes,
                                n_enc=args.n_enc_layers,
                                n_dec=args.n_dec_layers,
                                model_dim=args.model_dim,
                                encoder_kernel_size=args.encoder_kernel_size,
                                encoder_dilation_factor=args.encoder_dilation_factor,
                                max_rel_pos=args.max_rel_pos)
    elif args.model_type == 'LFNet-CNN':
        model = make_lfnet_cnn_seq2seq(n_input_classes,
                                        n_output_classes,
                                        n_enc=args.n_enc_layers,
                                        n_dec=args.n_dec_layers,
                                        model_dim=args.model_dim,
                                        window_size=args.window_size,
                                        lambd_L1=args.lambd_L1,
                                        dropout=args.dropout,
                                        decoder_kernel_size=args.decoder_kernel_size)
    else:
        model = make_hybrid_seq2seq(n_input_classes,
                                    n_output_classes,
                                    n_enc=args.n_enc_layers,
                                    n_dec=args.n_dec_layers,
                                    model_dim=args.model_dim,
                                    fourier_type=args.model_type,
                                    max_rel_pos=args.max_rel_pos,
                                    dim_filter=args.filter_size,
                                    window_size=args.window_size,
                                    lambd_L1 = args.lambd_L1,
                                    dropout=args.dropout)
    if args.mode == 'start':
        attach_pointer_output(model,args.model_dim)

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
                                encoder_kernel_size=args.encoder_kernel_size,
                                decoder_kernel_size=args.decoder_kernel_size,
                                encoder_dilation_factor=args.encoder_dilation_factor,
                                decoder_dilation_factor=args.decoder_dilation_factor)
    elif args.model_type == 'CNN-Transformer':
        seq2seq = make_cnn_transformer_seq2seq(n_input_classes,
                                n_output_classes,
                                n_enc=args.n_enc_layers,
                                n_dec=args.n_dec_layers,
                                model_dim=args.model_dim,
                                encoder_kernel_size=args.encoder_kernel_size,
                                encoder_dilation_factor=args.encoder_dilation_factor,
                                max_rel_pos=args.max_rel_pos)
    elif args.model_type == 'LFNet-CNN':
        seq2seq = make_lfnet_cnn_seq2seq(n_input_classes,
                                        n_output_classes,
                                        n_enc=args.n_enc_layers,
                                        n_dec=args.n_dec_layers,
                                        model_dim=args.model_dim,
                                        window_size=args.window_size,
                                        lambd_L1=args.lambd_L1,
                                        dropout=args.dropout,
                                        decoder_kernel_size=args.decoder_kernel_size)
    else:
        seq2seq = make_hybrid_seq2seq(n_input_classes,
                                        n_output_classes,
                                        n_enc=args.n_enc_layers,
                                        n_dec=args.n_dec_layers,
                                        fourier_type=args.model_type,
                                        model_dim=args.model_dim,
                                        max_rel_pos=args.max_rel_pos,
                                        dim_filter=args.filter_size,
                                        window_size=args.window_size,
                                        lambd_L1=args.lambd_L1,
                                        dropout=args.dropout)
    if args.mode == 'start':
        attach_pointer_output(seq2seq,args.model_dim)

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
    logger.info(f"# trainable parameters = {human_format(num_params)}")

def build_or_restore_model(args):

    device = "cpu"

    vocab_fields = build_standard_vocab()

    if args.checkpoint:
        # whether the generator task is different
        if args.finetune:
            logger.info(f'Finetuning {args.model_type} model {args.checkpoint} on dataset {args.train_src} and task {args.mode}')
            # tokenize and numericalize to obtain vocab 
            #train_dataset,val_dataset = dataset_from_df([df_train,df_val.copy()],mode=args.mode)
            seq2seq = restore_model_from_args(args,vocab=vocab_fields)
            replace_generator(seq2seq.model_dim)
        else:
            logger.info(f'Resuming {args.model_type} model {args.checkpoint} on dataset {args.train_src} and task {args.mode}')
            # use the saved vocab and generator
            seq2seq = restore_model_from_args(args,vocab=vocab_fields)
    else:
        logger.info(f'Training a new {args.model_type} model on dataset {args.train_src} and task {args.mode}')
        # tokenize and numericalize to obtain vocab 
        seq2seq = build_model_from_args(args,vocab=vocab_fields)
        
    return seq2seq

def train_helper(rank,args,seq2seq,tune=False):

    """ Train and validate on subset of data. In DistributedDataParallel setting, use one GPU per process.
    Args:
        rank (int): order of process in distributed training
        args (argparse.namespace): See above
        seq2seq (bioseq2seq.models.NMTModel): Encoder-Decoder + generator to train
        random_seed (int): Used for deterministic dataset partitioning
    """
    # seed to control allocation of data to devices
    vocab_fields = build_standard_vocab(with_start=args.mode == 'start')

    device = "cpu"
    # GPU training
    if args.num_gpus > 0:
        # One CUDA device per process
        device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(device)
        seq2seq.cuda()
    
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
    #starts_file = args.train_src.replace("RNA_balanced.fa","balanced_starts.txt")
    starts_file = args.train_src.replace("RNA_no_lncPEP.fa","no_lncPEP_starts.txt")
    
    print('STARTS',starts_file)
    train_iter = iterator_from_fasta(src=args.train_src,
                                    tgt=args.train_tgt,
                                    vocab_fields=vocab_fields,
                                    mode=args.mode,
                                    is_train=True,
                                    max_tokens=args.max_tokens,
                                    rank=rank,
                                    world_size=world_size,
                                    starts=starts_file) 
    train_iter = IterOnDevice(train_iter,gpu)
    #starts_file = args.val_src.replace("RNA_nonredundant_80.fa","nonredundant_80_starts.txt")
    starts_file = args.val_src.replace("RNA_no_lncPEP.fa","no_lncPEP_starts.txt")
    print('STARTS',starts_file)
    valid_iter = iterator_from_fasta(src=args.val_src,
                                    tgt=args.val_tgt,
                                    vocab_fields=vocab_fields,
                                    mode=args.mode,
                                    is_train=False,
                                    max_tokens=args.max_tokens,
                                    rank=rank,
                                    world_size=world_size,
                                    starts=starts_file) 
    valid_iter = IterOnDevice(valid_iter,gpu)
    
    # computes position-wise NLLoss
    pos_decay_rate = args.pos_decay_rate if args.pos_decay_rate > 0 else None 
    reduction = 'none' if (args.mode == 'bioseq2seq' and pos_decay_rate is not None) else 'sum'
    criterion = torch.nn.NLLLoss(ignore_index=1,reduction=reduction)
    train_loss_computer = NMTLossCompute(criterion,generator=seq2seq.generator)
    val_loss_computer = NMTLossCompute(criterion,generator=seq2seq.generator)

    # optimizes model parameters
    optimizer = AdamW(params=seq2seq.parameters())
    lr_fn = make_learning_rate_decay_fn(args.lr_warmup_steps,args.model_dim)
    optim = Optimizer(optimizer,learning_rate = args.learning_rate,learning_rate_decay_fn=lr_fn)
    if args.fp16:
        optim._fp16 = "amp"

    # Ray Tune manages early stopping externally when in tuning mode
    early_stopping = None if tune else EarlyStopping(tolerance=args.patience,scorers=[ClassAccuracyScorer(),AccuracyScorer()])
    
    if tune: # use Ray Tune monitoring
        from bioseq2seq.utils.raytune_utils import RayTuneReportMgr # unconditional import is unsafe, Ray takes over exception handling
        report_manager = RayTuneReportMgr(report_every=args.report_every)
    else: # ONMT monitoring 

        writer = None
        if rank == 0 or args.num_gpus == 1:
            writer = SummaryWriter()
        report_manager = ReportMgr(report_every=args.report_every,tensorboard_writer=writer)

    saver = None
    # only one device is responsible for saving models
    if rank == 0 or args.num_gpus == 1:
        time = datetime.now().strftime('%b%d_%H-%M-%S')
        save_path = f'{args.save_directory}{args.name}_{time}/'
        if not os.path.isdir(save_path):
            try:
                os.mkdir(save_path)
                logger.info(f"Built directory {save_path}")
            except FileExistsError:
                logger.info(f"Directory {save_path} already exists, skipping creation")
        # controls saving model checkpoints
        saver = ModelSaver(base_path=save_path,
                           model=seq2seq,
                           model_opt=args,
                           fields=vocab_fields,
                           optim=optim)

    trainer = Trainer(seq2seq,
                    train_loss=train_loss_computer,
                    earlystopper=early_stopping,
                    valid_loss=val_loss_computer,
                    optim=optim,
                    gpu_rank=rank,
                    n_gpu=args.num_gpus,
                    accum_count=[args.accum_steps],
                    report_manager=report_manager,
                    model_saver=saver,
                    model_dtype='fp16',
                    mode=args.mode,
                    pos_decay_rate=pos_decay_rate)
    
    # training loop
    trainer.train(train_iter=train_iter,
                    train_steps=args.max_epochs,
                    save_checkpoint_steps=args.report_every,
                    valid_iter=valid_iter,
                    valid_steps=args.report_every)

def train_seq2seq(args,tune=False):
    
    init_logger()
    
    # set random seed for reproducibility
    seed = args.random_seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    seq2seq = build_or_restore_model(args) 
    print_model_size(seq2seq)

    if args.num_gpus > 1:
        logger.info("Multi-GPU training")
        torch.multiprocessing.spawn(train_helper, nprocs=args.num_gpus, join=True, args=(args,seq2seq,tune))
        torch.distributed.destroy_process_group()
    elif args.num_gpus == 1:
        logger.info("Single-GPU training")
        train_helper(args.rank,args,seq2seq,tune)
    else:
        logger.info("CPU training")
        train_helper(0,args,seq2seq,tune)
    logger.info("Training complete")
    
