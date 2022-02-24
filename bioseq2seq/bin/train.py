#!/usr/bin/env python
import os
import argparse
import pandas as pd
import random
import torch
import functools
from math import log, floor
from torch import nn
from torchsummary import summary

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bioseq2seq.utils.optimizers import Optimizer,noam_decay,AdaFactor
from bioseq2seq.trainer import Trainer
from bioseq2seq.utils.report_manager import ReportMgr
from bioseq2seq.utils.earlystopping import EarlyStopping, AccuracyScorer, ClassAccuracyScorer
from bioseq2seq.models import ModelSaver
from bioseq2seq.translate import Translator
from bioseq2seq.utils.loss import NMTLossCompute

from bioseq2seq.bin.translate import make_vocab
from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition
from bioseq2seq.bin.models import make_transformer_seq2seq , make_transformer_classifier
from bioseq2seq.bin.models import make_cnn_seq2seq, make_hybrid_seq2seq, Generator

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    """Parse required and optional configuration arguments.""" 
    
    parser = argparse.ArgumentParser()
    # required args
    parser.add_argument("--train",help = "File containing RNA to Protein dataset")
    parser.add_argument("--val",help = "File containing RNA to Protein dataset")
    
    # optional args
    parser.add_argument("--save-directory","--s", default = "checkpoints/", help = "Name of directory for saving model checkpoints")
    parser.add_argument("--learning-rate","--lr", type = float, default = 1e-3,help = "Optimizer learning rate")
    parser.add_argument("--max-epochs","--e", type = int, default = 100000,help = "Maximum number of training epochs" )
    parser.add_argument("--report-every",'--r', type = int, default = 750, help = "Number of iterations before calculating statistics")
    parser.add_argument("--mode", default = "combined", help = "Training mode. Classify for binary classification. Translate for full translation")
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "Number of GPUs to use on node")
    parser.add_argument("--accum_steps", type = int, default = 4, help = "Number of batches to accumulate gradients before update")
    parser.add_argument("--max_tokens",type = int , default = 4500, help = "Max number of tokens in training batch")
    parser.add_argument("--patience", type = int, default = 5, help = "Maximum epochs without improvement")
    parser.add_argument("--address",default =  "127.0.0.1",help = "IP address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "Port for master process in distributed training")
    parser.add_argument("--n_enc_layers",type=int,default = 6,help= "Number of encoder layers")
    parser.add_argument("--n_dec_layers",type=int,default = 6,help="Number of decoder layers")
    parser.add_argument("--model_dim",type=int,default = 64,help ="Size of hidden context embeddings")
    parser.add_argument("--max_rel_pos",type=int,default = 8,help="Max value of relative position embedding")

    # optional flags
    parser.add_argument("--checkpoint", "--c", help = "Name of .pt model to initialize training")
    parser.add_argument("--finetune",action = "store_true", help = "Reinitialize generator")
    parser.add_argument("--verbose",action="store_true")

    return parser.parse_args()

def make_learning_rate_decay_fn(warmup_steps,model_size):
    
    return functools.partial(noam_decay,warmup_steps=warmup_steps,model_size=model_size)

def train_helper(rank,args,seq2seq,random_seed):

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

    # apply differential weighting for classification tokens
    tgt_vocab = train_iterator.fields['tgt'].vocab.itos
    weights = [] 
    for i,c in enumerate(tgt_vocab):
        if c == "<PC>" or c == "<NC>":
            weights.append(1)
            #weights.append(10000)
        else:
            weights.append(1)
    weights = torch.Tensor(weights).to(device) 
    #weights = None

    # computes position-wise NLLoss
    criterion = torch.nn.NLLLoss(weight=weights,ignore_index=1,reduction='sum')
    train_loss_computer = NMTLossCompute(criterion,generator=seq2seq.generator)
    val_loss_computer = NMTLossCompute(criterion,generator=seq2seq.generator)

    # optimizes model parameters
    optimizer = Adam(params = seq2seq.parameters())
    #adafactor = AdaFactor(params=seq2seq.parameters())
    
    lr_fn = make_learning_rate_decay_fn(4000,args.model_dim)
    #lr_fn = None
    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    optim = Optimizer(optimizer,learning_rate = args.learning_rate,learning_rate_decay_fn=lr_fn,fp16=False)
    
    # concludes training if progress does not improve for |patience| epochs
    early_stopping = EarlyStopping(tolerance=args.patience)#,scorers=[ClassAccuracyScorer(),AccuracyScorer()])
    
    report_manager = None
    saver = None
    valid_iterator = None

    # only rank 0 device is responsible for saving models and reporting progress
    if args.num_gpus == 1 or rank == 0:
        # controls metric and time reporting
        report_manager = ReportMgr(report_every=args.report_every,
                                   tensorboard_writer=SummaryWriter())
        # controls saving model checkpoints
        save_path =  args.save_directory + datetime.now().strftime('%b%d_%H-%M-%S')+"/"

        if not os.path.isdir(save_path):
            print("Building directory ...")
            os.mkdir(save_path)
       
        valid_iterator = iterator_from_dataset(val,max_tokens_in_batch,device,train=False)
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
                valid_steps=args.report_every,
                valid_state=None,
                mode=args.mode)

def restore_transformer_seq2seq(checkpoint,n_input_classes,n_output_classes,args):
    
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''

    model = make_transformer_seq2seq(n_input_classes,n_output_classes,
            n_enc=args.n_enc_layers,n_dec=args.n_dec_layers,
            model_dim=args.model_dim,max_rel_pos=args.max_rel_pos)
    model.load_state_dict(checkpoint['model'],strict = False)
    
    return model

def restore_hybrid_seq2seq(checkpoint,n_input_classes,n_output_classes,args):
    
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''
    
    model = make_hybrid_seq2seq(n_input_classes,n_output_classes,
            n_enc=args.n_enc_layers,n_dec=args.n_dec_layers,
            model_dim=args.model_dim,dropout=0.2)
    model.load_state_dict(checkpoint['model'],strict=False)

    #blur_weights(model)

    return model

def blur_weights(model):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.size()) * 0.025)

def restore_cnn_seq2seq(checkpoint,n_input_classes,n_output_classes,args):
    
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''

    model = make_cnn_seq2seq(n_input_classes,n_output_classes,
            n_enc=args.n_enc_layers,n_dec=args.n_dec_layers,
            model_dim=args.model_dim)
    model.load_state_dict(checkpoint['model'],strict = False)
    
    return model

def human_format(number):
    
    units = ['','K','M','G','T','P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

def train(args):
    
    # controls pseudorandom shuffling and partitioning of dataset
    seed = 65

    df_train = pd.read_csv(args.train,sep='\t')
    df_val = pd.read_csv(args.val,sep='\t')

    device = "cpu"
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,map_location = "cpu")
        if args.finetune:
            print(f'Finetuning model {args.checkpoint} on dataset {args.train} and task {args.mode}')
            # tokenize and numericalize to obtain vocab 
            train_dataset,val_dataset = dataset_from_df([df_train.copy(),df_val.copy()],mode=args.mode)
            fields = train_dataset.fields
            n_input_classes = len(fields['src'].vocab)
            n_output_classes = len(fields['tgt'].vocab)
            seq2seq = restore_transformer_seq2seq(checkpoint,n_input_classes,n_output_classes,args)
            # initialize` new output layer
            generator = Generator(args.model_dim,n_output_classes)
            for p in generator.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            seq2seq.generator = generator
        else:
            print(f'Resuming model {args.checkpoint} on dataset {args.train} and task {args.mode}')
            # use the provided vocab
            vocab = checkpoint['vocab'] 
            n_input_classes = len(vocab['src'].vocab.stoi)
            n_output_classes = len(vocab['tgt'].vocab.stoi)
            print('RNA from checkpoint',vocab['src'].vocab.stoi)
            
            seq2seq = restore_hybrid_seq2seq(checkpoint,n_input_classes,n_output_classes,args)
            #seq2seq = restore_cnn_seq2seq(checkpoint,n_input_classes,n_output_classes,args)
            seq2seq.generator.load_state_dict(checkpoint['generator'])
    else:
        print(f'Training a new model on dataset {args.train} and task {args.mode}')
        # tokenize and numericalize to obtain vocab 
        train_dataset,val_dataset = dataset_from_df([df_train.copy(),df_val.copy()],mode=args.mode)
        fields = train_dataset.fields
        n_input_classes = len(fields['src'].vocab)
        n_output_classes = len(fields['tgt'].vocab)
        ''' 
        seq2seq = make_transformer_seq2seq(n_input_classes,
                                            n_output_classes,
                                            n_enc=args.n_enc_layers,
                                            n_dec=args.n_dec_layers,
                                            model_dim=args.model_dim,
                                            max_rel_pos=args.max_rel_pos)
        
        seq2seq = make_cnn_seq2seq(n_input_classes,
                                    n_output_classes,
                                    n_enc=args.n_enc_layers,
                                    n_dec=args.n_dec_layers,
                                    model_dim=args.model_dim)
        ''' 
        seq2seq = make_hybrid_seq2seq(n_input_classes,
                                            n_output_classes,
                                            n_enc=args.n_enc_layers,
                                            n_dec=args.n_dec_layers,
                                            model_dim=args.model_dim,
                                            dim_filter=100)
    
    num_params = sum(p.numel() for p in seq2seq.parameters() if p.requires_grad)
    print(f'# Input classes = {n_input_classes} , # Output classes = {n_output_classes}') 
    print(f"# trainable parameters = {human_format(num_params)}")

    if args.num_gpus > 1:
        print("Multi-GPU training")
        torch.multiprocessing.spawn(train_helper, nprocs=args.num_gpus, args=(args,seq2seq,seed))
        torch.distributed.destroy_process_group()
    elif args.num_gpus > 0:
        print("Single-GPU training")
        train_helper(0,args,seq2seq,seed)
    else:
        print("CPU training")
        train_helper(0,args,seq2seq,seed)
        
    print("Training complete")

if __name__ == "__main__":

    args = parse_args()
    train(args)
    
