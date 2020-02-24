#!/usr/bin/env python
import os
import argparse
import pandas as pd
import random
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD

from bioseq2seq.utils.optimizers import Optimizer
from bioseq2seq.trainer import Trainer
from bioseq2seq.utils.report_manager import ReportMgr
from bioseq2seq.utils.earlystopping import EarlyStopping
from bioseq2seq.models import ModelSaver
from bioseq2seq.translate import Translator
from bioseq2seq.utils.loss import NMTLossCompute
from bioseq2seq.bin.translate import make_vocab

from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition,train_test_val_split
from bioseq2seq.bin.models import make_transformer_model

def parse_args():

    parser = argparse.ArgumentParser()

    # required args
    parser.add_argument("--input",'--i',help = "File containing RNA to Protein dataset")

    # optional args
    parser.add_argument("--save-directory","--s", help = "Name of directory for saving model checkpoints")
    parser.add_argument("--log-directory",'--l', help = "Name of directory for saving TensorBoard log files" )
    parser.add_argument("--learning-rate","--lr", type = float, default = 5e-3,help = "Optimizer learning rate")
    parser.add_argument("--max-epochs","--e", type = int, default = 100000,help = "Maximum number of training epochs" )
    parser.add_argument("--report-every",'--r', type = int, default = 3031, help = "Number of iterations before calculating statistics")
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "Number of GPUs to use on node")
    parser.add_argument("--accum_steps", type = int, default = 1, help = "Number of batches to accumulate gradients before update")
    parser.add_argument("--rank", type = int, default = 0, help = "Rank of node in multi-node training")
    parser.add_argument("--max_len_transcript", type = int, default = 1000, help = "Maximum length of transcript")
    parser.add_argument("--patience", type = int, default = 15, help = "Maximum epochs without improvement")

    # optional flags
    parser.add_argument("--verbose",action="store_true")

    return parser.parse_args()

def train_helper(rank,args,seq2seq,random_seed):

    random.seed(random_seed)
    random_state = random.getstate()

    # determined by GPU memory
    max_tokens_in_batch = 7000

    # raw GENCODE transcript data. cols = ['ID','RNA','PROTEIN']
    dataframe = pd.read_csv(args.input)

    # obtain splits. Default 80/10/10. Filter below max_len_transcript
    df_train,df_test,df_val = train_test_val_split(dataframe,args.max_len_transcript,random_seed)

    # convert to torchtext.Dataset
    train,test,val = dataset_from_df(df_train,df_test,df_val)

    device = "cpu"

    if args.num_gpus > 0: # GPU training

        # One CUDA device per process
        device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(device)
        seq2seq.cuda()

    if args.num_gpus > 1: # multi-GPU training

        # provide unique data to each process
        splits = [1.0/args.num_gpus for _ in range(args.num_gpus)]
        train_partitions = partition(train,split_ratios = splits,random_state = random_state)
        local_slice = train_partitions[rank]

        # iterator over training batches
        train_iterator = iterator_from_dataset(local_slice,max_tokens_in_batch,device,train=True)

        # configure distributed training with environmental variables
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = '6000'

        torch.distributed.init_process_group(
            backend="nccl",
            init_method= "env://",
            world_size=args.num_gpus,
            rank=rank)
    else:
        train_iterator = iterator_from_dataset(train,max_tokens_in_batch,device,train=True)

    # computes position-wise NLLoss
    criterion = torch.nn.NLLLoss(ignore_index = 1,reduction='sum')
    train_loss_computer = NMTLossCompute(criterion,generator=seq2seq.generator)
    val_loss_computer = NMTLossCompute(criterion,generator=seq2seq.generator)

    # optimizes model parameters
    adam = Adam(params = seq2seq.parameters())
    optim = Optimizer(adam,learning_rate = args.learning_rate)

    # concludes training if progress does not improve for |patience| epochs
    early_stopping = EarlyStopping(tolerance = args.patience)

    report_manager = saver = valid_state = valid_iterator = None

    # only rank 0 device is responsible for saving models and reporting progress
    if args.num_gpus == 1 or rank == 0:

        # controls metric and time reporting
        report_manager = ReportMgr(report_every = args.report_every,
                                   tensorboard_writer = SummaryWriter())

        # controls saving model checkpoints
        saver = ModelSaver(base_path = args.save_directory,
                           model = seq2seq,
                           model_opt = None,
                           fields = train_iterator.fields,
                           optim = optim)

        valid_iterator = iterator_from_dataset(val,max_tokens_in_batch,device,train=False)

        # Translator builds its own iterator from unprocessed data
        valid_state = wrap_validation_state(fields = valid_iterator.fields,
                                            rna = df_val['RNA'].tolist()[:100],
                                            protein = df_val['Protein'].tolist()[:100])

    # controls training and validation
    trainer = Trainer(seq2seq,
                      train_loss = train_loss_computer,
                      earlystopper = early_stopping,
                      valid_loss = val_loss_computer,
                      optim = optim,
                      rank = rank,
                      gpus = args.num_gpus,
                      accum_count = [args.accum_steps],
                      report_manager = report_manager,
                      model_saver = saver)

    # training loop
    trainer.train(train_iter = train_iterator,
                  train_steps = args.max_epochs,
                  save_checkpoint_steps = args.report_every,
                  valid_iter = valid_iterator,
                  valid_steps = args.report_every,
                  valid_state = valid_state)

def wrap_validation_state(fields,rna,protein):

    fields = make_vocab(fields,rna,protein)
    return tuple([fields,rna,protein])

def train(args):

    # controls pseudorandom shuffling and partitioning of dataset
    seed = 65

    seq2seq = make_transformer_model()

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
