import os
import argparse
import pandas as pd
import time
import random
import numpy as np
import torch
from tqdm import tqdm
import copy

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam,SGD
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from torchtext.data import Dataset, Example,Batch,Field
from torchtext.data.dataset import RandomShuffler

from bioseq2seq.utils.optimizers import Optimizer
from bioseq2seq.trainer import Trainer
from bioseq2seq.utils.report_manager import ReportMgr
from bioseq2seq.utils.earlystopping import EarlyStopping
from bioseq2seq.models import ModelSaver
from bioseq2seq.translate import Translator

from batcher import dataset_from_csv, iterator_from_dataset,filter_by_length, dataset_from_csv_v2
from models import make_transformer_model, make_loss_function

def parse_args():

    parser = argparse.ArgumentParser()

    # required args
    parser.add_argument("--input",'--i',help = "File containing RNA to Protein dataset")

    # optional args
    parser.add_argument("--save-directory","--s",help="Name of directory for saving model checkpoints")
    parser.add_argument("--log-directory",'--l',help = "Name of directory for saving TensorBoard log files" )
    parser.add_argument("--learning-rate","--lr",type = float,default = 3e-3,help = "Optimizer learning rate")
    parser.add_argument("--max-epochs","--e",type = int,default = 150000,help = "Maximum number of training epochs" )
    parser.add_argument("--report-every",'--r',type = int, default = 100, help = "Number of iterations before calculating statistics")
    parser.add_argument("--num_gpus","--g",type = int,default = 1, help = "Number of GPUs for training")
    parser.add_argument("--rank",type = int, default = 0, help = "Rank of node in multi-machine training")

    # optional flags
    parser.add_argument("--verbose",action="store_true")

    return parser.parse_args()

def cleanup():
    """Kill all spawned threads"""
    dist.destroy_process_group()

def partition(dataset, split_ratios, random_state):

    """Create a random permutation of examples, then split them by ratios

    Arguments:
        dataset (torchtext.dataset): Dataset to partition
        splits (list): split fractions for Dataset partitions.
        random_state (int) : Random seed for shuffler
    """
    N = len(dataset.examples)
    rnd = RandomShuffler(random_state)
    randperm = rnd(range(N))

    indices = []
    current_idx = 0

    for ratio in split_ratios[:-1]:
        partition_len = int(round(ratio*N))
        partition = randperm[current_idx:current_idx+partition_len]
        indices.append(partition)
        current_idx +=partition_len

    last_partition = randperm[current_idx:]
    indices.append(last_partition)

    data = tuple([dataset.examples[i] for i in index] for index in indices)
    data_lens = [len(x) for x in data]

    splits = tuple(Dataset(d, dataset.fields)
                       for d in data )

    return splits

def train_helper(rank,args,seq2seq,generator,random_seed):

    random.seed(random_seed)
    random_state = random.getstate()

    device = torch.device("cuda:{}".format(rank)) # One process per GPU

    dataframe = pd.read_csv(args.input,index_col = 0) # load translation mapping table
    train,test,dev = dataset_from_csv_v2(dataframe,1000,random_seed) # obtain splits

    if args.num_gpus > 1: # in multi-GPU case provide unique data to each process

        splits = [1.0/args.num_gpus for _ in range(args.num_gpus)]
        train_partitions = partition(train,split_ratios = splits,random_state = random_state)
        train_iterator = iterator_from_dataset(train_partitions[rank],7000,device)

        dist.init_process_group(
            backend="nccl",
            init_method= "env://",
            world_size=4,
            rank=rank)

    else:
        train_iterator = iterator_from_dataset(train,7000,device)

    dev_iterator = iterator_from_dataset(dev,7000,device)

    torch.cuda.set_device(device)
    seq2seq.cuda()
    generator.cuda()

    params = list(seq2seq.parameters())+list(generator.parameters())

    loss_computer = make_loss_function(device = train_iterator.device,generator = generator)

    adam = Adam(params)
    optim = Optimizer(adam,learning_rate = args.learning_rate)
    #seq2seq,optim = amp.initialize(seq2seq,optim,opt_level="O1")

    early_stopping = EarlyStopping(tolerance = 5)

    report_manager = None
    saver = None

    if args.num_gpus ==1 or rank == 0: # only rank 0 device is responsible for saving models and reporting progress

        save_wrapper = copy.deepcopy(seq2seq)
        save_wrapper.generator = copy.deepcopy(generator)

        report_manager = ReportMgr(report_every = args.report_every,
                                   tensorboard_writer = SummaryWriter())

        saver = ModelSaver(base_path = args.save_directory,
                           model = save_wrapper,
                           model_opt = None,
                           fields = train_iterator.fields,
                           optim = optim)

    trainer = Trainer(seq2seq,
                      train_loss = loss_computer,
                      earlystopper = early_stopping,
                      valid_loss = loss_computer,
                      optim = optim, rank = rank,
                      gpus = args.num_gpus,
                      report_manager = report_manager,
                      model_saver = saver)

    for i in range(args.max_epochs):
        s = time.time()
        stats = trainer.train(train_iter = train_iterator,
                              train_steps = 1500000,
                              save_checkpoint_steps = args.report_every,
                              valid_iter = dev_iterator,valid_steps = args.report_every)
        e = time.time()
        print("Epoch: "+str(i)+" | Time elapsed: "+str(e-s))

def train(args):

    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = '6000'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

    seq2seq,generator = make_transformer_model()

    n_procs = args.num_gpus
    seed = 65

    if n_procs > 1:
        print("Multi-GPU training")
        spawn(train_helper, nprocs=n_procs, args=(args,seq2seq,generator,seed))
    else:
        print("Single-GPU training")
        train_helper(0,args,seq2seq,generator,state)

if __name__ == "__main__":

    args = parse_args()
    train(args)

