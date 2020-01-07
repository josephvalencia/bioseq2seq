import os
import argparse
import pyfiglet
import pandas as pd
import time
import random
import numpy as np


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchtext.data import Dataset, Example,Batch,Field,BucketIterator


from torchtext.data.dataset import RandomShuffler

from batcher import dataset_from_csv, iterator_from_dataset
from models import make_transformer_model, make_loss_function

from bioseq2seq.utils.optimizers import Optimizer
from bioseq2seq.trainer import Trainer
from bioseq2seq.utils.report_manager import build_report_manager, ReportMgr
from bioseq2seq.utils.earlystopping import EarlyStopping
from bioseq2seq.models import ModelSaver
from bioseq2seq.translate import Translator

def parse_args():

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--verbose",action="store_true")

    # required args
    parser.add_argument("--input",'--i',help = "File containing RNA to Protein dataset")
    parser.add_argument("--save-directory","--s",help="Name of directory for saving model checkpoints")
    parser.add_argument("--log-directory",'--l',help = "Name of directory for saving TensorBoard log files" )

    # args
    parser.add_argument("--learning-rate","--lr",type = float,default = 3e-3,help = "Optimizer learning rate")
    parser.add_argument("--max-epochs","--e",type = int,default = 100000,help = "Maximum number of training epochs" )
    parser.add_argument("--report-every",'--r',type = int, default = 100, help = "Number of iterations before calculating statistics")

    return parser.parse_args()

def start_message():

    width = os.get_terminal_size().columns
    bar = "-"*width+"\n"

    centered = lambda x : x.center(width)

    welcome = "DeepTranslate"
    welcome = pyfiglet.figlet_format(welcome)

    print(welcome+"\n")

    info = {"Author":"Joseph Valencia"\
            ,"Date": "11/08/2019",\
            "Version":"1.0.0",
            "License": "Apache 2.0"}

    for k,v in info.items():

        formatted = k+": "+v
        print(formatted)

    print("\n")

def test_batch_sizes(iterator):

    batch_sizes = []

    for batch in iterator:

        src_size = torch.sum(batch.src[1])
        tgt_size = batch.tgt.size(0)*batch.tgt.size(1)

        batch_sizes.append(batch.tgt.size()[1])
        total_tokens = src_size+tgt_size

        memory = torch.cuda.memory_allocated() / (1024*1024)

    df = pd.DataFrame()
    df['BATCH_SIZE'] = batch_sizes
    print(df.describe())

def cleanup():

    dist.destroy_process_group()

def partition(dataset, split_ratios, random_state):

    """Create a random permutation of examples, then split them by ratios

    Arguments:
        examples: a list of data
        splits: split fractions.
        rnd: a random shuffler
    """
    N = len(dataset.examples)
    print("Length of dataset {}".format(N))
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
    print("Partition Lengths: {}".format(data_lens))

    splits = tuple(Dataset(d, dataset.fields)
                       for d in data if d)

    return splits

def train_helper(rank,args,seq2seq,random_state):

    device = torch.device("cuda:{}".format(rank))

    dataframe = pd.read_csv(args.input,index_col = 0) # torchtext Dataset
    train,test,dev = dataset_from_csv(dataframe,1000,random_state) # obtain splits
    #train_partitions = partition(train,split_ratios = [1.0/4 for _ in range(4)],random_state = random_state) # unique data to each machine

    #train_iterator = iterator_from_dataset(train_partitions[rank],10000,device)
    train_iterator = iterator_from_dataset(train,8000,device)
    dev_iterator = iterator_from_dataset(dev,8000,device)

    #test_batch_sizes(train_iterator)
    #test_batch_sizes(dev_iterator)

    dist.init_process_group(
        backend="nccl",
        init_method= "env://",
        world_size=4,
        rank=rank
    )

    #seq2seq.half() # cast to FP16
    seq2seq.to(device)
    seq2seq.parallelize(device_ids = [rank],output_device = rank)
    loss_computer = make_loss_function(device = train_iterator.device,generator = seq2seq.generator)

    adam = Adam(seq2seq.parameters())
    optim = Optimizer(adam,learning_rate = args.learning_rate)

    #seq2seq,optim = amp.initialize(seq2seq,optim,opt_level="O1")

    early_stopping = EarlyStopping(tolerance = 5)

    report_manager = None
    saver = None

    if rank == 0: # only rank 0 device is responsible for saving models and reporting progress

        report_manager = ReportMgr(report_every = args.report_every,tensorboard_writer = SummaryWriter())
        saver = ModelSaver(base_path = args.save_directory,model = seq2seq,\
                model_opt = None,fields = train_iterator.fields,optim = optim)

    trainer = Trainer(seq2seq,train_loss = loss_computer,earlystopper = early_stopping,\
                valid_loss = loss_computer,optim = optim,report_manager = report_manager,\
                model_saver = saver)

    print("Beginning training")

    for i in range(args.max_epochs):
        s = time.time()
        stats = trainer.train(train_iter = train_iterator,train_steps = 1000000,save_checkpoint_steps = args.report_every,valid_iter = dev_iterator)
        e = time.time()
        print("Epoch: "+str(i)+" | Time elapsed: "+str(e-s))

    print("Training complete")

def train(args):

    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = '6000'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

    seq2seq = make_transformer_model()

    n_procs = 4
    random.seed(65)
    state = random.getstate()

    spawn(train_helper, nprocs=n_procs, args=(args,seq2seq,state)) 

if __name__ == "__main__":

    args = parse_args()
    train(args)
