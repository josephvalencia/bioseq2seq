import os
import argparse
import pyfiglet
import pandas as pd
from torch.optim import Adam 
import time
import torch

from batcher import iterator_from_csv
from models import make_transformer_model, make_loss_function

from onmt.utils.optimizers import Optimizer
from onmt import Trainer
from onmt.utils.report_manager import build_report_manager, ReportMgr
from onmt.utils.earlystopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from onmt.models import ModelSaver
from onmt.translate import Translator

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
    parser.add_argument("--report-every",'--r',type = int, default = 10000, help = "Number of iterations before calculating statistics")

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

    for batch in iterator:

        src_size = torch.sum(batch.src[1])
        tgt_size = batch.tgt.size(0)*batch.tgt.size(1)

        print(batch.src[1])
        print(batch.tgt.siz())

        total_tokens = src_size+tgt_size

        print("Total tokens: "+str(total_tokens))

        memory = torch.cuda.memory_allocated() / 1048576

        print("Total memory: "+str(memory))

def train(args):

    data = pd.read_csv(args.input,index_col = 0)
    data_iterator = iterator_from_csv(data)

    #test_batch_sizes(data_iterator)

    seq2seq = make_transformer_model()
    seq2seq.to(device = data_iterator.device)

    loss_computer = make_loss_function(device = data_iterator.device,generator = seq2seq.generator)

    adam = Adam(seq2seq.parameters())
    optim = Optimizer(adam,learning_rate = args.learning_rate)

    report_manager = ReportMgr(report_every = args.report_every,tensorboard_writer = SummaryWriter())
    early_stopping = EarlyStopping(tolerance = 5)

    saver = ModelSaver(base_path = args.save_directory,model = seq2seq,\
                       model_opt = None,fields = data_iterator.fields,optim = optim)

    trainer = Trainer(seq2seq,train_loss = loss_computer,earlystopper = early_stopping,\
                      valid_loss = loss_computer,optim = optim,report_manager = report_manager,\
                      model_saver = saver)

    print("Beginning training")

    for i in range(args.max_epochs):
        s = time.time()
        stats = trainer.train(data_iterator,1000000,save_checkpoint_steps = args.report_every)
        e = time.time()
        print("Epoch: "+str(i)+" | Time elapsed: "+str(e-s))

    print("Training complete")

if __name__ == "__main__":

    args = parse_args()
    start_message()
    train(args)
