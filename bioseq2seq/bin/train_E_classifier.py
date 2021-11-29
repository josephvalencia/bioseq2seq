#!/usr/bin/env python
import os
import argparse
import pandas as pd
import random
import torch
import numpy as np
import time
from datetime import datetime
from math import log, floor
from sklearn.metrics import accuracy_score

from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP

from bioseq2seq.models import ModelSaver
from bioseq2seq.utils.optimizers import Optimizer
from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition
from bioseq2seq.bin.models import make_transformer_seq2seq , make_transformer_classifier


def parse_args():
    """Parse required and optional configuration arguments.""" 
    
    parser = argparse.ArgumentParser()
    # required args
    #parser.add_argument("--input",'--i',help = "File containing RNA to Protein dataset")
    parser.add_argument("--train",help = "File containing RNA to Protein dataset")
    parser.add_argument("--val",help = "File containing RNA to Protein dataset")
    
    # optional args
    parser.add_argument("--save-directory","--s", default = "checkpoints/", help = "Name of directory for saving model checkpoints")
    parser.add_argument("--log-directory",'--l',default = "runs/", help = "Name of directory for saving TensorBoard log files" )
    parser.add_argument("--learning-rate","--lr", type = float, default = 1e-3,help = "Optimizer learning rate")
    parser.add_argument("--max-epochs","--e", type = int, default = 100000,help = "Maximum number of training epochs" )
    parser.add_argument("--report-every",'--r', type = int, default = 750, help = "Number of iterations before calculating statistics")
    parser.add_argument("--mode", default = "combined", help = "Training mode. Classify for binary classification. Translate for full translation")
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "Number of GPUs to use on node")
    parser.add_argument("--accum_steps", type = int, default = 8, help = "Number of batches to accumulate gradients before update")
    parser.add_argument("--max_tokens",type = int , default = 3000, help = "Max number of tokens in training batch")
    parser.add_argument("--rank", type = int, default = 0, help = "Rank of node in multi-node training")
    parser.add_argument("--max_len_transcript", type = int, default = 1000, help = "Maximum length of transcript")
    parser.add_argument("--patience", type = int, default = 8, help = "Maximum epochs without improvement")
    parser.add_argument("--address",default =  "127.0.0.1",help = "IP address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "Port for master process in distributed training")
    parser.add_argument("--n_enc_layers",type=int,default = 6,help= "Number of encoder layers")
    parser.add_argument("--n_dec_layers",type=int,default = 6,help="Number of decoder layers")
    parser.add_argument("--model_dim",type=int,default = 256,help ="Size of hidden context embeddings")
    parser.add_argument("--max_rel_pos",type=int,default = 8,help="Max value of relative position embedding")

    # optional flags
    parser.add_argument("--checkpoint", "--c", help = "Name of .pt model to initialize training")
    parser.add_argument("--finetune",action = "store_true", help = "Reinitialize generator")
    parser.add_argument("--verbose",action="store_true")

    return parser.parse_args()

def human_format(number):
    units = ['','K','M','G','T','P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k**magnitude, units[magnitude])

def train_helper(rank,df_train,df_val,model,args,random_seed):

    """ Train and validate on subset of data. In DistributedDataParallel setting, use one GPU per process.
    Args:
        rank (int): order of process in distributed training
        args (argparse.namespace): See above
        seq2seq (bioseq2seq.models.NMTModel): Encoder-Decoder + generator to train
        random_seed (int): Used for deterministic dataset partitioning
    """
    random.seed(random_seed)
    random_state = random.getstate()
    
    # determined by GPU memory
    max_tokens_in_batch = args.max_tokens
    train_dataset,val_dataset = dataset_from_df([df_train.copy(),df_val.copy()],mode="ENC_only")
    
    if args.num_gpus > 0: # GPU training
        # One CUDA device per process
        device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(device)
        model.cuda()

    if args.num_gpus > 1: # multi-GPU training
        # provide unique data to each process
        splits = [1.0/args.num_gpus for _ in range(args.num_gpus)]
        train_partitions = partition(train_dataset,split_ratios = splits,random_state = random_state)
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
        train_iterator = iterator_from_dataset(train_dataset,max_tokens_in_batch,device,train=True)

    valid_iterator = iterator_from_dataset(val_dataset,max_tokens_in_batch,device,train=False)
   
    # controls saving model checkpoints
    job_name = datetime.now().strftime('%b%d_%H-%M-%S') 
    save_path =  args.save_directory + job_name+ "/"
    log_file = job_name+"-training_log.csv"

    # computes position-wise NLLoss
    criterion = torch.nn.NLLLoss(ignore_index=1,reduction='sum')
    optimizer = Adam(params = model.parameters())

    saver = None

    if rank == 0 :
        
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        saver = ModelSaver(base_path=save_path,
                               model=model,
                               model_opt=args,
                               fields=valid_iterator.fields,
                               optim=optimizer)

        with open(log_file,'w') as outFile:
            outFile.write("time, step, train_accuracy, train_loss, val_accuracy, val_loss\n")

    model.train()
    
    running_loss = 0.0
    running_pred = []
    running_tgt = []
    
    fp16 = True

    for i,  batch in enumerate(train_iterator):
        src, src_lens = batch.src
        tgt = torch.squeeze(batch.tgt,dim=0)
        tgt = torch.squeeze(tgt,dim=1)
        print(f'src = {src.shape}, tgt = {tgt.shape}, rank={rank}')

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=fp16): 
            if args.num_gpus > 1:
                parallel_model = DDP(model,device_ids = [rank],output_device = rank)
                outputs = parallel_model(src,src_lens)
            else:
                outputs = model(src,src_lens)

            loss = criterion(outputs, tgt)
            running_loss += loss.item()

        curr_tgt = tgt.detach().cpu().numpy()
        curr_pred = torch.max(outputs.detach().cpu(),dim=1)

        running_tgt.append(curr_tgt)
        running_pred.append(curr_pred.indices)

        loss.backward()
        optimizer.step()
        
        if i>0 and i % args.report_every == 0 and rank == 0:
            print('[ %5d] loss: %.3f' %
                  (i+1, running_loss / args.report_every))
           
            val_loss,val_accuracy = validate(model,valid_iterator,criterion)

            true = np.concatenate(running_tgt)
            pred = np.concatenate(running_pred)
            running_accuracy = accuracy_score(true,pred)

            with open(log_file,'a') as outFile:
                save_time = time.time()
                entry =  "{},{},{},{},{},{}".format(save_time,i,running_accuracy,running_loss,val_accuracy,val_loss)
                outFile.write(entry+"\n")

            saver.save(i, moving_average=None)
            running_loss = 0.0
            running_tgt = []
            running_pred = []

    if rank == 0:
        print('Finished Training')
        torch.distributed.destroy_process_group()

def validate(model,iterator,criterion):

    # put in evaluation mode
    model.eval()

    preds = []
    truth = []
    running_loss = 0.0
    fp16 = True

    with torch.no_grad(),torch.cuda.amp.autocast(enabled=fp16): 
        for i,batch in enumerate(iterator):
            src, src_lens = batch.src
            tgt = torch.squeeze(batch.tgt,dim=0)
            tgt = torch.squeeze(tgt,dim=1)
            
            outputs = model(src,src_lens)
            loss = criterion(outputs,tgt)
            running_loss += loss

            pred = torch.max(outputs.detach().cpu(),dim=1)
            preds.append(pred.indices)
            truth.append(tgt.detach().cpu())
    
    preds = np.concatenate(preds)
    truth = np.concatenate(truth)

    # return to training mode
    model.train()

    return running_loss,accuracy_score(truth,preds)

def restore_transformer_classifier(checkpoint,n_input_classes,n_output_classes,args,reset_generator=False):
    ''' Restore a Transformer model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''

    model = make_transformer_classifier(n_input_classes,
                                        n_output_classes,
                                        n_enc=args.n_enc_layers,
                                        model_dim=args.model_dim,
                                        max_rel_pos=args.max_rel_pos)
    
    model.load_state_dict(checkpoint['model'],strict = False)
    
    if reset_generator:
        generator = Generator(args.model_dim,n_output_classes)
        for p in generator.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        model.generator = generator
    else: 
        model.generator.load_state_dict(checkpoint['generator'])
    
    return model

def train(args):

    # controls pseudorandom shuffling and partitioning of dataset
    seed = 65

    df_train = pd.read_csv(args.train,sep='\t')
    df_val = pd.read_csv(args.val,sep='\t')
    
    # convert to torchtext.Dataset
    train_dataset,val_dataset = dataset_from_df([df_train.copy(),df_val.copy()],mode="ENC_only")
    fields = train_dataset.fields
    n_input_classes = len(fields['src'].vocab)
    n_output_classes = len(fields['tgt'].vocab)

    device = "cpu"
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,map_location = "cpu")
        model = restore_transformer_classifier(checkpoint,n_input_classes,n_output_classes,args)

    else:
        model = make_transformer_classifier(n_input_classes,
                                            n_output_classes,
                                            n_enc=args.n_enc_layers,
                                            model_dim=args.model_dim,
                                            max_rel_pos=args.max_rel_pos)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# trainable parameters = {}".format(human_format(num_params)))

    if args.num_gpus > 1:
        print("Multi-GPU training")
        torch.multiprocessing.spawn(train_helper, nprocs=args.num_gpus, args=(df_train,df_val,model,args,seed))
    elif args.num_gpus > 0:
        print("Single-GPU training")
        train_helper(0,args,model,seed)
    else:
        print("CPU training")
        train_helper(0,args,model,seed)
        
    print("Training complete")

if __name__ == "__main__":

    args = parse_args()
    train(args)
    
