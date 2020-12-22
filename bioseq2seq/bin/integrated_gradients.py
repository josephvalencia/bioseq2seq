#!/usr/bin/env python
import torch
import argparse
import pandas as pd
import random
import time
import numpy as np
import os
import json
import tqdm
import math
import scipy
import warnings

from captum.attr import IntegratedGradients,DeepLift, Saliency
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

from torch.nn.parallel import DistributedDataParallel as DDP

from bioseq2seq.modules import PositionalEncoding
from bioseq2seq.bin.translate import make_vocab, restore_transformer_model
from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition,train_test_val_split


class PredictionWrapper(torch.nn.Module):
    
    def __init__(self,model):
        
        super(PredictionWrapper,self).__init__()
        self.model = model
        
    def forward(self,src,src_lens,decoder_input,batch_size):

        src = src.transpose(0,1)
        src, enc_states, memory_bank, src_lengths, enc_attn = self.run_encoder(src,src_lens,batch_size)

        self.model.decoder.init_state(src,memory_bank,enc_states)
        memory_lengths = src_lens

        log_probs, attn = self.decode_and_generate(
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=0)

        classes = log_probs
        probs = torch.exp(log_probs)
        probs_list = probs.tolist()

        return classes

    def run_encoder(self,src,src_lengths,batch_size):

        enc_states, memory_bank, src_lengths, enc_attn = self.model.encoder(src,src_lengths,grad_mode=False)

        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch_size) \
                                .type_as(memory_bank) \
                                .long() \
                                .fill_(memory_bank.size(0))

        return src, enc_states, memory_bank, src_lengths,enc_attn

    def decode_and_generate(self,decoder_in, memory_bank, memory_lengths, step=None):

        dec_out, dec_attn = self.model.decoder(decoder_in,
                                            memory_bank,
                                            memory_lengths=memory_lengths,
                                            step=step,grad_mode=True)

        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        log_probs = self.model.generator(dec_out.squeeze(0))

        return log_probs,attn

class FeatureAttributor:

    def __init__(self,model,device,sos_token,pad_token,vocab,rank=0,world_size=1):
        
        self.device = device
        self.model = model
        self.model.eval()
        self.model.zero_grad()

        self.rank = rank
        self.world_size=world_size

        self.sos_token = sos_token
        self.pad_token = pad_token
        self.vocab = vocab
        self.average = None
        self.nucleotide = None

        self.positional = PositionalEncoding(0,128,10).to(self.device)
        self.interpretable_emb = configure_interpretable_embedding_layer(self.model,'encoder.embeddings')
        self.predictor = PredictionWrapper(self.model)

    def zero_embed(self,src,batch_size):

        decoder_input = self.sos_token * torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)
        
        src_size = list(src.size())
        
        baseline_emb = torch.zeros(size=(src_size[0],src_size[1],128),dtype=torch.float).to(self.device)
        baseline_emb = self.positional(baseline_emb)
        input_emb = self.interpretable_emb.indices_to_embeddings(src)
        
        return decoder_input,baseline_emb,input_emb

    def precompute_average(self):

        tensor_list = []

        for nuc in ['A','G','C','T']:
            i = self.vocab[nuc]
            test = torch.tensor([[[i]]]).to(self.device)
            emb = self.interpretable_emb.indices_to_embeddings(test)
            tensor_list.append(torch.squeeze(emb,dim=0))

        summary = torch.mean(torch.stack(tensor_list,dim=0),dim=0)
        summary = torch.unsqueeze(summary,dim=0)
        self.average = summary

    def precompute_nucleotide(self,nuc):
      
      i = self.vocab[nuc]
      test = torch.tensor([[[i]]]).to(self.device)
      emb = self.interpretable_emb.indices_to_embeddings(test)
      self.nucleotide = emb
    
    def nucleotide_embed(self,src,batch_size):
        
        src_size = list(src.size())
        baseline_emb = self.nucleotide.repeat(*src_size)
        baseline_emb = self.positional(baseline_emb)
        input_emb = self.interpretable_emb.indices_to_embeddings(src)
        
        return baseline_emb,input_emb

    def average_embed(self,src,batch_size):
    
        src_size = list(src.size())
        baseline_emb = self.average.repeat(*src_size)
        baseline_emb = self.positional(baseline_emb)
        input_emb = self.interpretable_emb.indices_to_embeddings(src)

        return baseline_emb,input_emb

    def decoder_input(self,batch_size):

        return self.sos_token * torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)

    def run_deeplift(self,savefile,val_iterator,target_pos,reduction):

        dl = DeepLift(self.predictor)

        self.precompute_average()
        self.precompute_nucleotide(nuc='A')

        with open(savefile,'w') as outFile:
            for batch in tqdm.tqdm(val_iterator):

                ids = batch.id
                src, src_lens = batch.src
                src = src.transpose(0,1)
                batch_size = batch.batch_size
                
                #decoder_input, baseline_embed, src_embed = self.average_embed(src,batch_size)
                baseline_embed, src_embed = self.zero_embed(src,batch_size)
                decoder_input = self.decoder_input(batch_size)
                #decoder_input, baseline_embed, src_embed = self.nucleotide_embed(src,batch_size)

                pred_classes = self.predictor(src_embed,src_lens,decoder_input,batch_size)
                pred,answer_idx = pred_classes.data.max(dim=-1)

                attributions = dl.attribute(inputs=src_embed,
                                            target=answer_idx,
                                            baselines=baseline_embed,
                                            return_convergence_delta=False,
                                            additional_forward_args=(src_lens,decoder_input,batch_size)) 
                
                attributions = attributions.detach().cpu().numpy()

                for j in range(batch_size):
                    curr_attributions = attributions[j,:,:]

                    if reduction == "sum":
                        curr_attributions = np.sum(curr_attributions,axis=1)
                    else: 
                        curr_attributions = np.linalg.norm(curr_attributions,2,axis=1)
                    
                    attr = [round(x,3) for x in curr_attributions.tolist()]
                    
                    entry = {"ID" : ids[j] , "attr" : attr}
                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")
                
        remove_interpretable_embedding_layer(self.model,self.interpretable_emb)

    def run_integrated_gradients(self,savefile,val_iterator,target_pos,reduction):

        ig = IntegratedGradients(self.predictor)
        
        with open(savefile,'w') as outFile:
            for batch in tqdm.tqdm(val_iterator):

                ids = batch.id
                src, src_lens = batch.src
                src = src.transpose(0,1)
                
                # can only do one batch at a time
                batch_size = batch.batch_size

                for j in range(batch_size):
                    
                    curr_src = torch.unsqueeze(src[j,:,:],0)
                    baseline_embed, curr_src_embed, = self.zero_embed(curr_src,1)
                    decoder_input = self.decoder_input(batch_size) 

                    curr_ids = batch.id

                    curr_tgt = batch.tgt[target_pos,j,:]
                    curr_tgt = torch.unsqueeze(curr_tgt,0)
                    curr_tgt = torch.unsqueeze(curr_tgt,2)

                    curr_src_lens = torch.max(src_lens)
                    curr_src_lens = torch.unsqueeze(curr_src_lens,0)

                    pred_classes = self.predictor(curr_src_embed,curr_src_lens,decoder_input,1)
                    pred,answer_idx = pred_classes.data.cpu().max(dim=-1)

                    attributions,convergence_delta = ig.attribute(inputs=curr_src_embed,
                                                target=answer_idx,
                                                baselines=baseline_embed,
                                                n_steps=50,
                                                internal_batch_size=2,
                                                return_convergence_delta=True,
                                                additional_forward_args = (curr_src_lens,decoder_input,1))

                    attributions = np.squeeze(attributions.detach().cpu().numpy(),axis=0)

                    summed  = 1000*np.sum(attributions,axis=1)
                    normed = 1000*np.linalg.norm(attributions,2,axis=1)

                    summed_attr = [round(x,3) for x in summed.tolist()]
                    normed_attr = [round(x,3) for x in normed.tolist()]

                    saved_src = np.squeeze(np.squeeze(curr_src.detach().cpu().numpy(),axis=0),axis=1).tolist()
                    saved_src = "".join([self.vocab.itos[x] for x in saved_src])

                    entry = {"ID" : ids[j] , "summed_attr" : summed_attr, "normed_attr" : normed_attr, "src" : saved_src}

                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")

    def run_salience(self,savefile,val_iterator,target_pos):

        sl = Saliency(self.predictor)
        
        with open(savefile,'w') as outFile:
            
            for batch in tqdm.tqdm(val_iterator):
                ids = batch.id
                src, src_lens = batch.src
                src = src.transpose(0,1)
                batch_size = batch.batch_size
                
                decoder_input = self.decoder_input(batch_size)
                pred_classes = self.predictor(src,src_lens,decoder_input,1)
                pred,answer_idx = pred_classes.data.cpu().max(dim=-1)

                attributions = sl.attribute(inputs=src,target=answer_idx,additional_forward_args = (src_lens,decoder_input,batch_size))
                attributions = np.squeeze(attributions.detach().cpu().numpy(),axis=0)

                attr = [round(x,3) for x in attributions.tolist()]

                saved_src = np.squeeze(np.squeeze(curr_src.detach().cpu().numpy(),axis=0),axis=1).tolist()
                saved_src = "".join([self.vocab.itos[x] for x in saved_src])

                entry = {"ID" : ids[j] , "salience" : attr, "src" : saved_src}

                summary = json.dumps(entry)
                outFile.write(summary+"\n")

    def merge_handler(self,attr,fh):

        # query across all machines for size
        local_size = torch.tensor([attr.numel()],dtype=torch.int64).to(self.device)
        size_list = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        torch.distributed.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)
        
        # pad tensors to maximum length 
        tensor_list = [torch.zeros(size=(max_size,),dtype=torch.float).to(self.device) for _ in range(self.world_size)]
        
        if local_size != max_size:
            padding = float('NaN') *torch.ones(size=(max_size - local_size,)).to(self.device)
            attr = torch.cat((attr, padding), dim=0)
            torch.distributed.all_gather(tensor_list, attr)
        else:
            torch.distributed.all_gather(tensor_list,attr)

        if self.rank == 0:
            attr = torch.stack(tensor_list,dim=1).cpu().numpy()
            attr = attr[~np.isnan(attr)]

            for i in range(self.world_size):
                curr_attr = attr[:,i]
                entry = "{}\t{}\n".format(id,curr_attr)
                fh.write(summary)

def parse_args():

    """ Parse required and optional configuration arguments"""
    parser = argparse.ArgumentParser()
    
    # optional flags
    parser.add_argument("--verbose",action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--checkpoint", "--c",help ="ONMT checkpoint (.pt)")
    parser.add_argument("--inference_mode",default ="combined")
    parser.add_argument("--attribution_mode",default="ig")
    parser.add_argument("--name",default = "temp")
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--reduction",default="norm")
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "Number of GPUs to use on node")
    parser.add_argument("--address",default =  "127.0.0.1",help = "IP address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "Port for master process in distributed training")
    
    return parser.parse_args()

def run_helper(rank,args,model,vocab):
    
    random_seed = 65
    random.seed(random_seed)
    random_state = random.getstate()

    data = pd.read_csv(args.input,sep="\t")
    data["CDS"] = ["-1" for _ in range(data.shape[0])]

    sos_token = vocab['tgt'].vocab['<sos>']
    pad_token = vocab['src'].vocab['<pad>']
    
    # replicate splits
    df_train,df_test,df_dev = train_test_val_split(data,1000,random_seed)
    train,test,dev = dataset_from_df(df_train.copy(),df_test.copy(),df_dev.copy(),mode=args.inference_mode,saved_vocab=vocab)
     
    max_tokens_in_batch = 1000

    device = "cpu"

    savefile = "{}.{}_{}.rank_{}".format(args.name,args.attribution_mode,args.reduction,rank)

    if args.num_gpus > 0: # GPU training
        # One CUDA device per process
        device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(device)
        model.cuda()

    if args.num_gpus > 1:

        splits = [1.0/args.num_gpus for _ in range(args.num_gpus)]
        dev_partitions = partition(dev,split_ratios = splits,random_state = random_state)
        local_slice = dev_partitions[rank]

        # iterator over evaluation batches
        val_iterator = iterator_from_dataset(local_slice,max_tokens_in_batch,device,train=False)
        attributor = FeatureAttributor(model,device,sos_token,pad_token,vocab['src'].vocab,rank=rank,world_size=args.num_gpus)

    else:
        attributor = FeatureAttributor(model,device,sos_token,pad_token,vocab['src'].vocab)
        val_iterator = iterator_from_dataset(dev,max_tokens_in_batch,device,train=False)

    target_pos = 0

    if args.attribution_mode == "ig":
        attributor.run_integrated_gradients(savefile,val_iterator,target_pos,args.reduction)
    elif args.attribution_mode == "deeplift":
        attributor.run_deeplift(savefile,val_iterator,target_pos,args.reduction)
    elif args.attribution_mode == "salience":
        attributor.run_salience(savefile,val_iterator,target_pos)

def run_attribution(args,device):
    
    checkpoint = torch.load(args.checkpoint,map_location = device)
    
    options = checkpoint['opt']
    vocab = checkpoint['vocab']
    model = restore_transformer_model(checkpoint,device,options)

    if args.num_gpus > 1:
        torch.multiprocessing.spawn(run_helper, nprocs=args.num_gpus, args=(args,model,vocab))
        torch.distributed.destroy_process_group()
    elif args.num_gpus > 0:
        run_helper(0,args,model,vocab)
    else:
        run_helper(0,args,model,vocab)
        

if __name__ == "__main__": 

    warnings.filterwarnings("ignore")
    args = parse_args()
    device = "cuda:0"
    run_attribution(args,device)
