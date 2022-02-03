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

from captum.attr import IntegratedGradients,DeepLift,DeepLiftShap,Saliency,InputXGradient,FeatureAblation
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from torchtext.data import Dataset

from torch.nn.parallel import DistributedDataParallel as DDP

from bioseq2seq.modules import PositionalEncoding
from bioseq2seq.bin.translate import make_vocab, restore_transformer_model
from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition


class PredictionWrapper(torch.nn.Module):
    
    def __init__(self,model,softmax):
        
        super(PredictionWrapper,self).__init__()
        self.model = model
        self.softmax = softmax 

    def forward(self,src,src_lens,decoder_input,batch_size):

        src = src.transpose(0,1)
        src, enc_states, memory_bank, src_lengths, enc_attn = self.run_encoder(src,src_lens,batch_size)

        self.model.decoder.init_state(src,memory_bank,enc_states)
        memory_lengths = src_lens

        scores, attn = self.decode_and_generate(
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=0)

        classes = scores
        probs = torch.exp(scores)
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
        
        scores = self.model.generator(dec_out.squeeze(0),softmax=self.softmax)
        return scores,attn

class FeatureAttributor:

    def __init__(self,model,device,vocab,method,rank=0,world_size=1,softmax=True):
        
        self.device = device
        self.model = model
        self.model.eval()
        self.model.zero_grad()

        if method == "ig":
            self.run_fn = self.run_integrated_gradients
        elif method == "deeplift":
            self.run_fn = self.run_deeplift
        elif method == "inputxgradient":
            self.run_fn = self.run_inputXgradient
        elif method == "ISM":
            self.run_fn = self.run_ISM
        
        self.rank = rank
        self.world_size=world_size

        self.sos_token = vocab['tgt'].vocab['<sos>']
        self.pc_token = vocab['tgt'].vocab['<PC>']
        self.src_vocab = vocab['src'].vocab

        self.average = None
        self.nucleotide = None

        self.positional = PositionalEncoding(dropout=0,dim=64).to(self.device)
         
        if method != "ISM":
            self.interpretable_emb = configure_interpretable_embedding_layer(self.model,'encoder.embeddings')
        self.predictor = PredictionWrapper(self.model,softmax)
   
    def old_zero_embed(self,src):

        src_size = list(src.size())
        baseline_emb = torch.zeros(size=(src_size[0],src_size[1],128),dtype=torch.float).to(self.device)
        return baseline_emb
    
    def zero_embed(self,src):

        src_size = list(src.size())
        baseline_emb = torch.zeros(size=(src_size[0],src_size[1],64),dtype=torch.float).to(self.device)
        baseline_emb = baseline_emb.permute(1,0,2)
        baseline_emb = self.positional(baseline_emb)
        baseline_emb = baseline_emb.permute(1,0,2)
        return baseline_emb

    def src_embed(self,src):

        src_emb = self.interpretable_emb.indices_to_embeddings(src.permute(1,0,2))
        src_emb = src_emb.permute(1,0,2)
        return src_emb 

    def old_nucleotide_embed(self,src,nucleotide):

        src_size = list(src.size())
        i=self.src_vocab[nucleotide]
        test = torch.tensor([[[i]]]).to(self.device)
        emb = self.interpretable_emb.indices_to_embeddings(test)
        baseline_emb = emb.repeat(*src_size)
        baseline_emb = self.positional(baseline_emb)
        return baseline_emb

    def nucleotide_embed(self,src,nucleotide):
       
        src_size = list(src.size())
        # retrieve embedding from torchtext
        n = self.src_vocab[nucleotide]
        # copy across length
        test =  n*torch.ones_like(src).to(self.device)
        baseline_emb = self.interpretable_emb.indices_to_embeddings(test.permute(1,0,2))
        baseline_emb = baseline_emb.permute(1,0,2)
        return baseline_emb

    def average_embed(self,src):

        tensor_list = []
        # gather all nucleotide embeddings
        for nuc in ['A','C','G','T']:
            nuc_emb = self.nucleotide_embed(src,nuc)
            tensor_list.append(nuc_emb)
        # find mean
        stack = torch.stack(tensor_list,dim=0)
        baseline_emb = torch.mean(stack,dim=0)
        return baseline_emb

    def old_average_embed(self,src):

        tensor_list = []

        for nuc in ['A','G','C','T']:
            i = self.src_vocab[nuc]
            test = torch.tensor([[[i]]]).to(self.device)
            emb = self.interpretable_emb.indices_to_embeddings(test)
            tensor_list.append(torch.squeeze(emb,dim=0))

        summary = torch.mean(torch.stack(tensor_list,dim=0),dim=0)
        summary = torch.unsqueeze(summary,dim=0)
        average = summary

        src_size = list(src.size())
        baseline_emb = average.repeat(*src_size)
        baseline_emb = self.positional(baseline_emb)
        return baseline_emb

    def decoder_input(self,batch_size):

        return self.sos_token * torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)
   
    def run(self,savefile,val_iterator,target_pos,baseline):

        self.run_fn(savefile,val_iterator,target_pos,baseline)

    def run_deeplift(self,savefile,val_iterator,target_pos,baseline):

        dl = DeepLift(self.predictor)
       
        with open(savefile,'w') as outFile:
            for batch in tqdm.tqdm(val_iterator):
                ids = batch.id
                src, src_lens = batch.src
                src = src.transpose(0,1)
                
                # can only do one batch at a time
                batch_size = batch.batch_size
                for j in range(batch_size):
                    
                    curr_src = torch.unsqueeze(src[j,:,:],0)
                    curr_src_embed = self.src_embed(curr_src)
                    
                    if baseline == "zero":
                        baseline_embed = self.zero_embed(curr_src)
                    elif baseline == "avg":
                        baseline_embed = self.average_embed(curr_src)
                    elif baseline in ['A','C','G','T']:
                        baseline_embed = self.nucleotide_embed(curr_src,baseline)
                    else:
                        raise ValueError('Invalid IG baseline given')

                    decoder_input = self.decoder_input(1) 
                    curr_ids = batch.id
                    curr_tgt = batch.tgt[target_pos,j,:]
                    curr_tgt = torch.unsqueeze(curr_tgt,0)
                    curr_tgt = torch.unsqueeze(curr_tgt,2)

                    curr_src_lens = torch.max(src_lens)
                    curr_src_lens = torch.unsqueeze(curr_src_lens,0)
                    pred_classes = self.predictor(curr_src_embed,curr_src_lens,decoder_input,1)
                    pred,answer_idx = pred_classes.data.cpu().max(dim=-1)
                    pc_class = torch.tensor([[[self.pc_token]]]).to(self.device)
                    
                    n_steps = 500
                    saved_src = np.squeeze(np.squeeze(curr_src.detach().cpu().numpy(),axis=0),axis=1).tolist()
                    saved_src = "".join([self.src_vocab.itos[x] for x in saved_src])
                     
                    attributions,convergence_delta = dl.attribute(inputs=curr_src_embed,
                                                target=pc_class,
                                                baselines=baseline_embed,
                                                return_convergence_delta=True,
                                                additional_forward_args = (curr_src_lens,decoder_input,1))
                    
                    attributions = np.squeeze(attributions.detach().cpu().numpy(),axis=0)
                    pct_error = convergence_delta.item() / np.sum(attributions)
                    summed = np.sum(attributions,axis=1)
                    normed = np.linalg.norm(attributions,2,axis=1)

                    # MDIG
                    if baseline in ['A','C','G','T']:
                        summed = -summed
                        normed = -normed

                    true_len = len(saved_src)
                    summed_attr = ['{:.3e}'.format(x) for x in summed.tolist()[:true_len]]
                    normed_attr = ['{:.3e}'.format(x) for x in normed.tolist()[:true_len]]
                    unreduced_attr = ['{:.3e}'.format(x) for x in attributions.ravel().tolist()]

                    entry = {"ID" : ids[j] , "summed_attr" : summed_attr, "normed_attr" : normed_attr, "src" : saved_src}
                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")
    '''
    def run_deeplift(self,savefile,val_iterator,target_pos,baseline):

        dl = DeepLift(self.predictor)
        
        with open(savefile,'w') as outFile:
            
            for batch in tqdm.tqdm(val_iterator):
                ids = batch.id
                src, src_lens = batch.src
                src = src.transpose(0,1)
                batch_size = batch.batch_size
                
                src_embed = self.src_embed(src)
                if baseline == "zero":
                    baseline_embed = self.zero_embed(src)
                elif baseline == "avg":
                    baseline_embed = self.average_embed(src)
                elif baseline in ['A','C','G','T']:
                    baseline_embed = self.nucleotide_embed(src,baseline)
                else:
                    raise ValueError('Invalid IG baseline given')
                
                decoder_input = self.decoder_input(batch_size)
                pred_classes = self.predictor(src_embed,src_lens,decoder_input,batch_size)
                pred,answer_idx = pred_classes.data.cpu().max(dim=-1)
                pc_class = torch.tensor([[[self.pc_token]]]).to(self.device)
                print(f'src_embed={src_embed.shape}, src={src.shape}, baseline_embed={baseline_embed.shape}')
                attributions = dl.attribute(inputs=src_embed,
                                            target=pc_class,
                                            baselines=baseline_embed,
                                            return_convergence_delta=False,
                                            additional_forward_args = (src_lens,decoder_input,batch_size))
                
                attributions = attributions.detach().cpu().numpy()
                saved_src = src.detach().cpu().numpy()
                summed = np.sum(attributions,axis=2)
                normed = np.linalg.norm(attributions,2,axis=2)

                # MDIG
                if baseline in ['A','C','G','T']:
                    summed = -summed
                    normed = -normed

                for j in range(batch_size):
                    curr_saved_src = "".join([self.src_vocab.itos[x] for x in saved_src[j,:,0]])
                    curr_saved_src = curr_saved_src.split('<pad>')[0]
                    
                    true_len = len(curr_saved_src)
                    summed_attr = ['{:.3e}'.format(x) for x in summed[j,:].tolist()[:true_len]]
                    normed_attr = ['{:.3e}'.format(x) for x in normed[j,:].tolist()[:true_len]]
                    entry = {"ID" : ids[j] , "summed_attr" : summed_attr, "normed_attr" : normed_attr, "src" : curr_saved_src}

                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")
    ''' 
    def run_integrated_gradients(self,savefile,val_iterator,target_pos,baseline):

        global_attr = True
        ig = IntegratedGradients(self.predictor,multiply_by_inputs=global_attr)
       
        with open(savefile,'w') as outFile:
            for batch in tqdm.tqdm(val_iterator):
                ids = batch.id
                src, src_lens = batch.src
                src = src.transpose(0,1)
                
                # can only do one batch at a time
                batch_size = batch.batch_size
                for j in range(batch_size):
                    
                    curr_src = torch.unsqueeze(src[j,:,:],0)
                    curr_src_embed = self.src_embed(curr_src)
                    
                    if baseline == "zero":
                        baseline_embed = self.zero_embed(curr_src)
                    elif baseline == "avg":
                        baseline_embed = self.average_embed(curr_src)
                    elif baseline in ['A','C','G','T']:
                        baseline_embed = self.nucleotide_embed(curr_src,baseline)
                    else:
                        raise ValueError('Invalid IG baseline given')

                    decoder_input = self.decoder_input(1) 
                    curr_ids = batch.id
                    curr_tgt = batch.tgt[target_pos,j,:]
                    curr_tgt = torch.unsqueeze(curr_tgt,0)
                    curr_tgt = torch.unsqueeze(curr_tgt,2)

                    curr_src_lens = torch.max(src_lens)
                    curr_src_lens = torch.unsqueeze(curr_src_lens,0)
                    pred_classes = self.predictor(curr_src_embed,curr_src_lens,decoder_input,1)
                    pred,answer_idx = pred_classes.data.cpu().max(dim=-1)
                    print(pred,answer_idx)
                    '''
                    pc_class = torch.tensor([[[self.pc_token]]]).to(self.device)
                    
                    n_steps = 20
                    saved_src = np.squeeze(np.squeeze(curr_src.detach().cpu().numpy(),axis=0),axis=1).tolist()
                    saved_src = "".join([self.src_vocab.itos[x] for x in saved_src])
                    saved_src = saved_src.split('<pad>')[0]

                    attributions,convergence_delta = ig.attribute(inputs=curr_src_embed,
                                                target=pc_class,
                                                baselines=baseline_embed,
                                                n_steps=n_steps,
                                                internal_batch_size=2,
                                                return_convergence_delta=True,
                                                additional_forward_args = (curr_src_lens,decoder_input,1))
                   
                    attributions = np.squeeze(attributions.detach().cpu().numpy(),axis=0)
                    pct_error = convergence_delta.item() / np.sum(attributions)
                   
                    summed = np.sum(attributions,axis=1)
                    normed = np.linalg.norm(attributions,2,axis=1)

                    # MDIG
                    if baseline in ['A','C','G','T']:
                        summed = -summed
                        normed = -normed

                    true_len = len(saved_src)
                    summed_attr = ['{:.3e}'.format(x) for x in summed.tolist()[:true_len]]
                    normed_attr = ['{:.3e}'.format(x) for x in normed.tolist()[:true_len]]

                    entry = {"ID" : ids[j] , "summed_attr" : summed_attr, "normed_attr" : normed_attr, "src" : saved_src}
                    summary = json.dumps(entry)

                    outFile.write(summary+"\n")
                    '''
    def run_inputXgradient(self,savefile,val_iterator,target_pos,baseline):

        sl = InputXGradient(self.predictor)
        
        with open(savefile,'w') as outFile:
            
            for batch in tqdm.tqdm(val_iterator):
                ids = batch.id
                src, src_lens = batch.src
                src = src.transpose(0,1)
                batch_size = batch.batch_size
                
                src_embed = self.src_embed(src)
                decoder_input = self.decoder_input(batch_size)
                pred_classes = self.predictor(src_embed,src_lens,decoder_input,batch_size)
                pred,answer_idx = pred_classes.data.cpu().max(dim=-1)
                pc_class = torch.tensor([[[self.pc_token]]]).to(self.device)

                attributions = sl.attribute(inputs=src_embed,target=pc_class,additional_forward_args = (src_lens,decoder_input,batch_size))
                attributions = attributions.detach().cpu().numpy()
                saved_src = src.detach().cpu().numpy()

                summed = np.sum(attributions,axis=2)
                normed = np.linalg.norm(attributions,2,axis=2)
                print(summed.shape,normed.shape)
                # MDIG
                if baseline in ['A','C','G','T']:
                    summed = -summed
                    normed = -normed

                for j in range(batch_size):
                    curr_saved_src = "".join([self.src_vocab.itos[x] for x in saved_src[j,:,0]])
                    curr_saved_src = curr_saved_src.split('<pad>')[0]
                                
                    true_len = len(curr_saved_src)
                    summed_attr = ['{:.3e}'.format(x) for x in summed[j,:].tolist()[:true_len]]
                    normed_attr = ['{:.3e}'.format(x) for x in normed[j,:].tolist()[:true_len]]
                    entry = {"ID" : ids[j] , "summed_attr" : summed_attr, "normed_attr" : normed_attr, "src" : curr_saved_src}

                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")

    def run_ISM(self,savefile,val_iterator,target_pos,baseline):

        ism = FeatureAblation(self.predictor)
        
        with open(savefile,'w') as outFile:
            
            for batch in tqdm.tqdm(val_iterator):
                ids = batch.id
                src, src_lens = batch.src
                src = src.transpose(0,1)
                batch_size = batch.batch_size
                
                for j in range(batch_size):
                    curr_src = torch.unsqueeze(src[j,:,:],0)
                    curr_src_lens = torch.max(src_lens)
                    curr_src_lens = torch.unsqueeze(curr_src_lens,0)
                    decoder_input = self.decoder_input(1) 
                    curr_ids = batch.id
                    curr_tgt = batch.tgt[target_pos,j,:]
                    curr_tgt = torch.unsqueeze(curr_tgt,0)
                    curr_tgt = torch.unsqueeze(curr_tgt,2)
                    pred_classes = self.predictor(curr_src,curr_src_lens,decoder_input,batch_size)
                    pred,answer_idx = pred_classes.data.cpu().max(dim=-1)
                    pc_class = torch.tensor([[[self.pc_token]]]).to(self.device)
                    
                    # save computational time by masking non-mutated locations to run simultaneously 
                    base = self.src_vocab[baseline]
                    baseline_class = torch.tensor([[[base]]]).to(self.device)
                    baseline_tensor = base*torch.ones_like(curr_src).to(self.device)
                    num_total_el = torch.numel(curr_src)
                    mask = torch.arange(1,num_total_el+1).reshape_as(curr_src).to(self.device)
                    unchanged_indices = curr_src == baseline_class
                    feature_mask = torch.where(unchanged_indices,0,mask)  
                    
                    '''
                    attributions = ism.attribute(inputs=src_embed,baselines=baseline_embed,
                                        target=pc_class,feature_mask=feature_mask,
                                        additional_forward_args = (src_lens,decoder_input,batch_size))
                    '''
                    attributions = ism.attribute(inputs=curr_src,
                                            baselines=baseline_tensor,
                                            target=pc_class,
                                            feature_mask=None,
                                            additional_forward_args = (curr_src_lens,decoder_input,1))
                    
                    attributions = attributions.detach().cpu().numpy().ravel().tolist()
                    saved_src = curr_src.detach().cpu().numpy().ravel().tolist()
                    
                    curr_saved_src = "".join([self.src_vocab.itos[x] for x in saved_src])
                    curr_saved_src = curr_saved_src.split('<pad>')[0]
                    true_len = len(curr_saved_src)
                    ism_attr = [f'{x:.3e}' for x in attributions[:true_len]]
                    
                    entry = {"ID" : ids[j] , "ism_attr" : ism_attr,"src" : curr_saved_src}
                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")
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
    parser.add_argument("--baseline",default="zero", help="zero|avg|A|C|G|T")
    parser.add_argument("--dataset",default="validation",help="train|test|validation")
    parser.add_argument("--name",default = "temp")
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "Number of GPUs to use on node")
    parser.add_argument("--address",default =  "127.0.0.1",help = "IP address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "Port for master process in distributed training")
    
    return parser.parse_args()

def run_helper(rank,args,model,vocab,use_splits=False):
    
    random_seed = 65
    random.seed(random_seed)
    random_state = random.getstate()

    df = pd.read_csv(args.input,sep="\t")
    df["CDS"] = ["-1" for _ in range(df.shape[0])]
    print(vocab['tgt'].vocab.stoi)
    
    dataset = dataset_from_df([df],mode=args.inference_mode,saved_vocab=vocab)[0]
    max_tokens_in_batch = 2000
    device = "cpu"
    savefile = "{}.{}.rank_{}".format(args.name,args.attribution_mode,rank)
    
    apply_softmax = False if args.attribution_mode == 'deeplift' else True
    
    if args.num_gpus > 0: # GPU training
        # One CUDA device per process
        device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(device)
        model.cuda()

    if args.num_gpus > 1:
        splits = [1.0/args.num_gpus for _ in range(args.num_gpus)]
        dev_partitions = partition(dataset,split_ratios = splits,random_state = random_state)
        local_slice = dev_partitions[rank]
        
        data = [local_slice.examples[i] for i in range(50)]
        local_slice = Dataset(data, local_slice.fields)

        # iterator over evaluation batches
        val_iterator = iterator_from_dataset(local_slice,max_tokens_in_batch,device,train=False)
        attributor = FeatureAttributor(model,device,vocab,args.attribution_mode,rank=rank,world_size=args.num_gpus,softmax=apply_softmax)

    else:
        data = [dataset.examples[i] for i in range(50)]
        dataset = Dataset(data, dataset.fields)
        attributor = FeatureAttributor(model,device,vocab,args.attribution_mode,softmax=apply_softmax)
        val_iterator = iterator_from_dataset(dataset,max_tokens_in_batch,device,train=False)

    target_pos = 1

    attributor.run(savefile,val_iterator,target_pos,args.baseline)

def run_attribution(args,device):
    
    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
    vocab = checkpoint['vocab']
    model = restore_transformer_model(checkpoint,device,options)

    if not options is None:
        model_name = ""
        print("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
        for k,v in vars(options).items():
            print(k,v)
 
    if args.num_gpus > 1:
        torch.multiprocessing.spawn(run_helper, nprocs=args.num_gpus, args=(args,model,vocab))
    elif args.num_gpus > 0:
        run_helper(0,args,model,vocab)
    else:
        run_helper(0,args,model,vocab)

if __name__ == "__main__": 

    warnings.filterwarnings("ignore")
    args = parse_args()
    device = "cpu"
    run_attribution(args,device)
