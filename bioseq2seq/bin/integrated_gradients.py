#!/usr/bin/env python
import torch
import argparse
import pandas as pd
import random
import time
import numpy as np
import json
import tqdm
import math
import scipy
import warnings

from captum.attr import LayerIntegratedGradients,IntegratedGradients,DeepLift,LayerDeepLift
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

from bioseq2seq.modules import PositionalEncoding
from bioseq2seq.bin.translate import make_vocab, restore_transformer_model
from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition,train_test_val_split
from bioseq2seq.bin.batcher import train_test_val_split

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

    def __init__(self,model,device,sos_token,pad_token,vocab):
        
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.model.zero_grad()

        self.sos_token = sos_token
        self.pad_token = pad_token
        self.vocab = vocab
        self.average = None
        self.nucleotide = None

        self.positional = PositionalEncoding(0,128,10).to(self.device)
        self.interpretable_emb = configure_interpretable_embedding_layer(self.model,'encoder.embeddings')
        self.predictor = PredictionWrapper(self.model)

    def prepare_input(self,src,batch_size):

        decoder_input = self.sos_token * torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)
        baseline = self.pad_token * torch.ones_like(src,dtype=torch.long)

        return decoder_input,baseline

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

      decoder_input = self.sos_token * torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)

      src_size = list(src.size())
      baseline_emb = self.nucleotide.repeat(*src_size)
      baseline_emb = self.positional(baseline_emb)
      input_emb = self.interpretable_emb.indices_to_embeddings(src)

      return decoder_input,baseline_emb,input_emb

    def average_embed(self,src,batch_size):
    
        decoder_input = self.sos_token * torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)
        
        src_size = list(src.size())
        baseline_emb = self.average.repeat(*src_size)
        baseline_emb = self.positional(baseline_emb)
        input_emb = self.interpretable_emb.indices_to_embeddings(src)

        return decoder_input,baseline_emb,input_emb

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
                decoder_input, baseline_embed, src_embed = self.zero_embed(src,batch_size)
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
                    decoder_input, baseline_embed, curr_src_embed, = self.zero_embed(curr_src,1)
                    
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

                    if reduction == "sum":
                        attributions = np.sum(attributions,axis=1)
                    else: 
                        attributions = np.linalg.norm(attributions,2,axis=1)
                    
                    attr = [round(x,3) for x in attributions.tolist()]
                    
                    saved_src = np.squeeze(np.squeeze(curr_src.detach().cpu().numpy(),axis=0),axis=1).tolist()
                    saved_src = "".join([self.vocab.itos[x] for x in saved_src])
                    saved_base = np.squeeze(baseline_embed.detach().cpu().numpy(),axis=0)[0].tolist()
                    
                    entry = {"ID" : ids[j] , "attr" : attr, "src" : saved_src, "baseline" : saved_base}

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
    parser.add_argument("--name",default = "temp")

    return parser.parse_args()

def run_attribution(args,device):
    
    random_seed = 65
    random.seed(random_seed)
    state = random.getstate()

    data = pd.read_csv(args.input,sep="\t")
    data["CDS"] = ["-1" for _ in range(data.shape[0])]

    checkpoint = torch.load(args.checkpoint,map_location = device)
    
    options = checkpoint['opt']
    vocab = checkpoint['vocab']

    sos_token = vocab['tgt'].vocab['<sos>']
    pad_token = vocab['src'].vocab['<pad>']

    model = restore_transformer_model(checkpoint,device,options)
    
    # replicate splits
    df_train,df_test,df_dev = train_test_val_split(data,1000,random_seed)
    train,test,dev = dataset_from_df(df_train.copy(),df_test.copy(),df_dev.copy(),mode=args.inference_mode,saved_vocab=vocab)
     
    # GPU training
    torch.cuda.set_device(device)
    max_tokens_in_batch = 1000

    val_iterator = iterator_from_dataset(dev,max_tokens_in_batch,device,train=False)
    attributor = FeatureAttributor(model,device,sos_token,pad_token,vocab['src'].vocab)

    savefile = args.name + ".attr"
    reduction = "norm"
    target_pos = 0

    if args.attribution_mode == "ig":
        attributor.run_integrated_gradients(savefile,val_iterator,target_pos,reduction)
    elif args.attribution_mode == "deeplift":
        attributor.run_deeplift(savefile,val_iterator,target_pos,reduction)

if __name__ == "__main__": 

    warnings.filterwarnings("ignore")
    args = parse_args()
    machine = "cuda:0"
    run_attribution(args,machine)
