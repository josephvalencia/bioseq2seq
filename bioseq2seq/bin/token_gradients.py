#!/usr/bin/env python
import torch
import argparse
import pandas as pd
import random
import time
import numpy as np
import json

from captum.attr import LayerGradientXActivation, LayerDeepLift, LayerIntegratedGradients

from bioseq2seq.inputters import TextDataReader , get_fields
from bioseq2seq.translate.transparent_translator import TransparentTranslator
from bioseq2seq.translate import GNMTGlobalScorer
from bioseq2seq.bin.translate import make_vocab, restore_transformer_model
from bioseq2seq.bin.batcher import dataset_from_df, iterator_from_dataset, partition,train_test_val_split
from bioseq2seq.bin.batcher import train_test_val_split

def parse_args():

    """ Parse required and optional configuration arguments"""

    parser = argparse.ArgumentParser()

    # optional flags
    parser.add_argument("--verbose",action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--stats-output","--st",help = "Name of file for saving evaluation statistics")
    parser.add_argument("--translation-output","--o",help = "Name of file for saving predicted translations")
    parser.add_argument("--checkpoint", "--c",help="ONMT checkpoint (.pt)")
    parser.add_argument("--mode",default = "combined")
    parser.add_argument("--name",default = "temp")

    # translate optional args
    parser.add_argument("--beam_size","--b",type = int, default = 8, help ="Beam size for decoding")
    parser.add_argument("--n_best", type = int, default = 4, help = "Number of beams to wait for")
    parser.add_argument("--alpha","--a",type = float, default = 1.0)

    return parser.parse_args()

def run_encoder(model,src,src_lengths,batch_size):

    enc_states, memory_bank, src_lengths, enc_attn = model.encoder(src,src_lengths,grad_mode=True)

    if src_lengths is None:
        assert not isinstance(memory_bank, tuple), \
            'Ensemble decoding only supported for text data'
        src_lengths = torch.Tensor(batch_size) \
                            .type_as(memory_bank) \
                            .long() \
                            .fill_(memory_bank.size(0))

    return src, enc_states, memory_bank, src_lengths,enc_attn

def decode_and_generate(
            model,
            decoder_in,
            memory_bank,
            memory_lengths,
            step=None):

        dec_out, dec_attn = model.decoder(decoder_in, memory_bank, memory_lengths=memory_lengths, step=step,grad_mode=True)

        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        log_probs = model.generator(dec_out.squeeze(0))

        return log_probs, attn

def prepare_input(src,batch_size,sos_token,pad_token,device):

    decoder_input = sos_token * torch.ones(size=(1,batch_size,1),dtype = torch.long).to(device)
    baseline = pad_token * torch.ones_like(src,dtype=torch.long)

    return decoder_input,baseline

def predict_first_token(src,src_lens,decoder_input,batch_size,model,device):
    
    src = src.transpose(0,1)

    #if src.size(1) ==1:
    #    print(src[:,0,0])

    src, enc_states, memory_bank, src_lengths, enc_attn = run_encoder(model,src,src_lens,batch_size)

    model.decoder.init_state(src,memory_bank,enc_states)
    memory_lengths = src_lens

    log_probs, attn = decode_and_generate(
            model,
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=0)

    classes = log_probs

    if classes.size(0) ==1:
        probs = torch.exp(log_probs)
        probs_list = probs.tolist()[0]

        if src.sum().item() / torch.numel(src) == 1:
            print("Baseline(anybase)")
        
        print("P(coding) = {} , P(noncoding) = {}".format(probs_list[24],probs_list[25]))
        # pred_idx = torch.max(probs,dim = 1)
        # print("Predicted class:",pred_idx)
    
    return classes

def collect_token_attribution(args,device):
    
    random_seed = 65
    random.seed(random_seed)
    state = random.getstate()

    data = pd.read_csv(args.input,sep="\t")
    data["CDS"] = ["-1" for _ in range(data.shape[0])]

    checkpoint = torch.load(args.checkpoint,map_location = device)

    model = restore_transformer_model(checkpoint,device)
    model.eval()
    model.zero_grad()

    vocab = checkpoint['vocab']
    # replicate splits
    df_train,df_test,df_dev = train_test_val_split(data,1000,random_seed)
    train,test,dev = dataset_from_df(df_train.copy(),df_test.copy(),df_dev.copy(),mode=args.mode,saved_vocab=vocab)
     
    max_tokens_in_batch = 1000
    num_gpus = 1

    # GPU training
    torch.cuda.set_device(device)
    model.cuda()
    
    val_iterator = iterator_from_dataset(dev,max_tokens_in_batch,device,train=False)
    
    ig = LayerIntegratedGradients(predict_first_token, model.encoder.embeddings)
    
    sos_token = checkpoint['vocab']['tgt'].vocab['<sos>']
    pad_token = checkpoint['vocab']['src'].vocab['<pad>']

    savefile = args.name + ".attr"

    target_pos = 0

    with open(savefile,'w') as outFile:    

        for i,batch in enumerate(val_iterator):

            ids = batch.id
            src,src_lens = batch.src
            src = src.transpose(0,1)
            
            # can only do one batch at a time
            batch_size = batch.batch_size

            for j in range(batch_size):
                
                curr_src = torch.unsqueeze(src[j,:,:],0)
                decoder_input, baseline = prepare_input(curr_src,1,sos_token,pad_token,device)
                
                curr_tgt = batch.tgt[target_pos,j,:]
                curr_tgt = torch.unsqueeze(curr_tgt,0)
                curr_tgt = torch.unsqueeze(curr_tgt,2)

                curr_src_lens = torch.max(src_lens)
                curr_src_lens = torch.unsqueeze(curr_src_lens,0)
                
                print(ids[j])
                
                attributions = ig.attribute(inputs=curr_src,
                                target=curr_tgt,
                                baselines = baseline,
                                internal_batch_size = 6,
                                return_convergence_delta=False,
                                additional_forward_args=(curr_src_lens,decoder_input,1,model,device))
                
                attributions = np.squeeze(attributions.detach().cpu().numpy(),axis=0)        
                #attributions = np.linalg.norm(attributions,2,axis=1)
                attributions = np.sum(attributions,axis=1)
                attr = [round(x,3) for x in attributions.tolist()]
                
                entry = {"ID" : ids[j] , "attr" : attr}
                summary = json.dumps(entry)
                outFile.write(summary+"\n")
            
if __name__ == "__main__": 

    args = parse_args()
    machine = "cuda:3"
    collect_token_attribution(args,machine)