'''Provides for gradient based interpretation at the level of the embedding vectors using the Captum library'''
import torch
import pandas as pd
from torch import nn
import numpy as np
import json
import tqdm
from scipy import stats, signal
import os
from approxISM.embedding import TensorToOneHot, OneHotToEmbedding 
from captum.attr import NoiseTunnel,DeepLiftShap,GradientShap,Saliency,\
        InputXGradient, IntegratedGradients,FeatureAblation
#from functorch import grad, vmap
from torch.autograd import grad
import bioseq2seq.bin.transforms as xfm
from bioseq2seq.bin.transforms import CodonTable, getLongestORF
from base import Attribution, PredictionWrapper
import torch.nn.functional as F
from functools import partial,reduce
from typing import Union

class SynonymousShuffleExpectedGradientsOneHot(Attribution):
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):

        storage = []
        for batch in tqdm.tqdm(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            
            # can only do one batch at a time
            for j in range(batch.batch_size):
                # setup batch elements
                tscript = transcript_names[ids[j]]
                curr_src = src[j,:,:].unsqueeze(0)
                curr_src_embed = self.src_embed(curr_src)
                curr_tgt = batch.tgt[target_pos,j,:].reshape(1,-1,1)
                curr_src_lens = torch.max(src_lens)
                curr_src_lens = curr_src_lens.unsqueeze(0)
                tgt_class = self.class_token
                
                # score original sequence
                pred_classes = self.predictor(curr_src_embed,curr_src_lens,self.decoder_input(1),1)
                src_score =  pred_classes.detach().cpu()[0,tgt_class]
                
                # check scores for non-classification token 
                probs = torch.exp(F.log_softmax(pred_classes.detach()))
                probs_list = probs.reshape(-1).tolist()
                probs_with_labels = list(zip(self.tgt_vocab.stoi.keys(),probs_list))
                good_share = probs_list[self.pc_token] + probs_list[self.nc_token]
                bad_share = 1.0 - good_share
                print(f'bad_share={bad_share}')

                batch_attributions = []
                batch_preds = []                     
                minibatch_size = 16
                decoder_input = self.decoder_input(minibatch_size)
                for y,baseline_batch in enumerate(self.extract_baselines(batch,j,self.sample_size,minibatch_size)):
                    # score baselines
                    baseline_embed = self.src_embed(baseline_batch)
                    baseline_pred_classes = self.predictor(baseline_embed,curr_src_lens,decoder_input,minibatch_size)
                    base_pred = baseline_pred_classes.detach().cpu()[:,tgt_class]
                    batch_preds.append(base_pred)
                    # sample along paths 
                    alpha = torch.rand(baseline_embed.shape[0],1,1).to(device=self.device)
                    direction = curr_src_embed - baseline_embed
                    interpolated = baseline_embed + alpha*direction
                    # calculate gradients  
                    grads = sl.attribute(inputs=interpolated,target=tgt_class,abs=False,
                                        additional_forward_args = (src_lens,decoder_input,minibatch_size))
                    grads = direction * grads
                    grads = grads.detach().cpu()
                    batch_attributions.append(grads)
               
                # take expectation
                batch_attributions = torch.cat(batch_attributions,dim=0)
                attributions = batch_attributions.mean(dim=0).detach().cpu().numpy() 
                # check convergence
                baseline_preds = torch.cat(batch_preds,dim=0)
                mean_baseline_score = baseline_preds.mean()
                diff = src_score - mean_baseline_score
                my_mae = diff - np.sum(attributions)
                    
                saved_src = np.squeeze(np.squeeze(curr_src.detach().cpu().numpy(),axis=0),axis=1).tolist()
                saved_src = "".join([self.src_vocab.itos[x] for x in saved_src])
                saved_src = saved_src.split('<blank>')[0]
              
                # optimize CDS codons using expected gradients scores
                start,end = getLongestORF(saved_src)
                cds = saved_src[start:end] 
                table = CodonTable()
                optimized_cds = ''
                summed = np.sum(attributions,axis=1)
                summed_attr = summed.tolist()[start:end]
                scores = []
                
                opt_mode = 'max'

                for i in range(0,len(cds),3):
                    codon = cds[i:i+3]
                    attr = summed_attr[i:i+3]
                    if opt_mode == 'min':
                        opt_codon = table.synonymous_codon_by_min_score(codon,attr) 
                    elif opt_mode == 'max':
                        opt_codon = table.synonymous_codon_by_max_score(codon,attr) 
                    optimized_cds += opt_codon
                optimized_seq = saved_src[:start] + optimized_cds + saved_src[end:]
                
                # convert sequence to embedding 
                optimized_src = torch.tensor([self.src_vocab.stoi[x] for x in optimized_seq])
                optimized_src = optimized_src.reshape(1,-1,1).to(self.device)
                opt_src_embed = self.src_embed(optimized_src)
               
                # score codon-optimized sequence
                pred_classes = self.predictor(opt_src_embed,curr_src_lens,self.decoder_input(1),1)
                opt_score = pred_classes.data.cpu()[0,self.class_token]
                pct_error = my_mae.item() / diff.item() if diff.item() != 0.0 else 0.0 
               
                # compare with best found from random search
                if opt_mode == 'min':
                    best_baseline_score = baseline_preds.min()
                else:
                    best_baseline_score = baseline_preds.max()

                if self.softmax:
                    src_score = torch.exp(src_score).item()
                    mean_baseline_score = torch.exp(mean_baseline_score).item()
                    best_baseline_score = torch.exp(best_baseline_score).item()
                    opt_score = torch.exp(opt_score).item()
                else:
                    src_score = src_score.item()
                    mean_baseline_score = mean_baseline_score.item()
                    best_baseline_score = best_baseline_score.item()
                    opt_score = opt_score.item()

                entry = {"ID" : tscript ,"pct_approx_error" : pct_error, "original" : src_score, "mean_sampled" \
                        : mean_baseline_score, "best_sampled" : best_baseline_score, "optimized" : opt_score}
                storage.append(entry)
    
        df = pd.DataFrame(storage)
        df.to_csv(savefile,sep='\t')

    def extract_baselines(self,batch,batch_index,num_copies,copies_per_step):

        baselines = []
        baseline_shapes = []
        for i in range(num_copies):
            seq_name = f'src_shuffled_{i}'
            attr = getattr(batch,seq_name)
            baselines.append(attr[0])

        upper = max(num_copies,1)
        for i in range(0,upper,copies_per_step):
            chunk = baselines[i:i+copies_per_step]
            stacked = torch.stack(chunk,dim=0).to(device=batch.src[0].device) 
            yield stacked[:,:,batch_index,:]


def get_module_by_name(parent: Union[torch.Tensor, nn.Module],
                               access_string: str):
    names = access_string.split(sep='.')
    return reduce(getattr, names, parent)

class OneHotGradientAttribution(Attribution):

    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None,times_input=False,smoothgrad=False):
        
        self.smoothgrad = smoothgrad
        # augment Embedding with one hot utilities
        embedding_modulelist = get_module_by_name(model,'encoder.embeddings.make_embedding.emb_luts')
        old_embedding = embedding_modulelist[0] 
        self.onehot_embed_layer = TensorToOneHot(old_embedding)
        dense_embed_layer= OneHotToEmbedding(old_embedding)
        embedding_modulelist[0] = dense_embed_layer
        self.predictor = PredictionWrapper(model,softmax)
        super().__init__(model,device,vocab,tgt_class,softmax=softmax,sample_size=sample_size,batch_size=batch_size)
    
    def predict_sample(self,onehot_src,src_lens,class_token):

        onehot_src = onehot_src.unsqueeze(0)
        src_lens = src_lens.unsqueeze(0)
        pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(1),1)
        # sum assumes no batch dim
        return pred_classes[0,class_token]

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            onehot_src = self.onehot_embed_layer(src) 
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size),batch_size)
            class_score = pred_classes[:,self.class_token]
            input_grad = grad(class_score.sum(),onehot_src)[0]
            saved_src = src.detach().cpu().numpy()
            for b in range(batch_size):
                tscript = transcript_names[ids[b]]
                true_len = src_lens[b].item()
                saliency = input_grad[b,:,0,:true_len]
                corrected = saliency - saliency.mean(dim=-1).unsqueeze(dim=1)
                onehot_batch = onehot_src[b,:,:true_len].squeeze()
                inputxgrad = (onehot_batch * corrected).sum(-1) 
                all_inputxgrad[tscript] = inputxgrad.detach().cpu().numpy() 
                all_grad[tscript] = corrected.detach().cpu().numpy() 
        
        inputxgrad_savefile = savefile.replace('grad','inputXgrad')
        print(f'saving {inputxgrad_savefile} and {savefile}')
        np.savez(savefile,**all_grad) 
        np.savez(inputxgrad_savefile,**all_inputxgrad) 
