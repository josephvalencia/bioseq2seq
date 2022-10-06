'''Provides for gradient based interpretation at the level of the embedding vectors using the Captum library'''
import torch
import pandas as pd
from torch import nn
import numpy as np
import json
import tqdm
from scipy import stats, signal
import os,re
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

def getLongestORF(mRNA):
    ORF_start = -1
    ORF_end = -1
    longestORF = 0
    for startMatch in re.finditer('ATG',mRNA):
        remaining = mRNA[startMatch.start():]
        if re.search('TGA|TAG|TAA',remaining):
            for stopMatch in re.finditer('TGA|TAG|TAA',remaining):
                ORF = remaining[0:stopMatch.end()]
                if len(ORF) % 3 == 0:
                    if len(ORF) > longestORF:
                        ORF_start = startMatch.start()
                        ORF_end = startMatch.start()+stopMatch.end()
                        longestORF = len(ORF)
                    break
    return ORF_start,ORF_end

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
        dense_embed_layer = OneHotToEmbedding(old_embedding)
        embedding_modulelist[0] = dense_embed_layer
        self.predictor = PredictionWrapper(model,softmax)
        
        super().__init__(model,device,vocab,tgt_class,softmax=softmax,sample_size=sample_size,batch_size=batch_size)
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        raise NotImplementedError

class OneHotSalience(OneHotGradientAttribution):

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            
            tscript = transcript_names[ids[0]]
            if tscript.startswith('XR') or tscript.startswith('NR'):
                continue
            
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            onehot_src = self.onehot_embed_layer(src) 
            
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size,batch.tgt[:10,:,:]),batch_size)
            probs = torch.nn.functional.softmax(pred_classes)
            #class_score = pred_classes[:,self.class_token]
            #third_pos = batch.tgt[3,0,0].item()
            #third_pos = self.pc_token
            third_pos = 3
            observed_score = pred_classes[:,third_pos]
            #class_score = pred_classes[:,25]
            
            counterfactual = [x for x in range(pred_classes.shape[1]) if x != third_pos]
            counter_idx = torch.tensor(counterfactual,device=pred_classes.device)
            class_score = pred_classes.index_select(1,counter_idx)
            print(class_score)
            input_grad = grad(observed_score-class_score.sum(),onehot_src)[0]
            saved_src = src.detach().cpu().numpy()
            
            for b in range(batch_size):
                tscript = transcript_names[ids[b]]
                true_len = src_lens[b].item()
                saliency = input_grad[b,:,0,:true_len]
                #corrected = saliency - saliency.mean(dim=-1).unsqueeze(dim=1)
                corrected = saliency 
                onehot_batch = onehot_src[b,:,:true_len].squeeze()
                inputxgrad = (onehot_batch * corrected).sum(-1) 
                all_inputxgrad[tscript] = inputxgrad.detach().cpu().numpy() 
                all_grad[tscript] = corrected.detach().cpu().numpy() 
            if i > 20:
                break
        
        inputxgrad_savefile = savefile.replace('grad','inputXgrad')
        print(f'saving {inputxgrad_savefile} and {savefile}')
        np.savez(savefile,**all_grad) 
        np.savez(inputxgrad_savefile,**all_inputxgrad)

class OneHotExpectedGradients(OneHotGradientAttribution):

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

            storage = []
            scores = [] 

            # score true
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size),batch_size)
            true_class_score = pred_classes[:,self.class_token]

            batch_attributions = []
            batch_preds = []                     
            minibatch_size = 16
            decoder_input = self.decoder_input(minibatch_size)
            for y,baseline_batch in enumerate(self.extract_baselines(batch,j,self.sample_size,minibatch_size)):
                # score baselines
                baseline_onehot = self.onehot_embed_layer(baseline_batch) 
                baseline_pred_classes = self.predictor(baseline_onehot,curr_src_lens,decoder_input,minibatch_size)
                base_pred = baseline_pred_classes.detach().cpu()[:,tgt_class]
                batch_preds.append(base_pred)
                # sample along paths 
                alpha = torch.rand(baseline_onehot.shape[0],1,1).to(device=self.device)
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

            saved_src = src.detach().cpu().numpy()
            all_grads = torch.stack(storage,dim=2)
            average_grad = direction * all_grads.mean(dim=2)
            
            # assess IG completeness property
            summed = average_grad.sum()
            diff = true_class_score - base_class_score  
            scores = torch.stack(scores,dim=0)
            print(f'true={true_class_score.item():.3f}, diff={diff.item():.3f},sum = {summed:.3f}, mean(IG) = {scores.mean():.3f}, var(IG) = {scores.var():.3f}')
            
            for b in range(batch_size):
                tscript = transcript_names[ids[b]]
                print(tscript)
                true_len = src_lens[b].item()
                saliency = average_grad[b,:,0,:true_len]
                corrected = saliency 
                onehot_batch = onehot_src[b,:,:true_len].squeeze()
                inputxgrad = (onehot_batch * corrected).sum(-1) 
                all_inputxgrad[tscript] = inputxgrad.detach().cpu().numpy() 
                all_grad[tscript] = corrected.detach().cpu().numpy() 
            if i > 20:
                break
        
        inputxgrad_savefile = savefile.replace('grad','inputXgrad')
        print(f'saving {inputxgrad_savefile} and {savefile}')
        np.savez(savefile,**all_grad) 
        np.savez(inputxgrad_savefile,**all_inputxgrad)

class OneHotIntegratedGradients(OneHotGradientAttribution):

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        count = 0 
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            onehot_src = self.onehot_embed_layer(src) 
            
            tscript = transcript_names[ids[0]]
            if tscript.startswith('XR') or tscript.startswith('NR'):
                continue
            
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size),batch_size)
            class_score = pred_classes[:,self.class_token]
            pc_score = pred_classes[:,self.pc_token]
            nc_score = pred_classes[:,self.nc_token]
            print(pc_score.shape)
            #class_score = pred_classes[:,25]
            
            #input_grad = grad(class_score.sum(),onehot_src)[0]
            input_grad = grad(pc_score[0]-nc_score[0],onehot_src)[0]
            saved_src = src.detach().cpu().numpy()

            storage = []
            n_samples = 50
            mdig = torch.zeros_like(onehot_src,device=onehot_src.device)

            # score true
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size),batch_size)
            true_class_score = pred_classes[:,self.class_token]
            # score baseline
            #for c in range(2,7): 
            #character = torch.nn.functional.one_hot(torch.tensor(c,device=onehot_src.device),num_classes=8)
            character = torch.tensor([0.0,0.0,0.25,0.25,0.25,0.25,0.0,0.0],device=onehot_src.device)
            baseline = character*torch.ones_like(onehot_src,device=onehot_src.device,requires_grad=True)
            pred_classes = self.predictor(baseline,src_lens,self.decoder_input(batch_size),batch_size)
            base_class_score = pred_classes[:,self.class_token]
            direction = onehot_src - baseline

            scores = [] 
            for n in range(n_samples):
                interpolated_src = baseline + (n/n_samples) * direction 
                pred_classes = self.predictor(interpolated_src,src_lens,self.decoder_input(batch_size),batch_size)
                class_score = pred_classes[:,self.class_token]
                input_grad = grad(class_score.sum(),interpolated_src)[0]
                storage.append(input_grad)
                scores.append(class_score)

            saved_src = src.detach().cpu().numpy()
            all_grads = torch.stack(storage,dim=2)
            average_grad = direction * all_grads.mean(dim=2)
            
            # assess IG completeness property
            summed = average_grad.sum()
            diff = true_class_score - base_class_score  
            scores = torch.stack(scores,dim=0)
            print(f'true={true_class_score.item():.3f}, diff={diff.item():.3f}, sum = {summed:.3f}, mean(IG) = {scores.mean():.3f}, var(IG) = {scores.var():.3f}')
            mdig += average_grad

            for b in range(batch_size):
                tscript = transcript_names[ids[b]]
                print(tscript)
                true_len = src_lens[b].item()
                saliency = average_grad[b,:,0,:true_len]
                #saliency = mdig[b,:,0,:true_len]
                corrected = saliency 
                onehot_batch = onehot_src[b,:,:true_len].squeeze()
                inputxgrad = (onehot_batch * corrected).sum(-1) 
                all_inputxgrad[tscript] = inputxgrad.detach().cpu().numpy() 
                all_grad[tscript] = corrected.detach().cpu().numpy() 
            count+=1
            if count == 8:
                break
        
        inputxgrad_savefile = savefile.replace('grad','inputXgrad')
        print(f'saving {inputxgrad_savefile} and {savefile}')
        np.savez(savefile,**all_grad) 
        np.savez(inputxgrad_savefile,**all_inputxgrad)

class OneHotSmoothGrad(OneHotGradientAttribution):

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        count = 0 
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            onehot_src = self.onehot_embed_layer(src) 
            
            tscript = transcript_names[ids[0]]
            if tscript.startswith('XR') or tscript.startswith('NR'):
                continue
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size),batch_size)
            class_score = pred_classes[:,self.class_token]
            input_grad = grad(class_score.sum(),onehot_src)[0]
            saved_src = src.detach().cpu().numpy()

            n_samples = 50
            noise_level = 0.025
            storage = []
            scores = [] 
            
            # score true
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size),batch_size)
            true_class_score = pred_classes[:,self.class_token]
            
            third_pos = batch.tgt[3,0,0].item()
            
            for n in range(n_samples):
                noise = noise_level*torch.rand_like(onehot_src,device=onehot_src.device,requires_grad=True) 
                noisy_src = onehot_src + noise 
                pred_classes = self.predictor(noisy_src,src_lens,self.decoder_input(batch_size),batch_size)
                #class_score = pred_classes[:,third_pos]
                #class_score = pred_classes[:,:self.class_token]
                counterfactual = [x for x in range(pred_classes.shape[1]) if x != self.class_token]
                counter_idx = torch.tensor(counterfactual,device=pred_classes.device)
                class_score = pred_classes.index_select(1,counter_idx)
                p = torch.nn.functional.softmax(class_score)
                input_grad = grad(-class_score.sum(),noisy_src)[0]
                input_grad = input_grad - input_grad.mean(dim=-1).unsqueeze(dim=1)
                storage.append(input_grad)
                scores.append(class_score)

            saved_src = src.detach().cpu().numpy()
            all_grads = torch.stack(storage,dim=2)
            average_grad =  all_grads.mean(dim=2)
            scores = torch.stack(scores,dim=0)
            print(f'true={true_class_score.item():.3f}, mean(SG) = {scores.mean():.3f}, var(SG) = {scores.var():.3f}')
            
            for b in range(batch_size):
                tscript = transcript_names[ids[b]]
                true_len = src_lens[b].item()
                saliency = average_grad[b,:,0,:true_len]
                onehot_batch = onehot_src[b,:,:true_len].squeeze()
                inputxgrad = (onehot_batch*saliency).sum(-1) 
                all_inputxgrad[tscript] = inputxgrad.detach().cpu().numpy() 
                all_grad[tscript] = saliency.detach().cpu().numpy()
            count+=1
            if count == 4:
                break
        
        inputxgrad_savefile = savefile.replace('grad','inputXgrad')
        print(f'saving {inputxgrad_savefile} and {savefile}')
        np.savez(savefile,**all_grad) 
        np.savez(inputxgrad_savefile,**all_inputxgrad)
