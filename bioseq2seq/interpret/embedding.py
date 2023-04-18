'''Provides for gradient based interpretation at the level of the embedding vectors using the Captum library'''
import torch
import pandas as pd
import numpy as np
import json
import tqdm
from scipy import stats, signal

from captum.attr import NoiseTunnel,DeepLiftShap,GradientShap,Saliency,\
        InputXGradient, IntegratedGradients,FeatureAblation

import bioseq2seq.bin.transforms as xfm
from bioseq2seq.bin.transforms import CodonTable, getLongestORF
from base import EmbeddingAttribution
import torch.nn.functional as F
from functools import partial

class EmbeddingIG(EmbeddingAttribution):
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            curr_src_embed = self.src_embed(src)
            batch_size = batch.batch_size
            tscript = transcript_names[ids[0]]

            # score true
            tgt_prefix = batch.tgt[:target_pos,:,:]
            true_logit, true_probs = self.predict_logits(curr_src_embed,src_lens,
                                                    self.decoder_input(batch_size,tgt_prefix),
                                                    batch_size,self.class_token,ratio=True)
            n_samples = 128
            minibatch_size = 16

            # score all single-nuc baselines one by one
            #baseline_embed = self.nucleotide_embed(src,'A')
            baseline_embed = self.rand_embed(src)
            base_class_logit, base_probs = self.predict_logits(baseline_embed,src_lens,
                                                            self.decoder_input(batch_size,tgt_prefix),
                                                            batch_size,self.class_token,ratio=True)
            base_probs = base_probs.squeeze()
            grads = []
            scores = []
            grid = torch.linspace(0,1,n_samples,device=baseline_embed.device) 
            direction = curr_src_embed - baseline_embed
            for n in range(0,n_samples,minibatch_size):
                alpha = grid[n:n+minibatch_size].reshape(minibatch_size,1,1,1) 
                interpolated_src = baseline_embed + alpha*direction 
                logit, probs = self.predict_logits(interpolated_src.squeeze(),src_lens,
                                                    self.decoder_input(minibatch_size,tgt_prefix),
                                                    minibatch_size,self.class_token,ratio=True)
                input_grad = self.input_grads(logit,interpolated_src)
                grads.append(input_grad.detach().cpu())

            # take riemannian sum
            diff = true_logit - base_class_logit
            all_grads = torch.cat(grads,dim=0)
            #print(f'direction = {direction.shape}, all_grads.mean() = {all_grads.mean(dim=0).shape}')
            average_grad = direction.detach().cpu() * all_grads.mean(dim=0)
            #print(f'direction = {direction}')
            #print(f'all_grads.mean = {all_grads.mean(dim=0)}')
            #print(f'average_grad = {average_grad}')
            # assess IG completeness property
            summed = average_grad.sum()
            print(f'diff = {diff}, sum of scores = {summed}')
            print('average_grad',average_grad.shape)  
            attr = average_grad.sum(dim=2)
            tscript = transcript_names[ids[0]]
            true_len = src_lens[0].item()
            saliency = attr[:,:true_len]
            all_grad[tscript] = saliency.detach().cpu().numpy() 
        
        np.savez_compressed(savefile,**all_grad) 

class EmbeddingMDIG(EmbeddingAttribution):
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            curr_src_embed = self.src_embed(src)
            batch_size = batch.batch_size
            tscript = transcript_names[ids[0]]

            # score true
            tgt_prefix = batch.tgt[:target_pos,:,:]
            true_logit, true_probs = self.predict_logits(curr_src_embed,src_lens,
                                                    self.decoder_input(batch_size,tgt_prefix),
                                                    batch_size,self.class_token,ratio=True)
            n_samples = 128
            minibatch_size = 16
            mdig = []

            # score all single-nuc baselines one by one
            for c,base in zip(range(2,6),'ACGT'): 
                baseline_embed = self.nucleotide_embed(src,base)
                base_class_logit, base_probs = self.predict_logits(baseline_embed,src_lens,
                                                                self.decoder_input(batch_size,tgt_prefix),
                                                                batch_size,self.class_token,ratio=True)
                base_probs = base_probs.squeeze()
                grads = []
                scores = []
                grid = torch.linspace(0,1,n_samples,device=baseline_embed.device) 
                direction = curr_src_embed - baseline_embed

                for n in range(0,n_samples,minibatch_size):
                    alpha = grid[n:n+minibatch_size].reshape(minibatch_size,1,1,1) 
                    interpolated_src = baseline_embed + alpha*direction 
                    logit, probs = self.predict_logits(interpolated_src.squeeze(),src_lens,
                                                        self.decoder_input(minibatch_size,tgt_prefix),
                                                        minibatch_size,self.class_token,ratio=True)
                    
                    #probs = probs.squeeze()
                    #print(f'alpha = {alpha.squeeze()}, P(class) = {probs[:,self.class_token]}')
                    input_grad = self.input_grads(logit,interpolated_src)
                    grads.append(input_grad.detach().cpu())

                # take riemannian sum
                diff = true_logit - base_class_logit
                all_grads = torch.cat(grads,dim=0)
                average_grad = direction.detach().cpu() * all_grads.mean(dim=0)
                # assess IG completeness property
                summed = average_grad.sum()
                print(f'base = {base}, diff = {diff}, sum of scores = {summed}')
                ig_summed = average_grad.sum(dim=2)
                mdig.append(ig_summed)

            attr = torch.cat(mdig,dim=0)
            tscript = transcript_names[ids[0]]
            true_len = src_lens[0].item()
            saliency = attr[:,:true_len]
            all_grad[tscript] = saliency.detach().cpu().numpy() 
        
        np.savez_compressed(savefile,**all_grad) 

class SynonymousShuffleExpectedGradients(EmbeddingAttribution):
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):

        sl = Saliency(self.predictor)

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
                print(f'pred_classes = {pred_classes.shape}')
                src_score = pred_classes.detach().cpu()[:,:,tgt_class]

                batch_attributions = []
                batch_preds = []                     
                
                minibatch_size = 16
                decoder_input = self.decoder_input(minibatch_size)
                for y,baseline_batch in enumerate(self.extract_baselines(batch,j,self.sample_size,minibatch_size)):
                    # score baselines
                    baseline_embed = self.src_embed(baseline_batch)
                    baseline_pred_classes = self.predictor(baseline_embed,curr_src_lens,decoder_input,minibatch_size)
                    print('baseline_pred_classes',baseline_pred_classes.shape)
                    base_pred = baseline_pred_classes.detach().cpu()[:,:,tgt_class]
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
                print(f'SCORE DIFF= {diff.item()}, sum(attr) = {attributions.sum()}, MAE = {my_mae.item()}')
                
                '''
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
    '''

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

class GradientAttribution(EmbeddingAttribution):

    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None,times_input=False,smoothgrad=False):
        
        super().__init__(model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None)
      
        self.smoothgrad = smoothgrad
        sl = IntegratedGradients(self.predictor)
        self.attr_fn = sl.attribute
        
        if times_input:
            sl = InputXGradient(self.predictor)
        else:
            sl = Saliency(self.predictor)
        
        if self.smoothgrad:
            sl = NoiseTunnel(sl)
            if times_input:
                self.attr_fn = partial(sl.attribute,nt_type='smoothgrad',nt_samples=self.sample_size,\
                                            nt_samples_batch_size=self.batch_size)
            else:
                self.attr_fn = partial(sl.attribute,abs=False,nt_type='smoothgrad',nt_samples=self.sample_size,\
                                            nt_samples_batch_size=self.batch_size)
        else:
            if times_input:
                self.attr_fn = sl.attribute 
            else: 
                self.attr_fn = partial(sl.attribute,abs=False)  

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        ig = IntegratedGradients(self.predictor)

        with open(savefile,'w') as outFile:
            
            for batch in tqdm.tqdm(val_iterator):
                ids = batch.indices.tolist()
                src, src_lens = batch.src
                print('SRC_LENS',src_lens.shape,src_lens)
                src = src.transpose(0,1)
                batch_size = batch.batch_size
                src_embed = self.src_embed(src)
                nc_total = []
                pc_total =[]
                IG = True
                if self.smoothgrad:
                    # set noise level as pct of src 
                    mean = torch.mean(src_embed).item()
                    std = torch.std(src_embed,unbiased=True).item()
                    noise_std = 0.2*std
                    # grad wrt positive and negative class
                    pc_attributions = self.attr_fn(inputs=src_embed,target=self.pc_token,stdevs=noise_std,
                                        additional_forward_args = (src_lens,decoder_input,batch_size))
                    nc_attributions = self.attr_fn(inputs=src_embed,target=self.nc_tokens,stdevs=noise_std,
                                        additional_forward_args = (src_lens,decoder_input,batch_size))
                else:
                    # grad wrt positive and negative class
                    pc_attributions = self.attr_fn(inputs=src_embed,target=self.pc_token,
                                        additional_forward_args = (src_lens,decoder_input,batch_size))
                    nc_attributions = self.attr_fn(inputs=src_embed,target=self.nc_token,
                                        additional_forward_args = (src_lens,decoder_input,batch_size))


                mean = torch.mean(pc_attributions).item()
                std = torch.std(pc_attributions,unbiased=True).item()
                # summarize
                pc_attributions = pc_attributions.detach().cpu().numpy()
                pc_summed = np.sum(pc_attributions,axis=2)
                print(f'pc_attributions = {pc_attributions.shape}, pc_summed = {pc_summed.shape}') 
                pc_normed = np.linalg.norm(pc_attributions,2,axis=2)
                nc_attributions = nc_attributions.detach().cpu().numpy()
                nc_summed = np.sum(nc_attributions,axis=2)
                nc_normed = np.linalg.norm(nc_attributions,2,axis=2)
               
                if baseline in ['A','C','G','T']:
                    pc_summed = -pc_summed
                    nc_summed = -nc_summed

                saved_src = src.detach().cpu().numpy()
                for j in range(batch_size):
                    tscript = transcript_names[ids[j]]
                    curr_saved_src = "".join([self.src_vocab.itos[x] for x in saved_src[j,:,0]])
                    curr_saved_src = curr_saved_src.split('<blank>')[0]
                    true_len = len(curr_saved_src)
                    pc_summed_attr = ['{:.3e}'.format(x) for x in pc_summed[:,j,:].tolist()[:true_len]]
                    nc_summed_attr = ['{:.3e}'.format(x) for x in nc_summed[:,j,:].tolist()[:true_len]]
                    pc_normed_attr = ['{:.3e}'.format(x) for x in pc_normed[:,j,:].tolist()[:true_len]]
                    nc_normed_attr = ['{:.3e}'.format(x) for x in nc_normed[:,j,:].tolist()[:true_len]]
                    
                    entry = {"ID" : tscript , "summed_attr_PC" : pc_summed_attr, "summed_attr_NC" : nc_summed_attr,\
                            "normed_attr_PC" : pc_normed_attr , "normed_attr_NC" : nc_normed_attr ,"src" : curr_saved_src}
                    
                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")