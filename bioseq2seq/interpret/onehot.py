'''Provides for gradient based interpretation at the level of the embedding vectors using the Captum library'''
import torch
import numpy as np
from bioseq2seq.bin.transforms import CodonTable, getLongestORF
from base import Attribution, OneHotGradientAttribution
import torch.nn.functional as F

class OneHotSalience(OneHotGradientAttribution):
    ''' ordinary gradients wrt. input using a one-hot encoding '''
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            # unpack batch
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            tscript = transcript_names[ids[0]]
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            onehot_src = self.onehot_embed_layer(src)
           
            # prediction and input grads given tgt_prefix
            tgt_prefix = batch.tgt[:target_pos,:,:] 
            class_token = batch.tgt[target_pos,0,0].item() if self.class_token == 'GT' else self.class_token
            logit,probs = self.predict_logits(onehot_src,src_lens,
                                            self.decoder_input(batch_size,tgt_prefix),
                                            batch_size,class_token)
            input_grad = self.input_grads(logit,onehot_src)
            
            # save grad and input * grad
            for b in range(batch_size):
                tscript = transcript_names[ids[b]]
                true_len = src_lens[b].item()
                saliency = input_grad[b,:,0,:true_len]
                onehot_batch = onehot_src[b,:,:true_len].squeeze()
                inputxgrad = (onehot_batch * saliency).sum(-1) 
                all_inputxgrad[tscript] = inputxgrad.detach().cpu().numpy() 
                all_grad[tscript] = saliency.detach().cpu().numpy() 
        
        inputxgrad_savefile = savefile.replace('grad','inputXgrad')
        print(f'saving {inputxgrad_savefile} and {savefile}')
        np.savez_compressed(savefile,**all_grad) 
        np.savez_compressed(inputxgrad_savefile,**all_inputxgrad)

class OneHotExpectedGradients(OneHotGradientAttribution):
    ''' Implementation of Erion et. al 2020, http://arxiv.org/abs/1906.10670 '''

    def extract_baselines(self,batch,batch_index,num_copies,copies_per_step):
        
        baselines = []
        # get shuffled copies by index
        for i in range(num_copies):
            seq_name = f'src_shuffled_{i}'
            attr = getattr(batch,seq_name)
            baselines.append(attr[0])

        upper = max(num_copies,1)
        # iterate in batches of size copies_per_step
        for i in range(0,upper,copies_per_step):
            chunk = baselines[i:i+copies_per_step]
            stacked = torch.stack(chunk,dim=0).to(device=batch.src[0].device)
            yield stacked.squeeze(3)
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            # score true
            onehot_src = self.onehot_embed_layer(src) 
            true_logit,true_probs = self.predict_logits(onehot_src,src_lens,
                                                        self.decoder_input(batch_size),
                                                        batch_size,self.class_token)
            # set minibatch size based on memory
            minibatch_size = 16
            decoder_input = self.decoder_input(minibatch_size)
            batch_attributions = []
            batch_preds = []                     
            
            for y,baseline_batch in enumerate(self.extract_baselines(batch,0,self.sample_size,minibatch_size)):
                # score baselines
                baseline_onehot = self.onehot_embed_layer(baseline_batch)
                base_logit,base_probs = self.predict_logits(baseline_onehot,src_lens,
                                                                decoder_input,minibatch_size,
                                                                self.class_token)
                batch_preds.append(base_logit.detach().cpu())
                # sample along linear path between true and each baseline 
                alpha = torch.rand(baseline_onehot.shape[0],1,1,1).to(device=self.device)
                direction = onehot_src - baseline_onehot 
                interpolated = baseline_onehot + alpha*direction
                # compute and save input gradients 
                interpolated_logit,interpolated_probs = self.predict_logits(interpolated,src_lens,
                                                                            decoder_input,minibatch_size,
                                                                            self.class_token)
                interpolated_grads = self.input_grads(interpolated_logit,interpolated)
                inner_integral = direction * interpolated_grads
                batch_attributions.append(inner_integral.detach().cpu())

            # take expectation
            batch_attributions = torch.cat(batch_attributions,dim=0)
            attributions = batch_attributions.mean(dim=0).numpy() 
            # check convergence
            baseline_preds = torch.cat(batch_preds,dim=0)
            mean_baseline_logit = baseline_preds.mean()
            diff = true_logit - mean_baseline_logit
            my_mae = diff - np.sum(attributions)
            print(f'SCORE DIFF= {diff.item():.3f}, sum(attr) = {attributions.sum():.3f}, error = {my_mae.item():.3f}')
            
            # save results
            tscript = transcript_names[ids[0]]
            true_len = src_lens[0].item()
            all_grad[tscript] = attributions[:,0,:true_len]
        
        print(f'saving {savefile}')
        np.savez_compressed(savefile,**all_grad) 
        
class OneHotMDIG(OneHotGradientAttribution):

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            tscript = transcript_names[ids[0]]

            # score true
            tgt_prefix = batch.tgt[:target_pos,:,:]
            onehot_src = self.onehot_embed_layer(src) 
            true_logit, true_probs = self.predict_logits(onehot_src,src_lens,
                                                    self.decoder_input(batch_size,tgt_prefix),
                                                    batch_size,self.class_token,ratio=True)
            n_samples = 128
            minibatch_size = 16
            mdig = []

            # score all single-nuc baselines one by one
            for c,base in zip(range(2,6),'ACGT'): 
                character = F.one_hot(torch.tensor(c,device=onehot_src.device),num_classes=8).type(torch.float).requires_grad_(True)
                baseline = character.repeat(batch_size,onehot_src.shape[1],onehot_src.shape[2],1) 
                base_class_logit, base_probs = self.predict_logits(baseline,src_lens,
                                                                self.decoder_input(batch_size,tgt_prefix),
                                                                batch_size,self.class_token,ratio=True)
                grads = []
                scores = []
                grid = torch.linspace(0,1,n_samples,device=baseline.device) 
                direction = onehot_src - baseline

                for n in range(0,n_samples,minibatch_size):
                    alpha = grid[n:n+minibatch_size].reshape(minibatch_size,1,1,1) 
                    interpolated_src = baseline + alpha*direction 
                    logit, probs = self.predict_logits(interpolated_src,src_lens,
                                                        self.decoder_input(minibatch_size,tgt_prefix),
                                                        minibatch_size,self.class_token,ratio=True)
                    probs = probs.squeeze()
                    #print(f'alpha = {alpha.squeeze()}, P(class) = {probs[:,self.class_token]}')
                    input_grad = self.input_grads(logit,interpolated_src)
                    grads.append(input_grad.detach().cpu())
                    scores.append(logit.detach().cpu())

                all_grads = torch.cat(grads,dim=0)
                average_grad = direction.detach().cpu() * all_grads.mean(dim=0)
                # assess IG completeness property
                summed = average_grad.sum()
                diff = true_logit - base_class_logit 
                ig_summed = average_grad.sum(dim=(2,3))
                scores = torch.stack(scores,dim=0)
                print(f'base = poly-{base}, true={true_logit.item():.3f}, diff={diff.item():.3f}, sum = {summed:.3f}')
                mdig.append(ig_summed)

            attr = torch.cat(mdig,dim=0)
            tscript = transcript_names[ids[0]]
            true_len = src_lens[0].item()
            saliency = attr[:,:true_len]
            all_grad[tscript] = saliency.detach().cpu().numpy() 
        
        np.savez_compressed(savefile,**all_grad) 

class OneHotIntegratedGradients(OneHotGradientAttribution):

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            print(f'src = {src.shape}') 
            tscript = transcript_names[ids[0]]
            batch_size = batch.batch_size

            # score true
            tgt_prefix = batch.tgt[:target_pos,:,:] 
            class_token = batch.tgt[:target_pos,0,0].item() if self.class_token == 'GT' else self.class_token
            onehot_src = self.onehot_embed_layer(src)
            true_logit,true_probs = self.predict_logits(onehot_src,src_lens,
                                                        self.decoder_input(batch_size,tgt_prefix),
                                                        batch_size,class_token)
            baseline_type = 'uniform'
            # score baseline
            if baseline_type == 'zeros':  
                baseline = onehot_src.new_zeros(*onehot_src.shape)
            else:
                character = torch.tensor([0.0,0.0,0.25,0.25,0.25,0.25,0.0,0.0],device=onehot_src.device)
                baseline = character*torch.ones_like(onehot_src,device=onehot_src.device,requires_grad=True)
            
            base_logit,base_probs = self.predict_logits(baseline,src_lens,
                                                    self.decoder_input(batch_size,tgt_prefix),
                                                    batch_size,class_token)
            grads = []
            scores = [] 
            n_samples = 128
            minibatch_size = 16
            grid = torch.linspace(0,1,n_samples,device=baseline.device) 
            direction = onehot_src - baseline

            for n in range(0,n_samples,minibatch_size):
                alpha = grid[n:n+minibatch_size].reshape(minibatch_size,1,1,1) 
                interpolated_src = baseline + alpha*direction 
                logit,probs = self.predict_logits(interpolated_src,src_lens,
                                            self.decoder_input(minibatch_size,tgt_prefix),
                                            minibatch_size,class_token)
                input_grad = self.input_grads(logit,interpolated_src)
                grads.append(input_grad.detach().cpu())
                scores.append(logit.detach().cpu())

            all_grads = torch.cat(grads,dim=0)
            average_grad = direction.detach().cpu() * all_grads.mean(dim=0)
            
            # assess IG completeness property
            summed = average_grad.sum()
            diff = true_logit - base_logit
            scores = torch.stack(scores,dim=0)
            print(f'true={true_logit.item():.3f}, diff={diff.item():.3f}, sum = {summed:.3f}')

            for b in range(batch_size):
                tscript = transcript_names[ids[b]]
                true_len = src_lens[b].item()
                saliency = average_grad[b,:,0,:true_len]
                print(saliency.shape) 
                all_grad[tscript] = saliency.detach().cpu().numpy() 
        
        print(f'saving {savefile}')
        np.savez_compressed(savefile,**all_grad) 
