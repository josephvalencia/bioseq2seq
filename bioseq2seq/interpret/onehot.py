'''Provides for gradient based interpretation at the level of the embedding vectors using the Captum library'''
import torch
import numpy as np
from bioseq2seq.bin.transforms import CodonTable, getLongestORF
from base import Attribution, OneHotGradientAttribution
import torch.nn.functional as F

class OneHotSalience(OneHotGradientAttribution):

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            
            tscript = transcript_names[ids[0]]
            
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            onehot_src = self.onehot_embed_layer(src)
            tgt_prefix = batch.tgt[:target_pos,:,:] 
            ground_truth = batch.tgt[target_pos,0,0].item()
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size,tgt_prefix),batch_size)
            probs = F.softmax(pred_classes)
            class_token = ground_truth if self.class_token == 'GT' else self.class_token
           
            #third_pos = batch.tgt[3,0,0].item()
            
            observed_score = pred_classes[:,:,class_token]
            likelihood = self.class_likelihood_ratio(pred_classes,class_token)
            input_grad = self.input_grads(likelihood,onehot_src)
            saved_src = src.detach().cpu().numpy()
              
            for b in range(batch_size):
                tscript = transcript_names[ids[b]]
                true_len = src_lens[b].item()
                saliency = input_grad[b,:,0,:true_len]
                corrected = saliency - saliency.mean(dim=-1).unsqueeze(dim=1)
                #corrected = saliency 
                onehot_batch = onehot_src[b,:,:true_len].squeeze()
                inputxgrad = (onehot_batch * corrected).sum(-1) 
                all_inputxgrad[tscript] = inputxgrad.detach().cpu().numpy() 
                all_grad[tscript] = corrected.detach().cpu().numpy() 
        
        inputxgrad_savefile = savefile.replace('grad','inputXgrad')
        print(f'saving {inputxgrad_savefile} and {savefile}')
        np.savez(savefile,**all_grad) 
        np.savez(inputxgrad_savefile,**all_inputxgrad)

class OneHotExpectedGradients(OneHotGradientAttribution):

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
            yield stacked.squeeze(3)
    
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
            print(f'pred_classes ={pred_classes.shape}')
            class_score = pred_classes[:,self.class_token]
            input_grad = torch.autograd.grad(class_score.sum(),onehot_src)[0]
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
            for y,baseline_batch in enumerate(self.extract_baselines(batch,0,self.sample_size,minibatch_size)):
                # score baselines
                baseline_onehot = self.onehot_embed_layer(baseline_batch)
                print(f'baseline = {baseline_onehot.shape} , decoder_input = {decoder_input[0].shape}')
                baseline_pred_classes = self.predictor(baseline_onehot,src_lens,decoder_input,minibatch_size)
                base_pred = baseline_pred_classes[:,self.class_token]
                batch_preds.append(base_pred.detach().cpu())
                # sample along paths 
                alpha = torch.rand(baseline_onehot.shape[0],1,1,1).to(device=self.device)
                direction = onehot_src - baseline_onehot 
                interpolated = baseline_onehot + alpha*direction
                # compute gradients 
                pred_classes = self.predictor(interpolated,src_lens,decoder_input,minibatch_size)
                interpolated_pred = pred_classes[:,self.class_token]
                interpolated_grads = self.input_grads(interpolated_pred,interpolated) 
                interpolated_grads = direction *interpolated_grads
                interpolated_grads = interpolated_grads.detach().cpu()
                batch_attributions.append(interpolated_grads)
           
            # take expectation
            batch_attributions = torch.cat(batch_attributions,dim=0)
            attributions = batch_attributions.mean(dim=0).detach().cpu().numpy() 
            # check convergence
            baseline_preds = torch.cat(batch_preds,dim=0)
            mean_baseline_score = baseline_preds.mean()
            diff = true_class_score - mean_baseline_score
            my_mae = diff - np.sum(attributions)
            print(diff.item(),np.sum(attributions),my_mae.item())
            
            '''
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
        #np.savez(savefile,**all_grad) 
        #np.savez(inputxgrad_savefile,**all_inputxgrad)
        
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
        '''

class OneHotMDIG(OneHotGradientAttribution):

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
            tgt_prefix = batch.tgt[:target_pos,:,:]
            
            saved_src = src.detach().cpu().numpy()
            
            raw_src = self.get_raw_src(src) 
            start,end = getLongestORF(raw_src)
            
            n_samples = 50
            mdig = torch.zeros_like(onehot_src,device=onehot_src.device)

            # score true
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size,tgt_prefix),batch_size)
            true_class_score = pred_classes[:,:,self.class_token]
            true_likelihood = self.class_likelihood_ratio(pred_classes)
            # score baseline
            for c,base in zip(range(2,6),['A','C','G','T']): 
                character = F.one_hot(torch.tensor(c,device=onehot_src.device),num_classes=8).type(torch.float).requires_grad_(True)
                baseline = character.repeat(batch_size,onehot_src.shape[1],onehot_src.shape[2],1) 
                pred_classes = self.predictor(baseline,src_lens,self.decoder_input(batch_size,tgt_prefix),batch_size)
                base_class_score = pred_classes[:,:,self.class_token]
                base_class_likelihood = self.class_likelihood_ratio(pred_classes)
                direction = onehot_src - baseline
                storage = []
                scores = [] 
                for n in range(n_samples):
                    interpolated_src = baseline + (n/n_samples)*direction 
                    pred_classes = self.predictor(interpolated_src,src_lens,self.decoder_input(batch_size,tgt_prefix),batch_size)
                    class_score = pred_classes[:,:,self.class_token]
                    likelihood = self.class_likelihood_ratio(pred_classes)
                    input_grad = self.input_grads(likelihood,interpolated_src)
                    storage.append(input_grad)
                    scores.append(class_score)

                saved_src = src.detach().cpu().numpy()
                all_grads = torch.stack(storage,dim=2)
                average_grad = direction * all_grads.mean(dim=2)
                # assess IG completeness property
                summed = average_grad.sum()
                #diff = true_class_score - base_class_score  
                diff = true_likelihood - base_class_likelihood 
                scores = torch.stack(scores,dim=0)
                print(f'true={true_class_score.item():.3f}, diff={diff.item():.3f}, sum = {summed:.3f}, mean(IG) = {scores.mean():.3f}, var(IG) = {scores.var():.3f}')
                mdig += average_grad

            for b in range(batch_size):
                tscript = transcript_names[ids[b]]
                print(tscript)
                true_len = src_lens[b].item()
                #saliency = average_grad[b,:,0,:true_len]
                saliency = mdig[b,:,0,:true_len]
                corrected = saliency 
                onehot_batch = onehot_src[b,:,:true_len].squeeze()
                inputxgrad = (onehot_batch * corrected).sum(-1) 
                all_inputxgrad[tscript] = inputxgrad.detach().cpu().numpy() 
                all_grad[tscript] = corrected.detach().cpu().numpy() 
        
        np.savez(savefile,**all_grad) 

class OneHotIntegratedGradients(OneHotGradientAttribution):

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        count = 0 
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            tscript = transcript_names[ids[0]]

            # score true
            batch_size = batch.batch_size
            onehot_src = self.onehot_embed_layer(src)
            tgt_prefix = batch.tgt[:target_pos,:,:] 
            ground_truth = batch.tgt[target_pos,0,0].item()
            class_token = ground_truth if self.class_token == 'GT' else self.class_token
            pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size,tgt_prefix),batch_size)
            true_likelihood = self.class_likelihood_ratio(pred_classes,class_token)
            probs = F.softmax(pred_classes)
            saved_src = src.detach().cpu().numpy()

            storage = []
            n_samples = 50
            
            # score baseline
            character = torch.tensor([0.0,0.0,0.25,0.25,0.25,0.25,0.0,0.0],device=onehot_src.device)
            baseline = character*torch.ones_like(onehot_src,device=onehot_src.device,requires_grad=True)
            pred_classes = self.predictor(baseline,src_lens,self.decoder_input(batch_size,tgt_prefix),batch_size)
            base_likelihood = self.class_likelihood_ratio(pred_classes,class_token)
            direction = onehot_src - baseline

            scores = [] 
            for n in range(n_samples):
                interpolated_src = baseline + (n/n_samples)*direction 
                pred_classes = self.predictor(interpolated_src,src_lens,self.decoder_input(batch_size,tgt_prefix),batch_size)
                class_score = self.class_likelihood_ratio(pred_classes,class_token)
                input_grad = self.input_grads(class_score.sum(),interpolated_src)
                storage.append(input_grad)
                scores.append(class_score)

            saved_src = src.detach().cpu().numpy()
            all_grads = torch.stack(storage,dim=2)
            average_grad = direction * all_grads.mean(dim=2)
            
            # assess IG completeness property
            summed = average_grad.sum()
            diff = true_likelihood - base_likelihood
            scores = torch.stack(scores,dim=0)
            print(f'true={true_likelihood.item():.3f}, diff={diff.item():.3f}, sum = {summed:.3f}, mean(IG) = {scores.mean():.3f}, var(IG) = {scores.var():.3f}')

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
            count+=1
            if count == 8:
                break
        
        np.savez(savefile,**all_grad) 

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
                p = F.softmax(class_score)
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
        
        inputxgrad_savefile = savefile.replace('SG','inputX-SG')
        print(f'saving {inputxgrad_savefile} and {savefile}')
        np.savez(savefile,**all_grad) 
        np.savez(inputxgrad_savefile,**all_inputxgrad)
