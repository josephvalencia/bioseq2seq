import torch
from Bio.Seq import Seq
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.SeqRecord import SeqRecord
from base import Attribution, OneHotGradientAttribution
from bioseq2seq.bin.transforms import CodonTable, getLongestORF
import torch.nn.functional as F
import time
import numpy as np

class DiscreteLangevinSampler(OneHotGradientAttribution):
 
    '''
    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None,times_input=False,smoothgrad=False):
    
        super().__init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None,times_input=False,smoothgrad=False):
    ''' 
    def multidim_synonymize(self,true_src,alt_src):
        
        batch_size = alt_src.shape[0]
        storage = []  
        for b in range(batch_size):
            result = self.reject_missense(true_src[b,:,:].unsqueeze(0),alt_src[b,:,:].unsqueeze(0))
            storage.append(result)
        return torch.cat(storage,dim=0)
        
    def reject_missense(self,true_src,alt_src):
        
        raw_true = self.get_raw_src(true_src)
        raw_alt = self.get_raw_src(alt_src)
     
        # optimize CDS codons using expected gradients scores
        start,end = getLongestORF(''.join(raw_true))
        cds = raw_true[start:end] 
        alt_cds = raw_alt[start:end]
        sep = '____________________________________'
        reject = 0
        proposed = 0
        table = CodonTable()
        mask = [False] * start
    
        has_upstream_start = False
        upstream_start_loc = -1
        frame = start % 3
        # disallow adding upstream in-frame methionine and deleting upstream in-frame stop codons
        for i in range(frame,start,3):
            assert frame == (i % 3)
            codon = ''.join(raw_true[i:i+3])
            alt_codon = ''.join(raw_alt[i:i+3])
            aa1 = table.codon_to_aa(codon)
            aa2 = table.codon_to_aa(alt_codon)
            if aa1 == 'M':
                has_upstream_start = True
                upstream_start_loc = i
            # cannot delete stop codon of uORF
            elif aa1 == '*' and aa2 != '*' and has_upstream_start:
                mask[i:i+3] = [True]*3
            # cannot add upstream methoionine 
            elif aa1 != 'M' and aa2 == 'M':
                mask[i:i+3] = [True]*3
        
        # enforce synonymous CDS  
        for i in range(0,len(cds),3):
            assert i+3 <= end
            codon = ''.join(cds[i:i+3])
            alt_codon = ''.join(alt_cds[i:i+3])
            if codon != alt_codon:
                diff = 0
                for c1,c2 in zip(codon,alt_codon):
                    if c1 != c2:
                        diff+=1
                proposed+=diff
                aa1 = table.codon_to_aa(codon)
                aa2 = table.codon_to_aa(alt_codon)
                if aa2 != aa1:
                    reject+=diff
                    mask.extend([True]*3)
                else:
                    mask.extend([False]*3)
            else:
                mask.extend([False]*3)

        old_mask_len = len(mask)
        mask = mask + [False]*(true_src.shape[1] - old_mask_len)
        mask = torch.tensor(mask,device=true_src.device).reshape(1,-1,1)
        synonymized = torch.where(mask,true_src,alt_src)
        print(synonymized.shape)
        raw_alt = self.get_raw_src(synonymized)
        alt_cds = raw_alt[start:end]
        aa_old = Seq(''.join(cds)).translate()
        aa_new = Seq(''.join(alt_cds)).translate()
        assert aa_old == aa_new
        return synonymized
    
    def mask_rare_tokens(self,logits):
        '''Non-nucleotide chars may receive gradient scores, set them to a small number'''
        
        mask1 = torch.tensor([0,0,1,1,1,1,0,0],device=logits.device)
        mask2 = torch.tensor([-30,-30,0,0,0,0,-30,-30],device=logits.device)
        return logits*mask1 + mask2

    def langevin_proposal_dist(self,grads,indexes,onehot):
       
        grad_current_char =  torch.gather(grads,dim=3,index=indexes.unsqueeze(2))
        mutation_scores = grads - grad_current_char
        temperature_term = (1.0 - onehot) / (self.alpha * self.scaled_preconditioner()) 
        logits = 0.5 * mutation_scores - temperature_term 
        logits = self.mask_rare_tokens(logits)
        proposal_dist = torch.distributions.Categorical(logits=logits.squeeze(2))
        return torch.distributions.Independent(proposal_dist,1)
  
    def metropolis_hastings(self,score_diff,forward_prob,reverse_prob):
        acceptance_log_prob = score_diff + reverse_prob - forward_prob
        return torch.minimum(acceptance_log_prob,torch.tensor([0.0],device=score_diff.device))
   
    def set_stepsize(self,alpha):
        self.alpha = alpha
    
    def get_stepsize(self):
        return self.alpha

    def update_preconditioner(self,grads,step):
       
        grads = grads.pow(2)
        diagonal = grads
        biased = self.beta*self.preconditioner + (1.0-self.beta)*diagonal 
        self.preconditioner = biased / (1.0 - self.beta**(step+1))

    def scaled_preconditioner(self):
        return self.preconditioner.sqrt() + 1.0

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        start = time.time()
        mutant_records = []
        
        for i,batch in enumerate(val_iterator):
            
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            tscript = transcript_names[ids[0]]
            
            # prediction and input grads given tgt_prefix
            tgt_prefix = batch.tgt[:target_pos,:,:] 
            class_token = batch.tgt[target_pos,0,0].item() if self.class_token == 'GT' else self.class_token
            
            # score true
            original = src
            original_raw_src = self.get_raw_src(original)
            original_onehot = self.onehot_embed_layer(original)
            original_logit, original_probs =  self.predict_logits(original_onehot,src_lens,
                                                    self.decoder_input(batch_size,tgt_prefix),
                                                    batch_size,class_token,ratio=True)

            MAX_STEPS = 512
            self.set_stepsize(0.20)
            self.beta = 0.90
            self.preconditioner = torch.zeros(size=(*src.shape,8),device=src.device,dtype=torch.float)
            max_score = original_logit
            best_seq = original 
            best_step = 0
            batch_preds = []

            for s in range(MAX_STEPS): 
                onehot_src = self.onehot_embed_layer(src) 
                # calc q(x'|x)
                logit, probs =  self.predict_logits(onehot_src,src_lens,
                                                    self.decoder_input(batch_size,tgt_prefix),
                                                    batch_size,class_token,ratio=True)
                curr_grads = self.input_grads(logit,onehot_src)
                self.update_preconditioner(curr_grads,s)
                proposal_dist = self.langevin_proposal_dist(curr_grads,src,onehot_src)
                resampled = proposal_dist.sample().unsqueeze(2)
                resampled = self.reject_missense(original,resampled)
                #resampled = self.multidim_synonymize(original,resampled)
                forward_prob = proposal_dist.log_prob(resampled.squeeze(2)) 
                diff = torch.count_nonzero(src != resampled)
                    
                # correct with MH step
                # calc q(x|x')
                resampled_onehot = self.onehot_embed_layer(resampled)
                resampled_logit, resampled_probs =  self.predict_logits(resampled_onehot,src_lens,
                                                    self.decoder_input(batch_size,tgt_prefix),
                                                    batch_size,class_token,ratio=True)
                resampled_grads = self.input_grads(resampled_logit,resampled_onehot)
                resampled_proposal_dist = self.langevin_proposal_dist(resampled_grads,resampled,resampled_onehot)
                reverse_prob = resampled_proposal_dist.log_prob(src.squeeze(2))

                score_diff = resampled_logit - logit
                accept_log_probs = self.metropolis_hastings(score_diff,forward_prob,reverse_prob) 
                random_draw = torch.log(torch.rand(src.shape[0],device=src.device))
                acceptances = accept_log_probs > random_draw
                #acceptances = acceptances.new_ones(*acceptances.shape)
                src = torch.where(acceptances,resampled,src)
                 
                # cache best sequence
                if resampled_logit > max_score:
                    max_score = resampled_logit 
                    best_seq = resampled
                    best_step = s
                    print(f'{tscript} found new best at step {s} seq, P(<NC>) = {resampled_probs[:,:,self.nc_token].item():.3f}, P(<PC>) = {resampled_probs[:,:,self.pc_token].item():.3f}')
                if s % 10 == 0:
                    best_onehot = self.onehot_embed_layer(src)
                    best_logit, best_probs = self.predict_logits(best_onehot,src_lens,
                                                        self.decoder_input(batch_size,tgt_prefix),
                                                        batch_size,class_token,ratio=True)
                    diff_original = torch.count_nonzero(src != original)
                    score_improvement = best_logit - original_logit
                    if probs[:,:,class_token].item() > 0.95:
                        break
                    end = time.time()
                    verbose = False 
                    if verbose: 
                        print(f'step={s}, {tscript}') 
                        print(f'# diff from original = {diff_original}/{src.shape[0]*src.shape[1]}')
                        print(f'P(<NC>) = {best_probs[:,:,self.nc_token].item():.3f}, P(<PC>) = {best_probs[:,:,self.pc_token].item():.3f}')
                        print(f'score improvement = {score_improvement.item():.3f}')
                        print(f'Time elapsed = {end-start} s')
                    start = time.time() 
            sep = '________________________________________'
            print(sep)
            print(f"The sequence has converged, found at step {best_step}")
            diff_original = torch.count_nonzero(best_seq != original)
            differences = (best_seq != original).reshape(-1).detach().cpu().numpy().tolist()
            print(f'# diff from original = {diff_original}/{best_seq.shape[0]*best_seq.shape[1]}')
            raw_src = self.get_raw_src(best_seq)
            print(''.join(raw_src))
            print('ORIGINAL',''.join(original_raw_src))
            best_onehot = self.onehot_embed_layer(best_seq)
            best_logit, best_probs = self.predict_logits(best_onehot,src_lens,
                                                self.decoder_input(batch_size,tgt_prefix),
                                                batch_size,class_token,ratio=True)
            
            print(f'P(<NC>) = {best_probs[:,:,self.nc_token].item():.3f}, P(<PC>) = {best_probs[:,:,self.pc_token].item():.3f}')
            print(sep)
            description = f'step_found : {best_step}, diff:{diff_original}/{best_seq.shape[0]*best_seq.shape[1]}, P(<NC>):{original_probs[:,:,5].item():.3f}->{best_probs[:,:,5].item():.3f}'
            rec = SeqRecord(Seq(''.join(raw_src)),id=f'{tscript}-MUTANT',description=description)
            mutant_records.append(rec)
        with open(f'{savefile}.fasta','w') as out_file:
            SeqIO.write(mutant_records,out_file,'fasta')
