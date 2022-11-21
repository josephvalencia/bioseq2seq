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

class DiscreteLangevinSampler(OneHotGradientAttribution):
 
    '''
    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None,times_input=False,smoothgrad=False):
    
        super().__init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None,times_input=False,smoothgrad=False):
    ''' 
   
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
            elif aa1 == '*' and aa2 != '*' and has_upstream_start:
                mask[i:i+3] = [True]*3
                #print(f'protecting STOP at {i-3} in frame {(i-3) % 3} compared to {start}  ({start % 3})') 
                #print(f'upstream START was at {upstream_start_loc} ({upstream_start_loc % 3})')
                #print(f'mutation was {codon} -> {alt_codon}') 
            elif aa1 != 'M' and aa2 == 'M':
                mask[i:i+3] = [True]*3
                #print(f'overwriting M at {i-3} in frame {(i-3) % 3} compared to {start} ({start % 3})') 
                #print(f'mutation was {codon} -> {alt_codon}') 
        
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
                    #print(f'rejected {codon} -> {alt_codon} ({aa2}) at residue {aa1}_{i / 3}')
                else:
                    mask.extend([False]*3)
                    #print(f'accepted {codon} -> {alt_codon} at residue {aa1}_{i / 3}')
            else:
                mask.extend([False]*3)

        #print(f'rejected {reject}/{proposed} proposed CDS mutations') 
        old_mask_len = len(mask)
        mask = mask + [False]*(true_src.shape[1] - old_mask_len)
        mask = torch.tensor(mask,device=true_src.device).reshape(1,-1,1)
        try:
            synonymized = torch.where(mask,true_src,alt_src)
            raw_alt = self.get_raw_src(synonymized)
            alt_cds = raw_alt[start:end]
            aa_old = Seq(''.join(cds)).translate()
            aa_new = Seq(''.join(alt_cds)).translate()
            #alignments = pairwise2.align.globalxx(aa_old,aa_new)
            #print(format_alignment(*alignments[0])) 
            #print(len(aa_old),len(aa_new))
            assert aa_old == aa_new
            ''' 
            if not is_synonymous:
                print(s1,e1)
                print(s2,e2)
                print('RNA original',''.join(raw_true))
                print('CDS original',''.join(raw_true[s1:e1]))
                print('RNA alt',''.join(raw_alt))
                print('CDS alt',''.join(raw_alt[s2:e2]))
                quit()
            '''
        except RuntimeError:
            print(f'start={start}, end={end}, old_mask_len = {old_mask_len}, raw_true_len = {len(raw_true)}, mask = {mask.shape}, true={true_src.shape}, alt={alt_src.shape}')
            print(raw_true)
            quit()
        return synonymized
    
    def mask_rare_tokens(self,logits):
        '''Non-nucleotide chars may receive gradient scores, set them to a small number'''
        
        mask1 = torch.tensor([0,0,1,1,1,1,0,0],device=logits.device)
        mask2 = torch.tensor([-100,-100,0,0,0,0,-100,-100],device=logits.device)
        return logits*mask1 + mask2

    def langevin_proposal_dist(self,grads,indexes,onehot):
       
        grad_current_char =  torch.gather(grads,dim=3,index=indexes.unsqueeze(2))
        mutation_scores = grads - grad_current_char
        temperature_term = (1.0 - onehot) / (self.alpha * self.scaled_preconditioner()) 
        logits = 0.5*mutation_scores - temperature_term 
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
        #diagonal = grads.squeeze(0).sum(-1)
        diagonal = grads
        biased = self.beta*self.preconditioner + (1.0-self.beta)*diagonal 
        self.preconditioner = biased / (1.0 - self.beta**(step+1))
        #self.preconditioner = grads.new_ones(*grads.shape)

    def scaled_preconditioner(self):
        #return self.preconditioner.new_ones(self.preconditioner.shape)
        return self.preconditioner.sqrt() + 1.0

    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        start = time.time()
        mutant_records = []
        for i,batch in enumerate(val_iterator):
            
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            original = src
            original_score = None
            original_probs = None

            tscript = transcript_names[ids[0]]
            
            MAX_STEPS = 1000
            self.set_stepsize(0.40)
            self.beta = 0.90
            self.preconditioner = torch.zeros(size=(*src.shape,8),device=src.device,dtype=torch.float)
            
            max_score_diff = -1000
            best_seq = None
            best_step = 0

            for s in range(MAX_STEPS): 
                onehot_src = self.onehot_embed_layer(src) 
                
                # calc q(x'|x)
                pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size),batch_size)
                true_pred = pred_classes[:,:,self.class_token]
                likelihood = self.class_likelihood_ratio(pred_classes,self.class_token)
                
                if original_score == None:
                    original_score = likelihood
                    original_probs = F.softmax(pred_classes,dim=-1)
                
                true_grads = self.input_grads(likelihood,onehot_src)
                self.update_preconditioner(true_grads,s)
                proposal_dist = self.langevin_proposal_dist(true_grads,src,onehot_src)
                resampled = proposal_dist.sample().unsqueeze(2)
                resampled = self.reject_missense(original,resampled) 
                diff = torch.count_nonzero(src != resampled)
                forward_prob = proposal_dist.log_prob(resampled.squeeze(2)) 
                
                # correct with MH step
                # calc q(x|x')
                resampled_onehot = self.onehot_embed_layer(resampled)
                pred_classes = self.predictor(resampled_onehot,src_lens,self.decoder_input(batch_size),batch_size)
                resampled_pred = pred_classes[:,:,self.class_token]
                resampled_likelihood = self.class_likelihood_ratio(pred_classes,self.class_token)
                
                resampled_grads = self.input_grads(resampled_likelihood,resampled_onehot)
                resampled_proposal_dist = self.langevin_proposal_dist(resampled_grads,resampled,resampled_onehot)
                reverse_prob = resampled_proposal_dist.log_prob(src.squeeze(2))

                #score_diff = resampled_pred - true_pred
                score_diff = resampled_likelihood - likelihood
                accept_log_probs = self.metropolis_hastings(score_diff,forward_prob,reverse_prob) 
                random_draw = torch.log(torch.rand(src.shape[0],device=src.device))
                acceptances = accept_log_probs > random_draw
                src = torch.where(acceptances,resampled,src)
                
                # cache best sequence
                if resampled_likelihood - original_score > max_score_diff:
                    max_score_diff = resampled_likelihood - original_score
                    best_seq = resampled
                    best_step = s
                    probs = F.softmax(pred_classes,dim=-1)
                    print(f'found new best seq, P(<NC>) = {probs[:,:,5].item():.3f}, P(<PC>) = {probs[:,:,4].item():.3f}')
                    #if probs[:,:,self.class_token].item() > 0.95:
                    #    break
                 
                if s % 10 == 0:
                    best_onehot = self.onehot_embed_layer(src)
                    pred_classes = self.predictor(best_onehot,src_lens,self.decoder_input(batch_size),batch_size)
                    best_pred = pred_classes[:,:,self.class_token]
                    best_likelihood = self.class_likelihood_ratio(pred_classes,self.class_token)
                    diff_original = torch.count_nonzero(src != original)
                    print(f'step={s}, {tscript}') 
                    print(f'# diff from original = {diff_original}/{src.shape[0]*src.shape[1]}')
                    probs = F.softmax(pred_classes,dim=-1)
                    print(f'P(<NC>) = {probs[:,:,5].item():.3f}, P(<PC>) = {probs[:,:,4].item():.3f}')
                    score_improvement = best_likelihood - original_score
                    print(f'score improvement = {score_improvement.item():.3f}')
                    if probs[:,:,self.class_token].item() > 0.50:
                        raw_src = self.get_raw_src(best_seq)
                        print(''.join(raw_src))
                    if probs[:,:,self.class_token].item() > 0.95:
                        break
                    end = time.time()
                    print(f'Time elapsed = {end-start} s')
                    start = time.time() 
            sep = '________________________________________'
            print(sep)
            print(f"The sequence has converged, found at step {best_step}")
            diff_original = torch.count_nonzero(best_seq != original)
            print(f'# diff from original = {diff_original}/{best_seq.shape[0]*best_seq.shape[1]}')
            raw_src = self.get_raw_src(best_seq)
            print(''.join(raw_src))
            best_onehot = self.onehot_embed_layer(best_seq)
            pred_classes = self.predictor(best_onehot,src_lens,self.decoder_input(batch_size),batch_size)
            probs = F.softmax(pred_classes,dim=-1)
            print(f'P(<NC>) = {probs[:,:,5].item():.3f}, P(<PC>) = {probs[:,:,4].item():.3f}')
            print(sep)
            description = f'step_found : {best_step}, diff:{diff_original}/{best_seq.shape[0]*best_seq.shape[1]}, P(<NC>):{original_probs[:,:,5].item():.3f}->{probs[:,:,5].item():.3f}'
            rec = SeqRecord(Seq(''.join(raw_src)),id=f'{tscript}-MUTANT',description=description)
            mutant_records.append(rec)

        with open(savefile,'w') as out_file:
            SeqIO.write(mutant_records,out_file,'fasta')
