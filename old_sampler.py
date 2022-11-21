import torch
from base import Attribution, OneHotGradientAttribution
        
class DiscreteLangevinSampler(OneHotGradientAttribution):

  
    def langevin_proposal_dist(self,grads,indexes,onehot):
        alpha = 0.1
        grad_current_char =  torch.gather(grads,dim=3,index=indexes.unsqueeze(2))
        mutation_scores = grads - grad_current_char
        temperature_term = (1.0 - onehot) / alpha 
        logits = 0.5*mutation_scores - temperature_term 
        mask = torch.tensor([-30,-30,1,1,1,1,-30,-30],device=logits.device)
        logits = logits * mask
        print(logits)
        proposal_dist = torch.distributions.Categorical(logits=logits.squeeze(2))
        return torch.distributions.Independent(proposal_dist,1)
  
    def metropolis_hastings(self,score_diff,forward_prob,reverse_prob):
        acceptance_log_prob = score_diff + reverse_prob - forward_prob
        return torch.minimum(acceptance_log_prob,torch.tensor([0.0],device=score_diff.device))
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            original = src
            original_score = None

            MAX_STEPS = 1000
            for s in range(MAX_STEPS): 
                onehot_src = self.onehot_embed_layer(src) 
                # calc q(x'|x)
                pred_classes = self.predictor(onehot_src,src_lens,self.decoder_input(batch_size),batch_size)
                #print(f'pred_classes = {pred_classes.shape}')
                true_pred = pred_classes[:,:,self.class_token]
                if original_score == None:
                    original_score = true_pred
                true_grads = self.input_grads(true_pred,onehot_src)
                proposal_dist = self.langevin_proposal_dist(true_grads,src,onehot_src)
                resampled = proposal_dist.sample().unsqueeze(2)
                diff = torch.count_nonzero(src != resampled)
                forward_prob = proposal_dist.log_prob(resampled.squeeze(2)) 
                
                # calc q(x|x')
                resampled_onehot = self.onehot_embed_layer(resampled)
                pred_classes = self.predictor(resampled_onehot,src_lens,self.decoder_input(batch_size),batch_size)
                resampled_pred = pred_classes[:,:,self.class_token]
                resampled_grads = self.input_grads(resampled_pred,resampled_onehot)
                reverse_proposal_dist = self.langevin_proposal_dist(resampled_grads,resampled,resampled_onehot)
                reverse_prob = reverse_proposal_dist.log_prob(src.squeeze(2)) 
              
                # correct with MH step
                accept_log_probs = self.metropolis_hastings(diff,forward_prob,reverse_prob) 
                random_draw = torch.log(torch.rand(src.shape[0],device=src.device))
                acceptances = accept_log_probs > random_draw
                #print(f'forward_prob = {torch.exp(forward_prob)}')
                #print(f'reverse_prob = {torch.exp(reverse_prob)}')
                #print(f'accept_prob = {torch.exp(accept_log_probs)}')
                #print(acceptances)
                src = torch.where(acceptances,resampled,src)
                diff_original = torch.count_nonzero(src != original)
                if s % 10 == 0:
                    # calc q(x|x')
                    best_onehot = self.onehot_embed_layer(src)
                    pred_classes = self.predictor(best_onehot,src_lens,self.decoder_input(batch_size),batch_size)
                    best_pred = pred_classes[:,:,self.class_token]
                    print(f'step={s}') 
                    print(f'# diff from original = {diff_original}/{src.shape[0]*src.shape[1]}')
                    print(f'score improvement = {best_pred - original_score}')
                
                #print(f'# proposed diff from previous = {diff}/{src.shape[0]*src.shape[1]}')
                #print(f'# accepts = {torch.count_nonzero(acceptances)}')

