import torch
from base import Attribution, PredictionWrapper
from bioseq2seq.bin.transforms import  getLongestORF
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import torch.nn.functional as F
import numpy as np

def plot(consensus_df,name,target_pos,class_token,axis=None):
    
    domain = list(range(-12,60))
    crp_logo = logomaker.Logo(consensus_df,shade_below=.5,fade_below=.5,flip_below=True,ax=axis)
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    threes = [x for x in domain if x % 3 == 0]
    
    crp_logo.ax.axvspan(-0.5, 2.5, color='green', alpha=0.3)
    if target_pos > 2:
        window_start = (target_pos-2)*3 
        crp_logo.ax.axvspan(window_start-0.5, window_start+2.5, color='red', alpha=0.3)
    crp_logo.ax.set_xticks(threes)
    crp_logo.ax.set_xticklabels(threes)
    crp_logo.ax.set_title(name)
   
    if class_token == '</s>':
        class_token = 'STOP'

    #plt_filename = f'{name}_ISM_{class_token}_{target_pos}_logo.svg'
    plt_filename = f'{name}_ISM_logo.svg'
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

def plot_examples(df):
    
    ncols = 2
    nrows = 2
    fig, axs = plt.subplots(nrows,ncols,sharey=False,figsize=(16,2))
    i = 0 
    
    labels = ['A','C','G','T']
    grad_df = pd.DataFrame(data=grad,index=labels,columns=list(range(grad.shape[1]))).T
    row = i // ncols
    col = i % ncols
    plot(grad_df,tscript,axis=axs[row,col])

    plt_filename = f'corrected_PC_logos.svg'
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

class InSilicoMutagenesis(Attribution):

    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,minibatch_size=None,times_input=False,smoothgrad=False):
        self.predictor = PredictionWrapper(model,softmax)
        super().__init__(model,device,vocab,tgt_class,softmax=softmax,sample_size=sample_size,minibatch_size=minibatch_size)
    
    def gen_point_mutations(self,raw_src,src,copies_per_step):

        baselines = []
        
        start,end = getLongestORF(''.join(raw_src))
        rel_start = -12
        rel_end = 60
        abs_start = start+rel_start
        abs_end = start+rel_end
        
        info = [] 
        bases =  ['A','G','C','T'] 
        int_bases = [torch.tensor([self.get_src_token(b)]) for b in bases]
        if abs_start >=0 and abs_end <= len(raw_src):
            for base,src_base in zip(bases,int_bases):
                for loc,rel_loc in zip(range(abs_start,abs_end),range(rel_start,rel_end)):
                    c = raw_src[loc]    
                    if base != c:
                        s_mutated = src.clone().squeeze()
                        s_mutated[loc] = src_base
                        s_mutated = s_mutated.reshape(-1,1)
                        diff = src - s_mutated
                        assert diff.sum() == (src[0,loc,0]-s_mutated[loc,0])
                        baselines.append(s_mutated)
                        info.append((rel_loc,base))

            upper = max(len(baselines),1)
            for i in range(0,upper,copies_per_step):
                chunk = baselines[i:i+copies_per_step]
                stacked = torch.stack(chunk,dim=0).to(device=src.device)
                yield stacked, info[i:i+copies_per_step]
        else:
            yield None
        
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_ism = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            
            tscript = transcript_names[ids[0]]
            src = src.transpose(0,1)
            raw_src = self.get_raw_src(src) 
             
            tgt_prefix_len = target_pos 
            
            start,end = getLongestORF(''.join(raw_src))
            codon_pos = start+3*(tgt_prefix_len-2)
            ground_truth = batch.tgt[tgt_prefix_len,0,0].item()
            class_token = ground_truth if self.class_token == 'GT' else self.class_token
            tgt_prefix = batch.tgt[:tgt_prefix_len,:,:]
            batch_size = batch.batch_size
            
            pred_classes = self.predictor(src,src_lens,self.decoder_input(batch_size,tgt_prefix),batch_size)
            class_logit, probs = self.predict_logits(src,src_lens,
                                                        self.decoder_input(batch_size,tgt_prefix),
                                                        batch_size,self.class_token,ratio=True)
            
            '''
            #print(f'GT({tscript}) : {ground_truth}, argmax : {torch.argmax(probs)}, class_token: {class_token}')
            
            probs = F.softmax(pred_classes,dim=-1)
            counterfactual = [x for x in range(pred_classes.shape[1]) if x != class_token]
            counter_idx = torch.tensor(counterfactual,device=pred_classes.device)
            counterfactuals = pred_classes.index_select(1,counter_idx)
            class_score = pred_classes[:,:,class_token] - counterfactuals.sum() 
            '''

            ism = []
            storage = []

            copies_per_step = 8
            for variant in self.gen_point_mutations(raw_src,src,copies_per_step):
                if variant is not None:
                    mutant,info = variant
                    B = mutant.shape[0]
                    '''
                    pred_classes = self.predictor(mutant,src_lens,self.decoder_input(B,tgt_prefix),B)
                    counterfactuals = pred_classes.index_select(1,counter_idx)
                    mutant_scores = pred_classes[:,:,class_token] - counterfactuals.sum(dim=-1)
                    '''
                    mutant_logit, probs = self.predict_logits(mutant,src_lens,
                                                                self.decoder_input(B,tgt_prefix),
                                                                B,self.class_token,ratio=True)
                    diff = mutant_logit - class_logit
                    diff = diff.detach().cpu().numpy()
                    for j in range(B):
                        entry = {'base' : info[j][1], 'loc' : info[j][0] , 'score' : diff[0][j]}
                        storage.append(entry)
            
            if len(storage) == 216:
                df = pd.DataFrame(storage)
                df = df.pivot(index='loc',columns='base',values='score').fillna(0.0)
                plot(df,tscript,target_pos,self.tgt_vocab.itos[class_token]) 
                all_ism[tscript] = df.to_numpy()
        
        print(f'saving {savefile}')
        np.savez_compressed(savefile,**all_ism) 
