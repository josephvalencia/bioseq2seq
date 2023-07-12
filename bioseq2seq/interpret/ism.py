import torch
from base import Attribution, PredictionWrapper
from bioseq2seq.bin.transforms import  getLongestORF
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import torch.nn.functional as F
import numpy as np
import time
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

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
    
    def gen_full_point_mutations(self,raw_src,src,copies_per_step):

        baselines = []
        
        start,end = getLongestORF(''.join(raw_src))
        
        info = [] 
        bases =  ['A','C','G','T'] 
        int_bases = [torch.tensor([self.get_src_token(b)]) for b in bases]
        for base,src_base in zip(bases,int_bases):
            for loc in range(len(raw_src)):
                c = raw_src[loc]    
                if base != c:
                    s_mutated = src.clone().squeeze()
                    s_mutated[loc] = src_base
                    s_mutated = s_mutated.reshape(-1,1)
                    diff = src - s_mutated
                    assert diff.sum() == (src[0,loc,0]-s_mutated[loc,0])
                    baselines.append(s_mutated)
                    info.append((loc,base))

        upper = max(len(baselines),1)
        for i in range(0,upper,copies_per_step):
            chunk = baselines[i:i+copies_per_step]
            stacked = torch.stack(chunk,dim=0).to(device=src.device)
            yield stacked, info[i:i+copies_per_step]
        else:
            yield None
    
    def gen_point_mutations(self,raw_src,src,copies_per_step):

        baselines = []
        
        start,end = getLongestORF(''.join(raw_src))
        rel_start = -12
        rel_end = 60
        abs_start = start+rel_start
        abs_end = start+rel_end
        
        info = [] 
        bases =  ['A','C','G','T'] 
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

    def safe_predict(self,mutant,src_lens,decoder_input,class_token):
        ''' recursively halve the batchsize until input does not cause OOM ''' 
        try:
            B = mutant.shape[0]
            with torch.no_grad():
                mutant_logit, probs = self.predict_logits(mutant,src_lens,
                                                            decoder_input,
                                                            B,class_token,ratio=True)
                return mutant_logit,probs
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                print(f'Recovering from OOM at batchsize={B}')
                mutant_chunks = mutant.chunk(2,dim=0)
                input_chunks = decoder_input[0].chunk(2,dim=0)
                total_logits = []
                total_probs = []
                for mut,item in zip(mutant_chunks,input_chunks): 
                    B = mutant.shape[0]
                    logit, probs = self.safe_predict(mut,src_lens,
						    (item,),class_token)
                    total_logits.append(logit)
                    total_probs.append(probs)
                return torch.cat(total_logits,dim=1), torch.cat(total_probs,dim=1) 
            else:
                raise RuntimeError(e)

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
            
            #pred_classes = self.predictor(src,src_lens,self.decoder_input(batch_size,tgt_prefix),batch_size)
            class_logit, probs = self.predict_logits(src,src_lens,
                                                        self.decoder_input(batch_size,tgt_prefix),
                                                        batch_size,class_token,ratio=True)
            ism = []
            storage = []
            fasta_storage = []
            save_fasta = False
            copies_per_step = self.minibatch_size
            s = time.time()
            count = 0 
            for variant in self.gen_full_point_mutations(raw_src,src,copies_per_step):
                count +=1
                if variant is not None:
                    mutant,info = variant
                    B = mutant.shape[0]
                    mutant_logit, probs = self.safe_predict(mutant,src_lens,
                                                            self.decoder_input(B,tgt_prefix),class_token)
                    diff = mutant_logit - class_logit
                    diff = diff.detach().cpu().numpy()
                    maxes = torch.max(probs,dim=2)
                    best_prob = maxes.values
                    best_pred = maxes.indices
                    curr = probs[:,:,class_token]
                    #print(f'original = {class_logit}, mutant = {mutant_logit}, diff = {diff}, best  = {curr}, info = {info}')
                    for j in range(B):
                        entry = {'base' : info[j][1], 'loc' : info[j][0] , 'score' : diff[0][j]}
                        storage.append(entry)
                        if save_fasta: 
                            src_entry = ''.join(self.get_raw_src(mutant[j,:].reshape(1,-1,1)))
                            variant_name=f'{tscript}-{info[j][1]}-{info[j][0]}'
                            description = f'P({self.get_tgt_string(best_pred[0,j])})={best_prob[0,j]:.3f}, delta={diff[0][j]}'
                            record = SeqRecord(Seq(src_entry),
                                        id=variant_name,
                                        description=description)
                            fasta_storage.append(record)
            e = time.time()
            print(f'{tscript} finished in {e-s} s.')
            df = pd.DataFrame(storage)
            df = df.pivot(index='loc',columns='base',values='score').fillna(0.0)
            all_ism[tscript] = df.to_numpy()
        
        print(f'saving {savefile}')
        np.savez_compressed(savefile,**all_ism)
       
        if save_fasta:
            with open('ism_results.fa','w') as outFile:
                SeqIO.write(fasta_storage, outFile, "fasta")


class LogitsOnly(Attribution):

    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,minibatch_size=None,times_input=False,smoothgrad=False):
        self.predictor = PredictionWrapper(model,softmax)
        super().__init__(model,device,vocab,tgt_class,softmax=softmax,sample_size=sample_size,minibatch_size=minibatch_size)
    
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        all_ref = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            batch_size = batch.batch_size
            tscript = transcript_names[ids[0]]

            # score true
            tgt_prefix = batch.tgt[:target_pos,:,:] 
            class_token = batch.tgt[target_pos,0,0].item() if self.class_token == 'GT' else self.class_token
            class_logit, probs = self.predict_logits(src,src_lens,
                                                        self.decoder_input(batch_size,tgt_prefix),
                                                        batch_size,class_token,ratio=True)
            print(f'bad = {1.0 - probs[0,0,self.pc_token].item() - probs[0,0,self.nc_token].item()}') 
            all_ref[tscript] = class_logit.detach().cpu().numpy() 
        
        print(f'saving {savefile}')
        np.savez_compressed(savefile,**all_ref) 
