import torch
from base import Attribution, PredictionWrapper
from bioseq2seq.bin.transforms import  getLongestORF
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import torch.nn.functional as F

def plot(consensus_df,name,axis=None):
    
    domain = list(range(-12,60))
    crp_logo = logomaker.Logo(consensus_df,shade_below=.5,fade_below=.5,flip_below=True,ax=axis)
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    threes = [x for x in domain if x % 3 == 0]
    
    crp_logo.ax.axvspan(0, 3, color='red', alpha=0.3)
    crp_logo.ax.set_xticks(threes)
    crp_logo.ax.set_xticklabels(threes)
    crp_logo.ax.set_title(name)
    
    plt_filename = f'{name}_logo.svg'
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

    def __init__(self,model,device,vocab,tgt_class,softmax=True,sample_size=None,batch_size=None,times_input=False,smoothgrad=False):
        self.predictor = PredictionWrapper(model,softmax)
        super().__init__(model,device,vocab,tgt_class,softmax=softmax,sample_size=sample_size,batch_size=batch_size)
    
    def gen_point_mutations(self,raw_src,src,copies_per_step):

        baselines = []
        
        start,end = getLongestORF(raw_src)
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
        
        all_grad = {}
        all_inputxgrad = {}
        
        for i,batch in enumerate(val_iterator):
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            
            tscript = transcript_names[ids[0]]
            if tscript.startswith('XR') or tscript.startswith('NR'):
                continue
            
            src = src.transpose(0,1)
            raw_src = self.get_raw_src(src) 
            
            class_token = 3 
            batch_size = batch.batch_size
            pred_classes = self.predictor(src,src_lens,self.decoder_input(batch_size,batch.tgt[:10,:,:]),batch_size)
            #pred_classes = self.predictor(src,src_lens,self.decoder_input(batch_size),batch_size)
            probs = F.softmax(pred_classes)
            print(f'GT({tscript}) : {torch.argmax(probs)}, W: {probs[:,class_token]}')
            
            counterfactual = [x for x in range(pred_classes.shape[1]) if x != class_token]
            counter_idx = torch.tensor(counterfactual,device=pred_classes.device)
            counterfactuals = pred_classes.index_select(1,counter_idx)
            class_score = pred_classes[:,class_token] - counterfactuals.sum() 
            
            ism = []
            storage = []

            copies_per_step = 16
            for variant in self.gen_point_mutations(raw_src,src,copies_per_step):
                if variant is not None:
                    mutant,info = variant
                    B = mutant.shape[0]
                    pred_classes = self.predictor(mutant,src_lens,self.decoder_input(B,batch.tgt[:10,:,:]),B)
                    counterfactuals = pred_classes.index_select(1,counter_idx)
                    mutant_scores = pred_classes[:,class_token] - counterfactuals.sum(dim=-1)
                    #mutant_scores = 2*pred_classes[:,class_token] - torch.sum(pred_classes,dim=1)
                    diff = mutant_scores - class_score
                    diff = diff.detach().cpu().numpy()
                    for j in range(B):
                        entry = {'base' : info[j][1], 'loc' : info[j][0] , 'score' : diff[j]}
                        storage.append(entry)
            
            if len(storage) == 216:
                df = pd.DataFrame(storage)
                df = df.pivot(index='loc',columns='base',values='score').fillna(0.0)
                plot(df,tscript) 
                print(tscript,df.iloc[12:18])

