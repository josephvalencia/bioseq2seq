#!/usr/bin/env python
import torch
import argparse
import pandas as pd
import random
import time
import numpy as np
import os
import json
import tqdm
import warnings
import copy
from scipy import stats , signal

from captum.attr import NoiseTunnel,DeepLiftShap,GradientShap,Saliency,InputXGradient,IntegratedGradients,FeatureAblation
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from bioseq2seq.modules import PositionalEncoding
from bioseq2seq.bin.models import restore_seq2seq_model
from bioseq2seq.bin.data_utils import iterator_from_fasta, build_standard_vocab, IterOnDevice, test_effective_batch_size
from bioseq2seq.inputters.corpus import maybe_fastafile_open
import bioseq2seq.bin.transforms as xfm
from bioseq2seq.bin.transforms import CodonTable
from argparse import Namespace

from analysis.average_attentions import plot_power_spectrum
from bioseq2seq.bin.transforms import getLongestORF 
import matplotlib.pyplot as plt


def plot_frequency_heatmap(tscript_name,attr,tgt_class,metric):

    periods = [18,9,6,5,4,3,2]
    tick_locs = [1.0 / p for p in periods]
   
    fft_window = 250
    # lower right NC spectrogram
    plt.subplot(212)
    Pxx, freqs, bins, im = plt.specgram(attr.ravel(),Fs=1.0,NFFT=fft_window,noverlap=fft_window/2,cmap='plasma',detrend = 'mean')
    bins = [int(x) for x in bins]
    plt.xlabel("Window center (nt.)")
    plt.ylabel("Period (nt.)")
    plt.colorbar(im).set_label('Attribution Power')
    plt.yticks(tick_locs,periods)
    plt.xticks(bins,bins)

    bins = [0]+bins+[attr.size]
    # upper left PC line plot
    plt.subplot(211)
    plt.plot(attr.ravel(),linewidth=1.0)
    plt.ylabel('Attribution')
    plt.xticks(bins,bins)
    '''
    plt.subplot(312)
    # set parameters for drawing gene
    exon_start = 50-.5
    exon_stop = 200+.5
    y = 0

    # draw gene
    plt.axhline(y, color='k', linewidth=1)
    plt.plot([exon_start, exon_stop],[y, y], color='k', linewidth=7.5, solid_capstyle='butt')
    ''' 
    plt.tight_layout() 
    plt.savefig(f'ps_examples/{tscript_name}_{tgt_class}_{metric}_spectrogram.svg')
    plt.close()

def print_pairwise_corrs(batch_attributions,steps):
    
    copy1 = batch_attributions[-2]
    copy2 = batch_attributions[-1]
    
    dist = np.linalg.norm(copy2 - copy1,ord='fro') / np.linalg.norm(copy2,ord='fro')
    print('% Distance between last two trials (Frobenius norm)',dist)
    
    stacked = np.stack(batch_attributions,axis=0)
    M = stacked.sum(axis=-1)
    cov = M @ M.T
    diag_term = np.linalg.inv(np.sqrt(np.diag(np.diag(cov))))
    pearson_corr = diag_term @ cov @ diag_term
   
    df = pd.DataFrame(pearson_corr,columns = steps,index=steps)
    print('pairwise Pearson corr by n_steps')
    print(df.round(3))

def extract_baselines(batch,batch_index,num_copies,copies_per_step):

    baselines = []
    baseline_shapes = []
    for i in range(num_copies):
        seq_name = f'src_shuffled_{i}'
        attr = getattr(batch,seq_name)
        baselines.append(attr[0])

    upper = max(num_copies,1)
    for i in range(0,upper,copies_per_step):
        #print(f'chunk[{i}:{i+copies_per_step}]')
        chunk = baselines[i:i+copies_per_step]
        stacked = torch.stack(chunk,dim=0).to(device=batch.src[0].device) 
        yield stacked[:,:,batch_index,:]

def add_synonymous_shuffled_to_vocab(num_copies,vocab_fields):

    for i in range(num_copies):
        shuffled_field = copy.deepcopy(vocab_fields['src'])
        seq_name = f'src_shuffled_{i}'
        shuffled_field.fields = [(seq_name,shuffled_field.base_field)] 
        vocab_fields[seq_name] = shuffled_field 

class PredictionWrapper(torch.nn.Module):
    
    def __init__(self,model,softmax):
        
        super(PredictionWrapper,self).__init__()
        self.model = model
        self.softmax = softmax 

    def forward(self,src,src_lens,decoder_input,batch_size):

        src = src.transpose(0,1)
        src, enc_states, memory_bank, src_lengths, enc_cache = self.run_encoder(src,src_lens,batch_size)

        self.model.decoder.init_state(src,memory_bank,enc_states)
        memory_lengths = src_lens
        #print(f'decoder_input = {decoder_input.shape} , memory_bank = {memory_bank.shape},memory_lengths={memory_lengths.shape}')
        scores, attn = self.decode_and_generate(
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=0)

        classes = scores
        
        return classes

    def run_encoder(self,src,src_lengths,batch_size):

        enc_states, memory_bank, src_lengths, enc_cache = self.model.encoder(src,src_lengths,grad_mode=False)
        #enc_states, memory_bank, src_lengths, enc_cache = self.model.encoder(src,src_lengths)

        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch_size) \
                                .type_as(memory_bank) \
                                .long() \
                                .fill_(memory_bank.size(0))

        return src, enc_states, memory_bank, src_lengths,enc_cache

    def decode_and_generate(self,decoder_in, memory_bank, memory_lengths, step=None):

        dec_out, dec_attn = self.model.decoder(decoder_in,
                                            memory_bank,
                                            memory_lengths=memory_lengths,
                                            step=step,grad_mode=True)

        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        
        scores = self.model.generator(dec_out.squeeze(0),softmax=self.softmax)
        return scores,attn

class FeatureAttributor:

    def __init__(self,model,device,vocab,method,tgt_class,softmax=True,sample_size=None):
        
        self.device = device
        self.model = model
        self.sample_size = sample_size
        self.softmax = softmax

        if method == "EG":
            self.run_fn = self.run_expected_gradients
        elif method == "ISM":
            self.run_fn = self.run_ISM
        elif method == "saliency":
            self.run_fn = self.run_saliency
        elif method == 'inputXgrad':
            self.run_fn = self.run_inputXgrad
        elif method == 'IG':
            self.run_fn = self.run_integrated_gradients

        self.tgt_vocab = vocab['tgt'].base_field.vocab
        self.sos_token = self.tgt_vocab['<s>']
        self.class_token = self.tgt_vocab[tgt_class]
        self.pc_token = self.tgt_vocab['<PC>']
        self.nc_token = self.tgt_vocab['<NC>']
        self.src_vocab = vocab["src"].base_field.vocab

        if method != "ISM":
            self.interpretable_emb = configure_interpretable_embedding_layer(self.model,'encoder.embeddings')
        self.predictor = PredictionWrapper(self.model,self.softmax)

    def src_embed(self,src):

        src_emb = self.interpretable_emb.indices_to_embeddings(src.permute(1,0,2))
        src_emb = src_emb.permute(1,0,2)
        return src_emb 

    def decoder_input(self,batch_size):

        return self.sos_token * torch.ones(size=(batch_size,1,1),dtype=torch.long).to(self.device)
   
    def run(self,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        self.run_fn(savefile,val_iterator,target_pos,baseline,transcript_names)
    
    def run_integrated_gradients(self,savefile,val_iterator,target_pos,baseline,transcript_names):

        global_attr = True
        #ig = NoiseTunnel(IntegratedGradients(self.predictor,multiply_by_inputs=global_attr))
        ig = IntegratedGradients(self.predictor,multiply_by_inputs=global_attr)
        sl = Saliency(self.predictor)

        with open(savefile,'w') as outFile:
            #for batch in tqdm.tqdm(val_iterator):
            for batch in val_iterator:
                
                ids = batch.indices.tolist()
                src, src_lens = batch.src
                src = src.transpose(0,1)
                
                # can only do one batch at a time
                batch_size = batch.batch_size
                for j in range(batch_size):
                    
                    curr_src = torch.unsqueeze(src[j,:,:],0)
                    curr_src_embed = self.src_embed(curr_src)
                    idx = torch.randperm(curr_src.nelement())
                    shuffled_src = curr_src.view(-1)[idx].view(curr_src.size())
                    
                    #baseline_embed = torch.zeros_like(curr_src_embed).to(self.device)
                    std = torch.std(curr_src_embed,unbiased=True).item()
                    #baseline_embed = std*torch.randn_like(curr_src_embed).to(self.device)
                    baseline_embed = self.src_embed(shuffled_src).to(self.device)
                
                    decoder_input = self.decoder_input(1) 
                    curr_tgt = batch.tgt[target_pos,j,:]

                    curr_src_lens = torch.max(src_lens)
                    curr_src_lens = torch.unsqueeze(curr_src_lens,0)
                    
                    pred_classes = self.predictor(curr_src_embed,curr_src_lens,decoder_input,1)
                    src_neuron = pred_classes[0,self.class_token]
                    src_pred,src_idx = pred_classes.data.cpu().max(dim=-1)
                    
                    pred_classes = self.predictor(baseline_embed,curr_src_lens,decoder_input,1)
                    base_neuron = pred_classes[0,self.class_token]
                    base_pred,base_idx = pred_classes.data.cpu().max(dim=-1)
                    
                    tgt_class = torch.tensor([[[self.class_token]]]).to(self.device)
                    
                    saved_src = np.squeeze(np.squeeze(curr_src.detach().cpu().numpy(),axis=0),axis=1).tolist()
                    saved_src = "".join([self.src_vocab.itos[x] for x in saved_src])
                    saved_src = saved_src.split('<blank>')[0]

                    tscript = transcript_names[ids[j]]
                    print(tscript)
                    batch_attributions = [] 
                    #steps = [2,4,8,16,32,64,128,256] #,512,1024,2048,4096] 
                    steps = [256,512,1024,2048]
                    samples = 8
                    samples_per_batch = 8
                    noise_std = 0.1*std

                    for s in steps:
                        
                        internal_batch = min(s,32)
                        attributions, convergence_delta = ig.attribute(inputs=curr_src_embed,
                                                    target=tgt_class,
                                                    baselines=baseline_embed,
                                                    n_steps=s,
                                                    method='riemann_middle',
                                                    #nt_type='smoothgrad',nt_samples=samples,nt_samples_batch_size=samples_per_batch,stdevs=noise_std,
                                                    internal_batch_size=internal_batch,
                                                    return_convergence_delta=True,
                                                    additional_forward_args = (curr_src_lens,decoder_input,1))
                       
                        attributions = np.squeeze(attributions.detach().cpu().numpy(),axis=0)
                        diff = src_neuron - base_neuron
                        attr_sum = np.sum(attributions)
                        mae = convergence_delta.abs().mean()
                        manual_mae = attr_sum - diff
                        summed = np.sum(attributions,axis=1)
                        batch_attributions.append(attributions)
                       
                        print(f'n_steps = {s}, mean absolute error = {mae}')
                        
                        '''
                        direction = curr_src_embed - baseline_embed
                        attribution_grid = []
                        # hand-rolled Riemann sum for comparison
                        for k in range(0,s):
                            midpoint_offset = (1/s)/2
                            interpolated = baseline_embed + (k/s+midpoint_offset ) * direction
                            #interpolated = baseline_embed + (k/s) * direction
                            grads = sl.attribute(inputs=interpolated,target=tgt_class,abs=False,additional_forward_args = (src_lens,decoder_input,batch_size))
                            print(k,torch.linalg.norm(grads.squeeze()))
                            attribution_grid.append(grads)

                        stacked = torch.stack(attribution_grid,dim=0)
                        accumulated = torch.mean(stacked,dim=0)
                        myIG = direction * accumulated 
                        myIG = np.squeeze(myIG.detach().cpu().numpy(),axis=0)
                        dist = np.linalg.norm(myIG - attributions,ord='fro') / np.linalg.norm(myIG,ord='fro')
                        captum_summed = attributions.sum(axis=1)
                        my_summed = myIG.sum(axis=1)
                        my_total = np.sum(myIG)
                        print(f'n_steps = {s}, mae={mae}, my MAE = {my_total-diff}')
                        print('% Distance between IG implementations (Frobenius norm)',dist)
                        print('Pearson corr between IG implementations',stats.pearsonr(captum_summed,my_summed))
                        ''' 
                    print_pairwise_corrs(batch_attributions,steps)
                    print()
                    summed = np.sum(attributions,axis=1)
                    normed = np.linalg.norm(attributions,2,axis=1)
                    
                    true_len = len(saved_src)
                    summed_attr = ['{:.3e}'.format(x) for x in summed.tolist()[:true_len]]
                    normed_attr = ['{:.3e}'.format(x) for x in normed.tolist()[:true_len]]

                    entry = {"ID" : "tscript" , "summed_attr" : summed_attr, "normed_attr" : normed_attr, "src" : saved_src}
                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")

    def run_expected_gradients(self,savefile,val_iterator,target_pos,baseline,transcript_names):

        sl = Saliency(self.predictor)
        storage = []

        for batch in tqdm.tqdm(val_iterator):
            
            ids = batch.indices.tolist()
            src, src_lens = batch.src
            src = src.transpose(0,1)
            
            # can only do one batch at a time
            batch_size = batch.batch_size

            for j in range(batch_size):
               
                # setup batch elements
                tscript = transcript_names[ids[j]]
                curr_src = torch.unsqueeze(src[j,:,:],0)
                curr_src_embed = self.src_embed(curr_src)
                curr_tgt = batch.tgt[target_pos,j,:]
                curr_tgt = torch.unsqueeze(curr_tgt,0)
                curr_tgt = torch.unsqueeze(curr_tgt,2)
                curr_src_lens = torch.max(src_lens)
                curr_src_lens = torch.unsqueeze(curr_src_lens,0)
               
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
                baseline_preds = []                     
                
                if not(tscript.startswith('XM') or tscript.startswith('NM')):
                    minibatch_size = 16
                    decoder_input = self.decoder_input(minibatch_size)
                    for y,baseline_batch in enumerate(extract_baselines(batch,j,self.sample_size,minibatch_size)):
                        # score baselines
                        baseline_embed = self.src_embed(baseline_batch)
                        print('baseline_embed',baseline_embed.shape,curr_src_lens)
                        baseline_pred_classes = self.predictor(baseline_embed,curr_src_lens,decoder_input,minibatch_size)
                        base_pred = baseline_pred_classes.detach().cpu()[:,tgt_class]
                        baseline_preds.append(base_pred)
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
                    baseline_preds = torch.cat(baseline_preds,dim=0)
                    mean_baseline_score = baseline_preds.mean()
                    diff = src_score - mean_baseline_score
                    my_mae = diff - np.sum(attributions)
                    print(f'my_mae ={my_mae},diff ={diff.item()}, sum ={np.sum(attributions)}')
                    
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
                        scores.append((codon,opt_codon,attr))
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
                   
                    # compare with optimal randomly found
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

    def run_saliency(self,savefile,val_iterator,target_pos,baseline,transcript_names):

        sl = NoiseTunnel(Saliency(self.predictor))
        self.run_grad(sl,savefile,val_iterator,target_pos,baseline,transcript_names)
    
    def run_inputXgrad(self,savefile,val_iterator,target_pos,baseline,transcript_names):

        sl = NoiseTunnel(InputXGradient(self.predictor))
        self.run_grad(sl,savefile,val_iterator,target_pos,baseline,transcript_names)
    
    def run_grad(self,sl,savefile,val_iterator,target_pos,baseline,transcript_names):
        
        with open(savefile,'w') as outFile:
            
            for batch in tqdm.tqdm(val_iterator):
                ids = batch.indices.tolist()
                src, src_lens = batch.src
                src = src.transpose(0,1)
                batch_size = batch.batch_size
                
                src_embed = self.src_embed(src)
                decoder_input = self.decoder_input(batch_size)

                samples = self.sample_size
                samples_per_batch = 16
                # set noise level as pct of src 
                mean = torch.mean(src_embed).item()
                std = torch.std(src_embed,unbiased=True).item()
                noise_std = 0.2*std
               
                # grad wrt <PC>
                pc_attributions = sl.attribute(inputs=src_embed,target=self.pc_token,abs=False,
                                    nt_type='smoothgrad',nt_samples=samples,nt_samples_batch_size=samples_per_batch,stdevs=noise_std,
                                    additional_forward_args = (src_lens,decoder_input,batch_size))
                # wrt <NC> 
                nc_attributions = sl.attribute(inputs=src_embed,target=self.nc_token,abs=False,
                                    nt_type='smoothgrad',nt_samples=samples,nt_samples_batch_size=samples_per_batch,stdevs=noise_std,
                                    additional_forward_args = (src_lens,decoder_input,batch_size))

                mean = torch.mean(pc_attributions).item()
                std = torch.std(pc_attributions,unbiased=True).item()
                
                # summarize
                pc_attributions = pc_attributions.detach().cpu().numpy()
                pc_summed = np.sum(pc_attributions,axis=2)
                pc_normed = np.linalg.norm(pc_attributions,1,axis=2)
                nc_attributions = nc_attributions.detach().cpu().numpy()
                nc_summed = np.sum(nc_attributions,axis=2)
                nc_normed = np.linalg.norm(nc_attributions,1,axis=2)
                
                ''' 
                # sanity check on inputXgrad
                print(f'attr ={pc_attributions.shape}  mean+-std =  {mean}+={std}') 
                noise = torch.randn(16,*src_embed.shape).to(self.device) * noise_std
                noisy_input = src_embed + noise
                ctrl_grads = torch.randn_like(noisy_input).to(self.device)*mean + std
                ctrl_attribution = ctrl_grads*noisy_input
                ctrl_attribution = ctrl_attribution.mean(dim=0)
                ctrl_attribution = ctrl_attribution.detach().cpu().numpy()
                summed_ctrl = np.sum(ctrl_attribution,axis=2)
                 
                # test for periodicity in raw input 
                input_raw = src.squeeze(dim=-1).detach().cpu().numpy()
                '''
                saved_src = src.detach().cpu().numpy()
                for j in range(batch_size):
                    tscript = transcript_names[ids[j]]
                    curr_saved_src = "".join([self.src_vocab.itos[x] for x in saved_src[j,:,0]])
                    curr_saved_src = curr_saved_src.split('<blank>')[0]
                    true_len = len(curr_saved_src)
                    
                    ''' 
                    freq,ps = signal.welch(summed,axis=1,scaling='density',average='mean')
                    start = 83
                    end = start+6
                    three_region = np.sum(ps[0,start:end])
                    noise = np.sum(ps) - three_region
                    snr = three_region/noise
                    print('summed shape',summed.shape)
                    print('tscript',tscript)
                    print('SNR of 3nt',snr)
                    
                    cross_corr = signal.correlate(summed.ravel(),nc_summed.ravel())
                    lags = signal.correlation_lags(summed.size, nc_summed.size)
                    cross_corr /= np.max(cross_corr)
                    plt.plot(lags,cross_corr,label='cross')
                    
                    pc_auto_corr = signal.correlate(summed.ravel(),summed.ravel())
                    pc_auto_corr /= np.max(pc_auto_corr)
                    plt.plot(lags,pc_auto_corr,label='auto_PC')

                    nc_auto_corr = signal.correlate(nc_summed.ravel(),nc_summed.ravel())
                    nc_auto_corr /= np.max(nc_auto_corr)
                    plt.plot(lags,nc_auto_corr,label='auto_NC')

                    plt.xlim(-9,9)
                    plt.xlabel('Lag')
                    plt.xlabel('Correlation')
                    plt.legend()
                    plt.savefig(f'ps_examples/{tscript}_inputXgrad_class_corr.svg')
                    plt.close()
                    
                    plot_power_spectrum(summed_ctrl.transpose(1,0),'ps_examples',tscript,'bioseq2seq','ctrl_inputXgrad',units='freq',labels=None)
                    plot_power_spectrum(input_raw.transpose(1,0),'ps_examples',tscript,'bioseq2seq','raw_input',units='freq',labels=None)
                    pc_total = np.sum(summed)
                    nc_total = np.sum(nc_summed)
                    print(f'{tscript} PC total ={pc_total} ,NC total ={nc_total}')

                    pc_adjusted = summed * normed
                    nc_adjusted = nc_summed * nc_normed
                    diff = pc_adjusted - nc_adjusted
                    plot_power_spectrum(pc_adjusted,'ps_examples',tscript,'bioseq2seq','_PC_inputXgrad',units='freq',labels=['PC'])
                    plot_power_spectrum(nc_adjusted,'ps_examples',tscript,'bioseq2seq','_NC_inputXgrad',units='freq',labels=['NC'])
                    plot_power_spectrum(diff,'ps_examples',tscript,'bioseq2seq','_diff_inputXgrad',units='freq',labels=['diff'])
                    plot_frequency_heatmap(tscript,pc_adjusted,'PC','inputXgrad') 
                    plot_frequency_heatmap(tscript,nc_adjusted,'NC','inputXgrad') 
                    plot_frequency_heatmap(tscript,diff,'diff','inputXgrad') 
                    '''            
                    pc_summed_attr = ['{:.3e}'.format(x) for x in pc_summed[j,:].tolist()[:true_len]]
                    nc_summed_attr = ['{:.3e}'.format(x) for x in nc_summed[j,:].tolist()[:true_len]]
                    pc_normed_attr = ['{:.3e}'.format(x) for x in pc_normed[j,:].tolist()[:true_len]]
                    nc_normed_attr = ['{:.3e}'.format(x) for x in nc_normed[j,:].tolist()[:true_len]]
                    
                    entry = {"ID" : tscript , "summed_attr_PC" : pc_summed_attr, "summed_attr_NC" : nc_summed_attr,\
                            "normed_attr_PC" : pc_normed_attr , "normed_attr_NC" : nc_normed_attr ,"src" : curr_saved_src}
                    
                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")


    def run_ISM(self,savefile,val_iterator,target_pos,baseline):

        ism = FeatureAblation(self.predictor)
        
        with open(savefile,'w') as outFile:
            
            for batch in tqdm.tqdm(val_iterator):
                ids = batch.id
                src, src_lens = batch.src
                src = src.transpose(0,1)
                batch_size = batch.batch_size
                
                for j in range(batch_size):
                    curr_src = torch.unsqueeze(src[j,:,:],0)
                    curr_src_lens = torch.max(src_lens)
                    curr_src_lens = torch.unsqueeze(curr_src_lens,0)
                    decoder_input = self.decoder_input(1) 
                    curr_ids = batch.id
                    curr_tgt = batch.tgt[target_pos,j,:]
                    curr_tgt = torch.unsqueeze(curr_tgt,0)
                    curr_tgt = torch.unsqueeze(curr_tgt,2)
                    pred_classes = self.predictor(curr_src,curr_src_lens,decoder_input,batch_size)
                    pred,answer_idx = pred_classes.data.cpu().max(dim=-1)
                    tgt_class = torch.tensor([[[self.class_token]]]).to(self.device)
                    
                    # save computational time by masking non-mutated locations to run simultaneously 
                    base = self.src_vocab[baseline]
                    baseline_class = torch.tensor([[[base]]]).to(self.device)
                    baseline_tensor = base*torch.ones_like(curr_src).to(self.device)
                    num_total_el = torch.numel(curr_src)
                    mask = torch.arange(1,num_total_el+1).reshape_as(curr_src).to(self.device)
                    unchanged_indices = curr_src == baseline_class
                    feature_mask = torch.where(unchanged_indices,0,mask)  
                    
                    '''
                    attributions = ism.attribute(inputs=src_embed,baselines=baseline_embed,
                                        target=pc_class,feature_mask=feature_mask,
                                        additional_forward_args = (src_lens,decoder_input,batch_size))
                    '''
                    attributions = ism.attribute(inputs=curr_src,
                                            baselines=baseline_tensor,
                                            target=tgt_class,
                                            feature_mask=None,
                                            additional_forward_args = (curr_src_lens,decoder_input,1))
                    
                    attributions = attributions.detach().cpu().numpy().ravel().tolist()
                    saved_src = curr_src.detach().cpu().numpy().ravel().tolist()
                    
                    curr_saved_src = "".join([self.src_vocab.itos[x] for x in saved_src])
                    curr_saved_src = curr_saved_src.split('<pad>')[0]
                    true_len = len(curr_saved_src)
                    ism_attr = [f'{x:.3e}' for x in attributions[:true_len]]
                    
                    entry = {"ID" : ids[j] , "ism_attr" : ism_attr,"src" : curr_saved_src}
                    summary = json.dumps(entry)
                    outFile.write(summary+"\n")
def parse_args():

    """ Parse required and optional configuration arguments"""
    parser = argparse.ArgumentParser()
    
    # optional flags
    parser.add_argument("--verbose",action="store_true")

    # translate required args
    parser.add_argument("--input",help="File for translation")
    parser.add_argument("--checkpoint", "--c",help ="ONMT checkpoint (.pt)")
    parser.add_argument("--inference_mode",default ="combined")
    parser.add_argument("--attribution_mode",default="ig")
    parser.add_argument("--baseline",default="zero", help="zero|avg|A|C|G|T")
    parser.add_argument("--tgt_class",default="<PC>", help="<PC>|<NC>")
    parser.add_argument("--max_tokens",type = int , default = 4500, help = "Max number of tokens in training batch")
    parser.add_argument("--name",default = "temp")
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "Number of GPUs to use on node")
    parser.add_argument("--address",default =  "127.0.0.1",help = "IP address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "Port for master process in distributed training")
    parser.add_argument("--mutation_prob",type=float, default=1.0 ,help = "Prob of mutation")
    
    return parser.parse_args()

def run_helper(rank,args,model,vocab,use_splits=False):
    
    random_seed = 65
    random.seed(random_seed)
    random_state = random.getstate()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    #torch.use_deterministic_algorithms(True)
    target_pos = 1

    vocab_fields = build_standard_vocab()

    device = "cpu"
    # GPU training
    if args.num_gpus > 0:
        # One CUDA device per process
        device = torch.device("cuda:{}".format(rank))
        torch.cuda.set_device(device)
        model.cuda()
    
    # multi-GPU training
    if args.num_gpus > 1:
        # configure distributed training with environmental variables
        os.environ['MASTER_ADDR'] = args.address
        os.environ['MASTER_PORT'] = args.port
        torch.distributed.init_process_group(
            backend="nccl",
            init_method= "env://",
            world_size=args.num_gpus,
            rank=rank)
   
    gpu = rank if args.num_gpus > 0 else -1
    world_size = args.num_gpus if args.num_gpus > 0 else 1
    offset = rank if world_size > 1 else 0
   
    class_name = args.tgt_class.replace('<','').replace('>','')
    savefile = "{}.{}.{}.rank_{}".format(args.name,class_name,args.attribution_mode,rank)
    
    tscripts = []
    with maybe_fastafile_open(args.input) as fa:
        for i,record in enumerate(fa):
            if (i % world_size) == offset:
                    tscripts.append(record.id)
  
    # set up synonymous shuffled copies
    n_samples = 32
    shuffle_options = Namespace(num_copies=n_samples,mutation_prob=args.mutation_prob)
    xforms = {'add_synonymous_mutations' : xfm.SynonymousCopies(opts=shuffle_options)}
    add_synonymous_shuffled_to_vocab(n_samples,vocab_fields)

    valid_iter = iterator_from_fasta(src=args.input,
                                    tgt=None,
                                    vocab_fields=vocab_fields,
                                    mode=args.inference_mode,
                                    is_train=False,
                                    max_tokens=args.max_tokens,
                                    external_transforms=xforms, 
                                    rank=rank,
                                    world_size=world_size) 
    valid_iter = IterOnDevice(valid_iter,gpu)
   
    apply_softmax = False
    model.eval()
    #tgt_class = args.attribution_mode
    tgt_class = "<PC>"
    attributor = FeatureAttributor(model,device,vocab,args.attribution_mode,tgt_class,softmax=apply_softmax,sample_size=n_samples)
    attributor.run(savefile,valid_iter,target_pos,args.baseline,tscripts)

def run_attribution(args,device):
    
    checkpoint = torch.load(args.checkpoint,map_location = device)
    options = checkpoint['opt']
    vocab = checkpoint['vocab']
    model = restore_seq2seq_model(checkpoint,device,options)

    if not options is None:
        model_name = ""
        print("----------- Saved Parameters for ------------ {}".format("SAVED MODEL"))
        for k,v in vars(options).items():
            print(k,v)
 
    if args.num_gpus > 1:
        torch.multiprocessing.spawn(run_helper, nprocs=args.num_gpus, args=(args,model,vocab))
    elif args.num_gpus > 0:
        run_helper(args.rank,args,model,vocab)
    else:
        run_helper(0,args,model,vocab)

if __name__ == "__main__": 

    warnings.filterwarnings("ignore")
    args = parse_args()
    device = "cpu"
    run_attribution(args,device)
