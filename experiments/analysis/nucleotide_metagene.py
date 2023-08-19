import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from utils import parse_config,build_EDA_file_list,load_CDS,ScalarFormatterClass,setup_fonts

def summarize_head(cds_storage,saved_file,grad=False,align_on="start",coding=True,exclude=False):

    samples = []
    sample_ids = []
    before_lengths = []
    after_lengths = []

    saved = np.load(saved_file)
    
    if exclude:
        homology = pd.read_csv("test_maximal_homology.csv")
        reduced = homology['score'] <=80
        homology = homology.loc[reduced]
        allowed = set()
        allowed.update(homology['ID'].tolist())
    
    for tscript, attr in saved.items():
        is_pc = lambda x : x.startswith('NM_') or x.startswith('XM_')
        if exclude and tscript not in allowed:
            continue
        #attr = attr.T
        if grad:
            attr = attr.reshape(1,-1)
        if tscript in cds_storage:
            # case 1 : protein coding  case 2 : non coding 
            if (coding and is_pc(tscript)) or (not coding and not is_pc(tscript)): 
                cds = cds_storage[tscript]
                if cds != "-1" :
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    splits = [clean(x) for x in splits]
                    start,end = tuple([int(x) for x in splits])
                    # align relative to start or stop codon 
                    if align_on == "start":
                        before_lengths.append(start)
                        after_lengths.append(len(attr) - start)
                    elif align_on == "end":
                        before_lengths.append(end)
                        after_lengths.append(len(attr) - end)
                    else:
                        raise ValueError("align_on must be 'start' or 'end'")
                    sample_ids.append(tscript)
                    samples.append(attr)

    percentiles = [10*x for x in range(11)]
    after_percentiles = np.percentile(after_lengths,percentiles)
    before_percentiles = np.percentile(before_lengths,percentiles)
    max_before = max(before_lengths)
    max_after = max(after_lengths)
    domain = np.arange(-max_before,1200).reshape(1,-1)
    
    if align_on == "start":
        samples = [align_on_start(id,attn,start,max_before) for id,attn,start in zip(sample_ids,samples,before_lengths)]
    else:
        samples = [align_on_end(attn,end,max_after) for attn,end in zip(samples,before_lengths)]
    
    samples = np.stack(samples,axis=2)
    first = samples[0,:,:]
    support = np.count_nonzero(~np.isnan(first),axis=1)
    sufficient = support >= 0.70*first.shape[1]
    samples = samples[:,sufficient,:]
    domain = domain[:,sufficient]
    print(f'domain=[{domain.min()}, {domain.max()}]') 
    consensus = np.nanmean(samples,axis=2)
    return consensus.transpose(0,1),domain.ravel()

def build_consensus_EDA(cds_storage,output_dir,prefix,attribution_dict,coding=True,exclude=False):

    file_list =  attribution_dict['path_list']
    model = attribution_dict['model']
    attr_type = attribution_dict['attr']
    n_layers = attribution_dict['n_layers']
    labels = [f'layer_{i}' for i in range(n_layers)]
    consensus = []
    for l in range(n_layers):
        layer = file_list[l] 
        summary,domain  = summarize_head(cds_storage,layer,align_on="start",coding=coding,exclude=exclude) 
        consensus.append(summary.reshape(1,summary.shape[0],summary.shape[1]))
         
    consensus = np.concatenate(consensus,axis=0)
    
    suffix = "PC" if coding else "NC"
    name = f'{prefix}_{suffix}'
    run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,coding,attr_type='EDA')

def build_consensus_multi_IG(cds_storage,output_dir,prefix,grad_file,coding=True):

    '''
    file_list =  attribution_dict['path_list']
    model = attribution_dict['model']
    attr_type = attribution_dict['attr']
    labels = attribution_dict['bases']
    '''
    consensus,domain  = summarize_head(cds_storage,grad_file,grad=False,align_on="start",coding=coding) 
    consensus = consensus.reshape(1,consensus.shape[0],consensus.shape[1]) 
    
    suffix = "group=PC" if coding else "group=NC"
    name = f'{prefix}_{suffix}'
    model = 'bioseq2seq'
    attr_type = 'grad'
    compute_heatmap = True 
    labels = [str(x) for x in range(8)]
    run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,attr_type=attr_type,heatmap=compute_heatmap)

def plot_line(domain,consensus,output_dir,name,model,attr_type,coding,plot_type='line',plt_std_error=False,labels=None):
   
    fig = plt.figure(figsize=(5.5,2.25))

    palette = sns.color_palette()
    # hardcoded to see the head that loses periodicity 
    consensus = consensus[0,6,:].reshape(1,1,-1) 
    n_layers,n_heads,n_positions = consensus.shape
    if attr_type == 'EDA':
        for layer in range(n_layers):
            for i in range(n_heads): 
                label = layer if i % 8 == 0 else None
                color = layer % len(palette)
                plt.plot(domain,consensus[layer,i,:],color=palette[color],label=label,alpha=0.8,linewidth=1)
    else: 
        for layer in range(n_layers):
            for i in range(n_heads):
                plt.plot(domain,consensus[layer,i,:],color=palette[i],label=labels[i],alpha=0.8,linewidth=1)
    ax = plt.gca()
    legend_title = f'{model} {attr_type}'
    #plt.legend(title=legend_title,fontsize=8)
    
    # inset at 150-200
    inset_start = 50
    inset_stop = 100
    inset_domain = np.arange(inset_start,inset_stop)
    s = inset_start - domain.min()
    width = inset_stop - inset_start
    inset_range = consensus[:,:,s:s+width]
    axins = ax.inset_axes([0.4, 0.2, 0.5, 0.5])
    axins.axhline(y=0, color='gray', linestyle=':')     
    
    if attr_type == 'EDA':
        for layer in range(n_layers):
            for i in range(n_heads): 
                label = layer if i % 8 == 0 else None
                subslice = inset_range[layer,i,:]
                color = layer % len(palette)
                axins.plot(inset_domain,subslice,color=palette[color],label=label,alpha=0.4,linewidth=1.5)
    else:
        for layer in range(n_layers):
            for i in range(n_heads): 
                subslice = inset_range[layer,i,:]
                axins.plot(inset_domain,subslice,color=palette[i],label=labels[i],alpha=0.8,linewidth=1.5)
    
    ax.indicate_inset_zoom(axins, edgecolor="black")
    axins.yaxis.tick_right() 
    yScalarFormatter = ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((0,0))
    axins.yaxis.set_major_formatter(yScalarFormatter)
    axins.set_xticks([inset_start,inset_stop],[inset_start,inset_stop])
    axins.tick_params(axis='both',labelsize=8)
    plt.axhline(y=0, color='gray', linestyle=':')     
    sns.despine(fig=fig)
    
    if coding:
        plt.xlabel("Position relative to mRNA CDS")
    else:
        plt.xlabel("Position relative to lncRNA longest ORF")

    plt.ylabel(f"Mean {attr_type} Score")
    plt.tight_layout(rect=[0,0.03, 1, 0.95])
    plt_filename = f'{output_dir}/{name}_{attr_type}_{plot_type}plot.svg'
    print(f'Saving {plt_filename}')
    plt.savefig(plt_filename)
    plt.close()

def align_on_start(id,attn,cds_start,max_start,max_len=1200):
    
    max_len = 1200
    left_remainder = max_start - cds_start
    prefix = np.ones((attn.shape[0],left_remainder)) * np.nan
    right_remainder = max_len + max_start - (left_remainder+attn.shape[-1])
    suffix = np.ones((attn.shape[0],right_remainder)) * np.nan
    total = np.concatenate([prefix,attn,suffix],axis=1)
    return total

def align_on_end(attn,cds_end,max_end):

    max_len = 1200
    indices = list(range(len(attn)))
    indices = [x-cds_end for x in indices]

    left_remainder = max_len-cds_end
    prefix = [np.nan for x in range(left_remainder)]
    right_remainder = max_end - indices[-1]-1
    suffix = [np.nan for x in range(right_remainder)]
    total = prefix+attn+suffix
    return total

def plot_power_spectrum(consensus,output_dir,name,model,attr_type,units='freq',labels=None):

    palette = sns.color_palette()
    freq, ps = signal.welch(consensus,axis=2,scaling='density',average='median')
    plt.figure(figsize=(3.75,1.75))
    ax1 = plt.gca() 

    max_sum = ps.sum(axis=2)
    heads_with_max_sum = np.argmax(max_sum,axis=1)
    print(heads_with_max_sum,heads_with_max_sum.shape)
    
    n_layers, n_heads, n_freq_bins = ps.shape
    x_label = "Period (nt)" if units == "period" else "Frequency (cycles/nt)"

    if attr_type == 'EDA':
        for l in range(n_layers):
            for i in range(n_heads):
                label = l if i % 8 == 0 else None
                color = l % len(palette)
                #if (l == 0 and i == 6) or (l == 1 and i == 4): 
                #if (l == 0 and i == 0): #or (l == 1 and i == 4): 
                #ax1.plot(freq,ps[l,i,:],label=f'{l}-{i}',alpha=0.6)
                ax1.plot(freq,ps[l,i,:],color=palette[color],label=f'{l}-{i}',alpha=0.6)
    else:
        for l in range(n_layers):
            for i in range(n_heads):
                label = labels[i] if labels is not None else None
                ax1.plot(freq,ps[l,i,:],color=palette[i],label=label,alpha=0.6)
   
    tick_labels = ["0",r'$\frac{1}{10}$']+[r"$\frac{1}{"+str(x)+r"}$" for x in range(5,1,-1)]
    tick_locs =[0,1.0/10]+ [1.0 / x for x in range(5,1,-1)]
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labels)

    if attr_type == 'EDA':
        #ax1.legend(title=f'{model} attention layer')
        ax1.set_ylabel("Attn. PSD")
    else:
        ax1.legend(title=f'{model} {attr_type} baseline')
        ax1.set_ylabel(f"{attr_type} Power Spectrum")
    
    ax1.set_xlabel(x_label)
    yScalarFormatter = ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((0,0))
    ax1.yaxis.set_major_formatter(yScalarFormatter)
    plt.tight_layout()
    plt_filename = f'{output_dir}/{name}_{attr_type}_PSD.svg'
    print(f'Saving {plt_filename}')
    plt.savefig(plt_filename)
    plt.close()

def run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,coding,attr_type,heatmap=False):
    
    plot_line(domain,consensus,output_dir,name,model,attr_type,coding,plot_type='line',labels=labels)
    plot_power_spectrum(consensus,output_dir,name,model,attr_type=attr_type,labels=labels)

def build_all(args):

    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    train_file = os.path.join(args.data_dir,args.train_prefix+'.csv')
    val_file = os.path.join(args.data_dir,args.val_prefix+'.csv')
    test_cds = load_CDS(test_file)
    val_cds = load_CDS(val_file)
    df_test = pd.read_csv(test_file,sep='\t')
  
    # load attribution files from config
    best_BIO_EDA = build_EDA_file_list(args.best_BIO_EDA,args.best_BIO_DIR)
    best_EDC_EDA = build_EDA_file_list(args.best_EDC_EDA,args.best_EDC_DIR)
   
    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}/'
    if not os.path.isdir(output_dir):
        print("Building directory ...")
        os.mkdir(output_dir)
   
    # build EDA consensus, both coding and noncoding
    build_consensus_EDA(test_cds,output_dir,'best_seq2seq_test',best_BIO_EDA,coding=True,exclude=True)
    build_consensus_EDA(test_cds,output_dir,'best_seq2seq_test',best_BIO_EDA,coding=False,exclude=True)
    build_consensus_EDA(test_cds,output_dir,'best_EDC_test',best_EDC_EDA,coding=True,exclude=True)
    build_consensus_EDA(test_cds,output_dir,'best_EDC_test',best_EDC_EDA,coding=False,exclude=True)

if __name__ == "__main__":
    
    args,unknown_args = parse_config()
    setup_fonts() 
    build_all(args)
