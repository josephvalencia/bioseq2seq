import logomaker
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import re,random
from utils import parse_config,build_EDA_file_list,load_CDS,setup_fonts, ScalarFormatterClass
import math
import matplotlib.ticker as mtick

def update_percentile_scores(running_percentile,scores,n_bins):

    cuts = np.linspace(0,scores.shape[1]-1,n_bins+1)
    cuts = [math.ceil(c) for c in cuts]
    percentile_splits = np.split(scores,cuts[1:-1],axis=1)
    averaged = [x.mean(axis=1) for x in percentile_splits]
    return running_percentile + np.asarray(averaged).T

def is_valid(start,end,total,n_bins):
    return start >= n_bins and (end-start) >= n_bins and (total-end) >= n_bins

def metagene(cds_storage,saved_file,n_bins,mode):

    valid_pc = 0 
    valid_nc = 0 
   
    # initialize sum for mRNAs
    layer_count = 8 if mode == 'EDA' else 1 
    five_prime = np.zeros((layer_count,n_bins)) 
    cds = np.zeros((layer_count,n_bins))
    three_prime = np.zeros((layer_count,n_bins)) 
    # initialize sum for lncRNAs
    upstream = np.zeros((layer_count,n_bins)) 
    longest_orf = np.zeros((layer_count,n_bins))
    downstream = np.zeros((layer_count,n_bins)) 
    
    onehot = None
    saved = np.load(saved_file)
    if mode == 'Taylor':
        onehot = np.load(saved_file.replace('grad','onehot'))
    for tscript, attr in saved.items():
        is_pc =  tscript.startswith('NM') or tscript.startswith('XM')
        if mode == 'Taylor':
            attr = attr - onehot[tscript] * attr
            attr = attr[:,2:6]
            attr = np.mean(np.abs(attr),axis=1,keepdims=True)
            attr = attr.T
        elif mode == 'MDIG' or mode == 'ISM':
            attr = np.mean(np.abs(attr),axis=1,keepdims=True)
            attr = attr.T
        if tscript in cds_storage:
            # case 1 : protein coding  case 2 : non coding 
            cds_loc = cds_storage[tscript]
            if cds_loc != "-1" :
                splits = cds_loc.split(":")
                clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                splits = [clean(x) for x in splits]
                start,end = tuple([int(x) for x in splits])
                if is_valid(start,end,attr.shape[1],n_bins): 
                    # divide by functional area
                    if is_pc: 
                        valid_pc +=1                         
                        five_prime = update_percentile_scores(five_prime,attr[:,:start],n_bins) 
                        cds = update_percentile_scores(cds,attr[:,start:end],n_bins)
                        three_prime = update_percentile_scores(three_prime,attr[:,end:],n_bins) 
                    else:
                        valid_nc +=1                         
                        upstream = update_percentile_scores(upstream,attr[:,:start],n_bins) 
                        longest_orf = update_percentile_scores(longest_orf,attr[:,start:end],n_bins)
                        downstream = update_percentile_scores(downstream,attr[:,end:],n_bins) 

    print(f'# valid = {valid_pc} mRNAs, {valid_nc} lncRNAs')
    total_pc = np.concatenate([five_prime,cds,three_prime],axis=1) / valid_pc
    total_nc = np.concatenate([upstream,longest_orf,downstream],axis=1) / valid_nc
    return total_pc, total_nc

def plot_EDA_metagene(cds_storage,output_dir,prefix,attribution_dict):

    file_list =  attribution_dict['path_list']
    model = attribution_dict['model']
    attr_type = attribution_dict['attr']
    n_layers = attribution_dict['n_layers']
    labels = [f'layer_{i}' for i in range(n_layers)]
    consensus = []
    
    n_bins = 25
    for l in range(n_layers):
        layer = file_list[l] 
        total_pc,total_nc = metagene(cds_storage,layer,n_bins,mode='EDA') 
        name = f'{prefix}_EDA_layer{l}'
        plot_line_EDA(total_pc,total_nc,output_dir,name,n_bins)

def plot_attribution_metagene(cds_storage,output_dir,prefix,attr_filenames,mode,sharey=True):

    n_bins = 25
    #fig,axs = plt.subplots(1,len(attr_filenames.keys()),figsize=(6.5,2),sharey=sharey)
    fig,axs = plt.subplots(len(attr_filenames.keys()),1,figsize=(3,3.5),sharey=sharey)
  
    for ax, model in zip(axs,attr_filenames.keys()): 
        best_attr_file, other_attr_files = attr_filenames[model]
        
        # start with metagenes for the best replicate
        total_pc,total_nc = metagene(cds_storage,best_attr_file,n_bins,mode=mode) 
        ax.plot(total_pc.ravel(),linewidth=1,label='mRNA',color='tab:red')
        ax.plot(total_nc.ravel(),linewidth=1,label='lncRNA',color='tab:blue')
       
        # if alternate replicates are provided, plot them
        for fname in other_attr_files:
            total_pc,total_nc = metagene(cds_storage,fname,n_bins,mode=mode) 
            ax.plot(total_pc.ravel(),linewidth=0.5,alpha=0.7,color='tab:red')
            ax.plot(total_nc.ravel(),linewidth=0.5,alpha=0.7,color='tab:blue')
        
        # mark the CDS with vertical lines
        ax.axvline(x=n_bins, color='black', linestyle=':')     
        ax.axvline(x=2*n_bins-1, color='black', linestyle=':')
        
        ylabel = r'abs($\Delta$S)'
        if mode != 'ISM': 
            ylabel+=f'-{mode}'
        ax.set_ylabel(ylabel)
        ax.set_xticks([0,3*n_bins-1],['5\'','3\''])
        ax.tick_params(axis='x',length=0)
        ax.tick_params(axis='y',labelleft=True)
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax.yaxis.set_major_formatter(yScalarFormatter)
        ax.set_title(model, loc='right',fontsize=10)
        sns.despine(ax=ax)
    
    axs[0].legend(title=f'{mode} metagene',loc="lower left",bbox_to_anchor=(0.0,1.1),ncol=2,fontsize=8,title_fontsize=8)
    plt.tight_layout()
    plt_filename = f'{output_dir}/{prefix}_{mode}_metagene.svg'
    print(f'Saving {plt_filename}')
    plt.savefig(plt_filename)
    plt.close()

def plot_line_EDA(total_pc,total_nc,output_dir,name,n_bins,attr_type='EDA'):
    
    fig,axes = plt.subplots(2,4,figsize=(7.5,3.5))
    for i,ax in enumerate(axes.flat): 
        
        if i == 0:
            ax.plot(total_pc[i,:],linewidth=1,label='mRNA',color='tab:red')
            ax.plot(total_nc[i,:],linewidth=1,label='lncRNA',color='tab:blue')
            ax.legend(loc='center',bbox_to_anchor=(0.5,1.5))
        else:
            ax.plot(total_pc[i,:],linewidth=1,color='tab:red')
            ax.plot(total_nc[i,:],linewidth=1,color='tab:blue')

        ax.axvline(x=n_bins, color='black', linestyle=':')     
        ax.axvline(x=2*n_bins-1, color='black', linestyle=':')
        ax.set_xticks([])
        ax.set_xticklabels([])
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax.yaxis.set_major_formatter(yScalarFormatter)
        ax.set_xlabel(f'head {i}')
        if i % 4 == 0:
            ax.set_ylabel(f'Mean {attr_type}')
   
    #handles, labels = axes[0,0].get_legend_handles_labels()
    plt.tight_layout()
    plt_filename = f'{output_dir}/{name}_metagene.svg'
    print(f'Saving {plt_filename}')
    plt.savefig(plt_filename)
    plt.close()

def run_consensus_pipeline(consensus,domain,output_dir,labels,name,model,attr_type,heatmap=False):

    plot_line(domain,consensus,output_dir,name,model,attr_type,plot_type='line',labels=labels)

def load_all_replicates(models):
    model_list = []
    with open(models) as inFile:
        for x in inFile.readlines():
            model_dir = x.rstrip().replace('/','').replace('.pt','')
            model_list.append(model_dir)
    return model_list

def load_models(best_replicate_dir,all_replicates_dir_file,suffix):
    ''' From a config file, return the filenames of all replicates as tuple(best_rep,list(other_reps))'''
    parent = os.path.split(os.path.split(best_replicate_dir)[0])[0] 
    best_replicate = os.path.join(best_replicate_dir,suffix) 
    other_reps = load_all_replicates(all_replicates_dir_file)
    other_reps = [os.path.join(parent,x,suffix) for x in other_reps if x != best_replicate]
    return best_replicate, other_reps


def build_all(args):

    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    train_file = os.path.join(args.data_dir,args.train_prefix+'.csv')
    val_file = os.path.join(args.data_dir,args.val_prefix+'.csv')
    test_cds = load_CDS(test_file)
    val_cds = load_CDS(val_file)
    train_cds = load_CDS(train_file)
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
   
    prefix = args.train_prefix.replace('train','train_RNA')
    #best_BIO_grad = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.grad.npz')
    #best_EDC_grad = os.path.join(args.best_EDC_DIR,f'{prefix}.{args.reference_class}.{args.position}.grad.npz')
    #best_BIO_MDIG = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.MDIG.npz')
    #best_EDC_MDIG = os.path.join(args.best_EDC_DIR,f'{prefix}.{args.reference_class}.{args.position}.MDIG.npz')
    #build_consensus_attribution(test_cds,output_dir,f'best_seq2seq_{prefix}_{args.reference_class}_{args.position}_grad',best_BIO_grad)
    
    prefix = f'verified_test_RNA.{args.reference_class}.{args.position}'
    best_BIO_grad = load_models(args.best_BIO_DIR,args.all_BIO_replicates,f'{prefix}.grad.npz')
    best_EDC_grad = load_models(args.best_EDC_DIR,args.all_EDC_replicates,f'{prefix}.grad.npz')
    best_BIO_MDIG = load_models(args.best_BIO_DIR,args.all_BIO_replicates,f'{prefix}.MDIG.max_0.50.npz')
    best_EDC_MDIG = load_models(args.best_EDC_DIR,args.all_EDC_replicates,f'{prefix}.MDIG.max_0.10.npz')
    best_BIO_ISM = load_models(args.best_BIO_DIR,args.all_BIO_replicates,f'{prefix}.ISM.npz')
    best_EDC_ISM = load_models(args.best_EDC_DIR,args.all_EDC_replicates,f'{prefix}.ISM.npz')
    

    #plot_attribution_metagene(train_cds,output_dir,f'best_seq2seq_{prefix}_{args.reference_class}_{args.position}_MDIG',best_BIO_MDIG,'MDIG')
    #plot_attribution_metagene(train_cds,output_dir,f'best_seq2seq_{prefix}_{args.reference_class}_{args.position}_grad',best_BIO_grad,'grad')
    attr_filenames = {'bioseq2seq' : best_BIO_grad,'EDC' : best_EDC_grad}
    plot_attribution_metagene(test_cds,output_dir,prefix,attr_filenames,'Taylor')
    attr_filenames = {'bioseq2seq' : best_BIO_MDIG,'EDC' : best_EDC_MDIG}
    plot_attribution_metagene(test_cds,output_dir,prefix,attr_filenames,'MDIG',sharey=False)
    attr_filenames = {'bioseq2seq' : best_BIO_ISM,'EDC' : best_EDC_ISM}
    plot_attribution_metagene(test_cds,output_dir,prefix,attr_filenames,'ISM')

    # build EDA consensus, both coding and noncoding
    #plot_EDA_metagene(test_cds,output_dir,'best_seq2seq_test',best_BIO_EDA)
    #plot_EDA_metagene(test_cds,output_dir,'best_EDC_test',best_EDC_EDA)
    
if __name__ == "__main__":
    
    args,unknown_args = parse_config()
    setup_fonts() 
    
    #sns.set_style(style="whitegrid",rc={'font.family' : ['Helvetica']})
    build_all(args)
