import logomaker
from utils import parse_config, load_CDS, build_output_dir, setup_fonts
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys,os

def plot(summary_df,signed_df,window,vlim,tallest,ref,window_name,tscript,save_dir,class_type,target_pos,attr,axis=None):

    
    left,right = window 
    domain = list(range(-left,right))
    full_width = 8.5
    width = full_width * len(domain) / 135 # total number of sliced positions
    print('width',width) 
    fig,axs = plt.subplots(2,2,figsize=(width,1.35),
                         gridspec_kw={'height_ratios':[1.25,2],
                             'width_ratios': [35,1],
                             'wspace' : 0.05,
                             'hspace' :0.0})
    
    crp_logo = logomaker.Logo(summary_df,shade_below=.5,fade_below=.5,flip_below=True,ax=axs[0][0])
    crp_logo.style_spines(visible=False)
  
    target_pos = int(target_pos)

    crp_logo.ax.axvspan(-0.5, 2.5, color='green', alpha=0.3)
    if target_pos > 2: 
        left = 3*(target_pos-2)-0.5
        right = left+2.5
        crp_logo.ax.axvspan(left,right, color='red', alpha=0.3)
    
    crp_logo.ax.set_xticks([])
    crp_logo.ax.set_xticklabels([])
    crp_logo.ax.set_yticks([])
    crp_logo.ax.set_yticklabels([])
    crp_logo.ax.set_ylim(0,tallest)
    
    short_title = attr.split('.')[0]
    g = sns.heatmap(signed_df,
                    cmap='RdBu_r',
                    linewidths=0,
                    cbar=True,
                    cbar_ax=axs[1][1],
                    center=0,
                    vmin=-vlim,
                    vmax=vlim,
                    ax=axs[1][0])
    
    g.yaxis.set_ticklabels('ACGU',rotation='horizontal',fontsize=8)
    g.tick_params(axis='x',length=2)
    g.tick_params(axis='y',length=2) 
    g.set_xticks([0,left,left+right]) 
    g.set_xticklabels([f'{-left}','0',f'+{right}'],rotation=0) 
    g.set_xlabel(f'{window_name}={ref}',fontsize=8)
    axs[1][1].set_title(short_title,fontsize=8,loc='right') 
    plt.subplots_adjust(bottom=0.3) 
    axs[0][1].axis('off')
    
    name = f'{tscript}_{window_name}' 
    plt_filename = f'{save_dir}/{name}_{class_type}.{target_pos}.{attr}.mutations_logo.svg'
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

def plot_line(domain,grads,name,save_dir,class_type,target_pos,attr):
   
    plt.figure(figsize=(6,3.2))
    labels = 'ACGT'
    for j in range(grads.shape[1]):
        plt.plot(domain,grads[:,j],linewidth=1,label=labels[j],alpha=0.6)
    
    plt_filename = f'{save_dir}/{name}_{class_type}.{target_pos}.{attr}_lineplot.svg'
    plt.legend()
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f'saved {plt_filename}')
    plt.close()

def plot_examples(save_dir,savefile,onehot_file,test_csv,attr_type,args):

    saved = np.load(savefile)
    test_cds = load_CDS(test_csv)
    df_test = pd.read_csv(test_csv,sep='\t')
    onehot_seqs = np.load(onehot_file)
   
    # top 5 closest to median reproducibility for each class
    examples = ['NR_109777.1', 'NR_105045.1', 'NR_135529.1', 'NR_122105.1', 'NR_126388.1', 'NM_001009141.1', 'NM_001257433.1', 'NM_001015628.1', 'NM_001891.3', 'NM_001164444.2']
    
    for i,(tscript,attr) in enumerate(saved.items()):
        if tscript in examples: 
            cds_loc = [int(x) for x in test_cds[tscript].split(':')] 
            s,e = tuple(cds_loc) 
            labels = ['A','C','G','U']
            onehot = onehot_seqs[tscript][:,2:6]
            
            if attr_type == 'Taylor':
                attr = attr[:,2:6]
                attr -= onehot * attr

            # importance defined as magnitude of largest change possible towards counterfactual 
            is_coding = lambda x : x.startswith('XM') or x.startswith('NM') 
            if is_coding(tscript): 
                # mRNA -> NC
                importance = np.min(attr,axis=1,keepdims=True) 
                positive_mask = importance > 0.0
            else:
                # lncRNA -> PC
                importance = np.max(attr,axis=1,keepdims=True)
                positive_mask = importance < 0.0

            summary = np.abs(importance)
            tallest = np.max(summary) 
            tallest_loc = np.argmax(summary)
            
            print(summary.shape) 
            # enforce minimum heights
            min_ratio = 0.15
            pos_min_ratio = 0.10
            small_negative_mask = summary < min_ratio * tallest
            # 10% of top height for pos. with small negative val as min 
            summary = np.where(small_negative_mask,min_ratio*tallest,summary)
            # 5% of top height for pos. with positive val as min (if all mutations are positive this pos. is not well conserved) 
            summary = np.where(positive_mask,pos_min_ratio*tallest,summary)
            summary = onehot * summary
            vlim = np.nanmax(np.abs(attr))
            print(f'tallest: val = {tallest:.3f}, loc = {tallest_loc}, vlim = {vlim:.3f}')

            # window of (-start_left_offset,+start_right_offset) around the start codon
            start_left_offset = 18 if s>= 18 else s 
            start_right_offset = 45 if s+45 < attr.shape[0] else attr.shape[0] - s
            # window of (-stop_left_offset,+stop_right_offset) around the stop codon
            stop_left_offset = 18 if e-18 > s+start_right_offset else 0 
            stop_right_offset = 18 if e+18 < attr.shape[0] else attr.shape[0] - e
            # window of (-stop_left_offset,+stop_right_offset) around the stop codon
            max_left_offset = 18 if tallest_loc-18 > 0 else tallest_loc
            max_right_offset = 18 if tallest_loc+18 < attr.shape[0] else attr.shape[0] - tallest

            references = [s,e-3,tallest_loc]
            names = ['Start','Stop','Max']
            windows = [(start_left_offset,start_right_offset),(stop_left_offset,stop_right_offset),(max_left_offset,max_right_offset)]
            
            for window_name,window,ref in zip(names,windows,references):
                left,right = window 
                print(f'{window_name} rel={(-left,right)},abs={(ref-left,ref+right)}')
                domain = list(range(-left,right))
                local_attr = attr[ref-left:ref+right,:] 
                local_summary = summary[ref-left:ref+right,:]
                summary_df = pd.DataFrame(data=local_summary,index=domain,columns=labels)
                signed_df = pd.DataFrame(data=local_attr,index=domain,columns=labels).T
                plot(summary_df,signed_df,window,vlim,tallest,ref,window_name,tscript,save_dir,args.reference_class,args.position,attr_type)
            
            domain = list(range(-s,attr.shape[0]-s))
            plot_line(domain,attr,tscript,save_dir,args.reference_class,args.position,attr_type)

if __name__ == "__main__":
    
    args,unknown_args = parse_config() 
    setup_fonts() 
    output_dir = build_output_dir(args)

    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    best_BIO_ISM = os.path.join(args.best_BIO_DIR,f'verified_test_RNA.{args.reference_class}.{args.position}.ISM.npz')
    best_BIO_onehot = os.path.join(args.best_BIO_DIR,f'verified_test_RNA.{args.reference_class}.{args.position}.onehot.npz')
    plot_examples(output_dir,best_BIO_ISM,best_BIO_onehot,test_file,'ISM',args)


