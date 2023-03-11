import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

from utils import parse_config, setup_fonts

def load_checkpoint(model_path):
        
    checkpoint = torch.load(model_path, map_location='cpu')
    opts = checkpoint['opt']
    print(f'Loaded model {model_path} with parent {opts.checkpoint} and mode {opts.mode}')
    return checkpoint

def load_filter_layers(checkpoint):
        
    global_filters = []
    has_next = True
    i_layer = 0    
    while has_next:
        layer = f'encoder.fnet.{i_layer}.global_filter'
        if layer in checkpoint['model']:
            weight = checkpoint['model'][layer]
            global_filters.append(weight[None])
            i_layer += 1
        else:
            has_next = False

    return global_filters

def approximate_index(period,filter_size):
    frequency = 1.0 / period
    index = int(2*frequency*filter_size)
    return index

def process(weight,mode):
    
    if mode == 'abs':
        weight = weight.abs()
    elif mode == 'real':
        weight = weight.real
    elif mode == 'imag':
        weight = weight.imag
    else:
        weight = weight.angle()
    return weight

def plot_filters(global_filter_list,filename,mode,model):

    periods = [18,12,9,6,5,4,3,2]
    L = len(global_filter_list)
    fig, axs = plt.subplots(1,6,figsize=(5.5,3))

    mid = L // 2
    index_list = [0,1,mid-1,mid,L-2,L-1]
    #index_list = [0,mid,L-1]
    #index_list = list(range(L))
    
    gf_list = [process(x,mode) for x in global_filter_list]
    total = torch.stack(gf_list,dim=0)
    index_tensor = torch.tensor(index_list)
    total = torch.index_select(total,0,index_tensor)
    global_min = total.min().item()
    global_max = total.max().item()

    if mode == 'abs':
        vmin = 0.0
        vmax = global_max
    elif mode == 'phase':
        vmin = -3.14
        vmax = 3.14
    else:
        global_range = max(global_max,abs(global_min))
        vmin = -global_range
        vmax = global_range

    cmap = 'plasma' if mode == 'abs' else 'seismic'

    for j,i in enumerate(index_list):

        gf = gf_list[i].numpy().squeeze()
        im = axs[j].imshow(gf,cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
        tick_locs = [approximate_index(p,gf.shape[0]) for p in periods]

        axs[j].set_title(f'Layer {i}',fontsize=8)
        if j == 0:
            axs[j].set_ylabel("Period (nt)")
            axs[j].set_yticks(tick_locs)
            axs[j].tick_params(axis='y',length=3)
            axs[j].set_yticklabels(periods,fontsize=8)
        else:
            axs[j].set_yticks([])
            axs[j].set_yticklabels([])

        if j ==0 or j == len(index_list) -1:
            first_last = [0,gf.shape[1]]
            axs[j].set_xticks(first_last)
            axs[j].tick_params(axis='x',length=3)
            axs[j].set_xticklabels(first_last,fontsize=8)
        else:
            axs[j].set_xticks([])
    
    plt.tight_layout(w_pad=0.1)
    fig.supxlabel("embed. dim.",fontsize=10)

    if mode == 'phase':
        ticks=[-3.14,-1.57,0,1.57,3.14]
        cb = plt.colorbar(im,ax=list(axs.flat),ticks=ticks)
        cb.ax.set_yticklabels(['-π','-π/2','0','π/2','π'],fontsize=8)
    else:
        cb = plt.colorbar(im,ax=list(axs.flat)) 

    label = 'magnitude' if mode == 'abs' else 'phase'
    cb.ax.set_ylabel(f'{model} filter {label}',fontsize=10)
    cb.ax.tick_params(labelsize=8)
    plt.savefig(filename)
    print(f'saved {filename}')

def phase_hist(global_filter_list,filename):
        
    L = len(global_filter_list)
    fig, axs = plt.subplots(1,6,figsize=(6.5,1.5),sharey=True)

    gf_list = [process(x,'phase') for x in global_filter_list]
    index_list = list(range(L))
    mid = L // 2
    index_list = [0,1,mid-1,mid,L-2,L-1]
    for i,ax in zip(index_list,axs.flat):
        phase = gf_list[i]
        flat = phase.ravel()
        g = sns.histplot(flat,bins=np.linspace(-3.14,3.14,32),stat='density',ax=ax,color='navy')
        ax.set_xticks([-3.14,-1.57,0,1.57,3.14],fontsize=8)
        ax.set_xticklabels(['-π',r'-π/2','0',r'π/2','π'],fontsize=8)
        ax.set_title(f'Layer {i}',fontsize=8)
        ax.set_xlabel('Phase',fontsize=8)
        if i != 0:
            ax.tick_params(axis='y',length=0) 
            ax.tick_params(axis='x',length=2.5) 
    plt.tight_layout(w_pad=0.1)
    plt.savefig(filename)

if __name__ == "__main__":

    args,unknown_args = parse_config()
    setup_fonts()
    
    best_bio_checkpoint = load_checkpoint(args.best_BIO_chkpt)
    best_edc_checkpoint = load_checkpoint(args.best_EDC_chkpt)
    EDC_filters = load_filter_layers(best_edc_checkpoint)
    print(f'EDC filters loaded {abs}')
    BIO_filters = load_filter_layers(best_bio_checkpoint)
    print(f'bioseq2seq filters loaded {abs}')
    
    for mode in ['abs','real','imag','phase']:
        plot_filters(EDC_filters,f'LFNet_filters_EDC-large_best_{mode}.svg',mode=mode,model='EDC')
        plot_filters(BIO_filters,f'LFNet_filters_bioseq2seq_best_{mode}.svg',mode=mode,model='bioseq2seq')
    
    phase_hist(EDC_filters,'LFNet_filters_EDC-large_best_phase_hist.svg')
    phase_hist(BIO_filters,f'LFNet_filters_bioseq2seq_best_phase_hist.svg')
