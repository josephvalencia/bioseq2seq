import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import pandas as pd
from utils import parse_config, setup_fonts, build_output_dir
from mpl_toolkits.axes_grid1 import ImageGrid


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

def process(weight,mode,to_numpy=False):
    
    if mode == 'abs':
        weight = weight.abs()
    elif mode == 'real':
        weight = weight.real
    elif mode == 'imag':
        weight = weight.imag
    else:
        weight = weight.angle()
    
    if to_numpy:
        return weight.numpy()
    else:
        return weight

def plot_filters(global_filter_list,filename,mode,model):

    periods = [18,12,9,6,5,4,3,2]
    L = len(global_filter_list)
    fig, axs = plt.subplots(1,6,figsize=(5.5,3))

    mid = L // 2
    index_list = [0,1,mid-1,mid,L-2,L-1]
    
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


def calc_global_windows(gf_list):

    xmin = np.inf
    ymin = np.inf
    xmax = -np.inf
    ymax = -np.inf
    for x,y in gf_list:
        xmin = min(x.min(),xmin)
        xmax = max(x.max(),xmax)
        ymin = min(y.min(),ymin)
        ymax = max(y.max(),ymax)
  
    
    global_range = ((xmin,xmax),(ymin,ymax))
    extent = [xmin,xmax,ymin,ymax]
    return global_range,extent

def real_imag_hist_2D(global_filter_list,filename,mode):
        
    L = len(global_filter_list)
    fig, axs = plt.subplots(2,4,figsize=(6.5,4),sharey=True,sharex=True)

    gf_list = [(process(x,'real',True),process(x,'imag',True)) for x in global_filter_list]
    
    global_range,extent = calc_global_windows(gf_list) 
    index_list = list(range(L))
    mid = L // 2
    index_list = [0,1,L-2,L-1]
    three_storage = []
    remainder_storage = []

    for i in index_list:
        real,imag = gf_list[i]
        three_index = approximate_index(3,real.shape[1])
        # real
        real_three = real[:,three_index-1:three_index+1,:]
        real_remainder = np.delete(real,[three_index-1,three_index,three_index+1],axis=1) 
        flat_real_remainder = real_remainder.ravel()
        flat_real_three = real_three.ravel()
        # imag 
        imag_three = imag[:,three_index-1:three_index+1,:]
        imag_remainder = np.delete(imag,[three_index-1,three_index,three_index+1],axis=1) 
        flat_imag_remainder = imag_remainder.ravel()
        flat_imag_three = imag_three.ravel()
        H,xedges,yedges = np.histogram2d(x=flat_real_remainder,y=flat_imag_remainder,bins=100,range=global_range)
        remainder_storage.append(H)
        H2,xedges,yedges = np.histogram2d(x=flat_real_three,y=flat_imag_three,bins=100,range=global_range)
        three_storage.append(H2)
    three_vmin = np.min(three_storage)
    three_vmax = np.max(three_storage)
    remainder_vmax = np.max(remainder_storage)
    remainder_vmin = np.min(remainder_storage)
   
    for j,(i,three,remainder) in enumerate(zip(index_list,three_storage,remainder_storage)):
        cmap = 'binary'
        remainder_center = np.unravel_index(np.argmax(remainder,axis=None),remainder.shape)
        x,y = remainder_center
        bin_x = xedges[x]
        bin_y = yedges[y]
        full_remainder = f'max=({bin_x:.3f},{bin_y:.3f})'
        remainder_im = axs[0,j].imshow(remainder.T,vmin=remainder_vmin,vmax=remainder_vmax,extent=extent,cmap=cmap)

        three_center = np.unravel_index(np.argmax(three,axis=None),three.shape)
        x,y = three_center
        bin_x = xedges[x]
        bin_y = yedges[y]
        full_three = f'max=({bin_x:.3f},{bin_y:.3f})'
        three_im = axs[1,j].imshow(three.T,vmin=three_vmin,vmax=three_vmax,extent=extent,cmap=cmap)
        axs[0,j].set_title(f'Layer {i}',fontsize=10,pad=20)
        axs[1,j].set_xlabel('Real',fontsize=10)
        axs[0,j].text(0.0, 1.03,full_remainder,fontsize=8,transform = axs[0,j].transAxes)
        axs[1,j].text(0.0, 1.03,full_three,fontsize=8,transform=axs[1,j].transAxes)
        axs[1,j].axvline(color='green',lw=0.2)
        axs[1,j].axhline(color='green',lw=0.2)
        axs[0,j].axvline(color='green',lw=0.2)
        axs[0,j].axhline(color='green',lw=0.2)
         
    axs[0,0].set_ylabel('Imaginary',fontsize=8)
    axs[1,0].set_ylabel('Imaginary',fontsize=8)
    fig.colorbar(remainder_im,ax=axs[0,:],location='right')
    fig.colorbar(three_im,ax=axs[1,:],location='right')
    plt.tight_layout(w_pad=0.1)
    plt.savefig(filename)
    print(f'saved {filename}')

def phase_hist(global_filter_list,filename):
        
    L = len(global_filter_list)
    fig, axs = plt.subplots(1,6,figsize=(7.5,1.5),sharey=True)

    gf_list = [process(x,'phase',True) for x in global_filter_list]
    index_list = list(range(L))
    mid = L // 2
    index_list = [0,1,mid-1,mid,L-2,L-1]
    for i,ax in zip(index_list,axs.flat):
        phase = gf_list[i]
        three_index = approximate_index(3,phase.shape[1])
        three = phase[:,three_index-1:three_index+1,:]
        remainder = np.delete(phase,[three_index-1,three_index,three_index+1],axis=1) 
        flat_remainder = remainder.ravel()
        flat_three = three.ravel()
        label1 = None
        label2 = None
        if i == index_list[-1]:
            label1 = 'Other'
            label2 = '3 nt'
        g = sns.histplot(flat_remainder,bins=np.linspace(-3.14,3.14,32),stat='density',ax=ax,color='navy',alpha=0.6,label=label1)
        g = sns.kdeplot(flat_three,clip=(-3.14,3.14),cut=0,ax=ax,c='k',alpha=0.6,label=label2) 
        if i == index_list[-1]:
            ax.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
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
    output_dir = build_output_dir(args)

    best_bio_checkpoint = load_checkpoint(args.best_BIO_chkpt)
    best_edc_checkpoint = load_checkpoint(args.best_EDC_chkpt)
    EDC_filters = load_filter_layers(best_edc_checkpoint)
    print(f'EDC filters loaded {abs}')
    BIO_filters = load_filter_layers(best_bio_checkpoint)
    print(f'bioseq2seq filters loaded {abs}')
   
    for mode in ['abs','real','imag','phase']:
        plot_filters(EDC_filters,f'{output_dir}/LFNet_filters_EDC-large_best_{mode}.svg',mode=mode,model='EDC')
        plot_filters(BIO_filters,f'{output_dir}/LFNet_filters_bioseq2seq_best_{mode}.svg',mode=mode,model='bioseq2seq')

    phase_hist(EDC_filters,f'{output_dir}/LFNet_filters_EDC-large_best_phase_hist.svg')
    phase_hist(BIO_filters,f'{output_dir}/LFNet_filters_bioseq2seq_best_phase_hist.svg')
    real_imag_hist_2D(EDC_filters,f'{output_dir}/LFNet_filters_EDC-large_best_real_imag_hist.svg','EDC')
    real_imag_hist_2D(BIO_filters,f'{output_dir}/LFNet_filters_bioseq2seq_best_real_imag_hist.svg','bioseq2seq')
