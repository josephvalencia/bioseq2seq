import matplotlib
matplotlib.use('TKAgg')
import xml.etree.ElementTree as ET
import re,os, sys
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

from utils import parse_config, add_file_list, getLongestORF, get_CDS_start

args, unknown_args = parse_config()

def parse_streme(xml_file,head):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    motifs = None

    for child in root:
        if child.tag == "motifs":
            motifs = child
            break
    if motifs is not None:
        pval_results = []
        # iterate through motifs
        for m in motifs:
            attrib = m.attrib
            motif_id = attrib['id']
            pval = float(attrib['test_pvalue'])
            log_pval = float(attrib['test_log_pvalue'])
            if pval < 1e-5:
                storage = []
                for pos in m.iter('pos'):
                    storage.append(pos.attrib)
                prob_df = pd.DataFrame(storage)
                dtypes = {'A' : float , 'G' : float , 'C' : float , 'U' : float}
                prob_df = prob_df.astype(dtypes)
                info_df = logomaker.transform_matrix(df=prob_df,from_type='probability',to_type='information')
                total_IC = info_df.values.sum()
                results = (head+'_'+motif_id,pval,total_IC,info_df) 
                pval_results.append(results)
        return pval_results
    else:
        print('No motifs found')
        return [head,1.0] 
        
def make_logo(df,layer,head,ax=None):

    # create figure
    num_cols = len(df)
    num_rows = 4
    height_per_row = 2*.8
    width_per_col = 2*1.5
    figsize=[width_per_col * num_cols, height_per_row * num_rows]
    alpha = np.random.uniform(low=0.1,high=1.0) 
    logo = logomaker.Logo(df,figsize=figsize,ax=ax,alpha=1.0)#,color_scheme='colorblind_safe')
    logo.ax.set_ylim([0,2])
   
    if layer == 0 :
        logo.ax.set_ylabel(head)
        logo.ax.set_yticks([0,1,2])
        logo.ax.set_yticklabels([0,1,2])
    else:
        logo.ax.set_yticks([])
        logo.ax.set_yticklabels([])

    if head == 7:
        logo.ax.set_xlabel(layer)

    logo.ax.set_xticks([])
    logo.ax.set_xticklabels([])

def motifs_per_head_plot(results,trial,cmap,norm):

    fig,axs = plt.subplots(8,4,figsize=(16,8),sharex=False,sharey=True)
    print(f,trial,len(results))
    for s,p,ic,df in sorted(results,key=lambda x : x[1]):
        print(s,p,ic) 

    has_motif = set()
    
    for s,p,ic,df in sorted(results,key=lambda x: x[1]):
        match = re.match('layer(\d)head(\d)_',s)
        if match:
            l = int(match.group(1))
            h = int(match.group(2))
            #ax = plt.subplot2grid((8,4),(h,l))
            flat_loc = 4*h+l
            if flat_loc not in has_motif: 
                has_motif.add(flat_loc)
                ax = axs[h,l]
                color = cmap(norm(p))
                #ax.set_facecolor(color)
              
                width = 3
                # Convert bottom-left and top-right to display coordinates
                x0, y0 = ax.transAxes.transform((0, 0))
                x1, y1 = ax.transAxes.transform((1, 1))

                # Convert back to Axes coordinates
                x0, y0 = ax.transAxes.inverted().transform((x0, y0))
                x1, y1 = ax.transAxes.inverted().transform((x1, y1))

                rect = plt.Rectangle(
                    (x0, y0), x1-x0, y1-y0,
                    color=color,
                    transform=ax.transAxes,
                    zorder=-1,
                    lw=2*width+1,
                    fill=None,
                )
                fig.patches.append(rect)

                make_logo(df,l,h,ax=ax)

    name = f'{f}_{trial}_motifs.svg'
    for i,ax in enumerate(axs.flat):
        sns.despine(bottom=not ax.get_subplotspec().is_last_row(), left=not ax.get_subplotspec().is_first_col(), ax=ax)
        
        if i not in has_motif: 
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
        if ax.get_subplotspec().is_last_row():
            l = i % 4
            ax.set_xlabel(l)
        if ax.get_subplotspec().is_first_col():
            h = i // 4
            ax.set_ylabel(h)
    plt.savefig(name)
    plt.close()


def colorbar(sm):
     
    figure, axes = plt.subplots(figsize =(1.3, 6))
    figure.subplots_adjust(bottom = 0.5)
    figure.colorbar(sm,
         cax = axes, orientation ='vertical',
         label ='STREME p-value')
    name = 'pval_colorbar.svg'
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def motifs_per_layer_plot(results,layer,ax):

    #for s,p,ic,df in sorted(results,key=lambda x : x[1]):
    #    print(s,p,ic)
    pvals = [x[1] for x in results]
    if len(pvals) > 1:
        zeros = [0]*len(pvals)
        ax.axhline(color='black',lw=0.5,alpha=0)
        ax.scatter(pvals,zeros,marker='v')
        #ax.set_xlim(min(pvals)/10, max(pvals)*10)
        sns.despine(bottom=False,left=True, ax=ax)
        ax.set_yticks([])
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))


if __name__ == "__main__":

    # build logarithm colorbar
    cmap = sns.color_palette("Greys_r",as_cmap=True)
    norm = colors.LogNorm(vmin=1e-40, vmax=1) 
    sm = ScalarMappable(cmap=cmap, norm=norm)    
    colorbar(sm)
    
    args, unknown_args = parse_config()

    # ingest stored data
    test_file = args.test_csv
    train_file = args.train_csv
    val_file = args.val_csv
    df_test = pd.read_csv(test_file,sep="\t").set_index("ID")

    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}/'
    attr_dir  =  f'{output_dir}/attr/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # make subdir for attribution loci results  
    if not os.path.isdir(attr_dir):
        os.mkdir(attr_dir)

    best_BIO_EDA = add_file_list(args.best_BIO_EDA,'layers')
    best_EDC_EDA = add_file_list(args.best_EDC_EDA,'layers')
    best_BIO_grad_PC = args.best_BIO_grad_PC
    best_EDC_grad_PC = args.best_EDC_grad_PC
    best_BIO_grad_NC = args.best_BIO_grad_NC
    best_EDC_grad_NC = args.best_EDC_grad_NC

    groups = [['PC','NC'],['PC','PC'],['NC','NC']]
    cross_metrics = [['max','max'],['max','min'],['min','max'],['rolling-abs','rolling-abs'],['random','random']]
    same_metrics = [['max','min'],['max','random'],['min','random'],['rolling-abs','random']]

    for i,g in enumerate(groups):
        metrics = same_metrics if i>0 else cross_metrics
        for m in metrics:
            # g[0] and g[1] are transcript type for pos and neg sets
            # m[0] and m[1] are loci of interest for pos and neg sets
            a = f'{g[0]}-{m[0]}'
            b = f'{g[1]}-{m[1]}'
            assert a != b
            trial_name = f'{a}_{b}'
            
            # build directories
            #EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/all/'
            #BIO_dir = f'{attr_dir}/best_seq2seq_{trial_name}/all/'
            # run all IG bases for both models 
        
            for l,f in enumerate(best_BIO_EDA['path_list']):
                heads = []
                for h in range(8):
                    BIO_dir = f'{attr_dir}/best_seq2seq_{trial_name}/layer{l}head{h}/'
                    xml = f'{BIO_dir}streme_out/streme.xml'
                    heads.append(xml)
                summary_file = f'{attr_dir}/best_seq2seq_{trial_name}/layer{l}_summary.txt'
                aggregate_motifs(heads,summary_file)
            for l,f in enumerate(best_EDC_EDA['path_list']):
                heads = []
                for h in range(8):
                    EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/layer{l}head{h}/'
                    xml = f'{EDC_dir}streme_out/streme.xml'
                    heads.append(xml)
                summary_file = f'{attr_dir}/best_EDC_{trial_name}/layer{l}_summary.txt'
                aggregate_motifs(heads,summary_file)



