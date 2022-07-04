import xml.etree.ElementTree as ET
import re,os, sys
import pandas as pd
import logomaker
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

def parse_streme(xml_file,head,trial_prefix):

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

def aggregate_motifs(streme_files,trial):

    all_files = ' '.join(streme_files)
    os.system(f'meme2meme {all_files} > {trial}_all_motifs.txt')

if __name__ == "__main__":

    #groups = [['PC','NC']]
    #metrics = [['max','max']]
    
    groups = [['PC','NC'],['PC','PC'],['NC','NC']]
    metrics = [['max','max'],['max','min'],['max','random'],['min','random'],['random','random']]
    models = ['seq2seq','EDC']
   
    # build logarithm colorbar
    cmap = sns.color_palette("Greys_r",as_cmap=True)
    norm = colors.LogNorm(vmin=1e-40, vmax=1) 
    sm = ScalarMappable(cmap=cmap, norm=norm)    
    colorbar(sm)
    
    for f in models:
        for g in groups:
            for m in metrics:
                a = f'{g[0]}-{m[0]}'
                b = f'{g[1]}-{m[1]}'
                # ensure comparison groups are different
                if a != b:
                    trial = f'{a}_{b}'

                    # IG attributions     
                    IG_results = []
                    for base in ['avg','zero']:
                        tgt_head = f'{base}_summed'                       
                        if f == 'EDC':
                            xml = f'new_attr/EDC_3_{trial}/EDC_3_{base}_pos_test_summed_attr/streme_out/streme.xml'
                            trial_prefix = f'new_attr/EDC_3_{trial}'
                            if os.path.isdir(trial_prefix):
                                r = parse_streme(xml,tgt_head,trial_prefix)
                                IG_results.append(r)
                        if f == 'seq2seq':
                            xml = f'new_attr/EDC_3_{trial}/EDC_3_{base}_pos_test_summed_attr/streme_out/streme.xml'
                            trial_prefix = f'new_attr/EDC_3_{trial}'
                            if os.path.isdir(trial_prefix):
                                r = parse_streme(xml,tgt_head,trial_prefix)
                                IG_results.append(r)
                    
                    #fig,axs = plt.subplots(4,1,sharex=True,sharey=True)
                    all_motif_names = []
                    
                    # EDA attributions
                    attn_results = []
                    for l in range(4):
                        head_results = [] 
                        for h in range(8):
                            tgt_head = "layer{}head{}".format(l,h)
                            if f == 'EDC': 
                                xml = f'new_attr/EDC_3_{trial}/EDC_3_test_layer{l}_{tgt_head}/streme_out/streme.xml'
                                txt = f'new_attr/EDC_3_{trial}/EDC_3_test_layer{l}_{tgt_head}/streme_out/streme.txt'
                                all_motif_names.append(txt)
                                trial_prefix = f'new_attr/EDC_3_{trial}'
                                if os.path.isdir(trial_prefix):
                                    r = parse_streme(xml,tgt_head,trial_prefix)
                                    attn_results.extend(r)
                                    head_results.extend(r)
                            elif f == 'seq2seq': 
                                xml = f'new_attr/seq2seq_3_{trial}/seq2seq_3_test_layer{l}_{tgt_head}/streme_out/streme.xml'
                                txt = f'new_attr/seq2seq_3_{trial}/seq2seq_3_test_layer{l}_{tgt_head}/streme_out/streme.txt'
                                all_motif_names.append(txt)
                                trial_prefix = f'new_attr/seq2seq_3_{trial}'
                                if os.path.isdir(trial_prefix):
                                    r = parse_streme(xml,tgt_head,trial_prefix)
                                    attn_results.extend(r)
                                    head_results.extend(r)
                        #ax = axs[l]    
                        #motifs_per_layer_plot(head_results,l,ax) 
                     
                    name = f'{f}_{trial}_layer_motifs.svg'
                    #plt.tight_layout()
                    #plt.savefig(name)
                    #plt.close()
                    aggregate_motifs(all_motif_names,trial)
                    motifs_per_head_plot(attn_results,trial,cmap,norm) 


