import orjson
import sys
import gzip
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats , signal
import re,random
from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition, mark_inset

def test_normality(data):

    n_samples,n_positions = data.shape
    alpha = 1e-5
    normality = []
    count = 0
    
    for i in range(n_positions):
        stat, p = stats.normaltest(data[:,i],nan_policy='omit')
        if p > alpha:
            count +=1
    print(count)

def summarize_head(cds_storage,saved_file,tgt_head,align_on="start"):

    samples = []
    sample_ids = []
    before_lengths = []
    after_lengths = []

    with open(saved_file) as inFile:
        for l in inFile:
            fields = orjson.loads(l)
            id_field = "ID"
            id = fields[id_field]
            
            if id in cds_storage:# and (id.startswith('NR_') or id.startswith('XR_')):
                cds = cds_storage[id]
                if cds != "-1" :
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    splits = [clean(x) for x in splits]
                    start,end = tuple([int(x) for x in splits])

                    attn = [float(x)/1000 for x in fields[tgt_head]]
                    total = sum(attn) / len(attn)
                    #print("tscript: {}, total {}".format(id,total))

                    if align_on == "start":
                        before_lengths.append(start)
                        after_lengths.append(len(attn) - start)
                    elif align_on == "end":
                        before_lengths.append(end)
                        after_lengths.append(len(attn) - end)
                    else:
                        raise ValueError("align_on must be 'start' or 'end'")
                    
                    sample_ids.append(id)
                    samples.append(attn)

    percentiles = [10*x for x in range(11)]
    after_percentiles = np.percentile(after_lengths,percentiles)
    before_percentiles = np.percentile(before_lengths,percentiles)

    max_before = max(before_lengths)
    max_after = max(after_lengths)
    domain = np.arange(-max_before,999).reshape(1,-1)

    if align_on == "start":
        samples = [align_on_start(attn,start,max_before) for attn,start in zip(samples,before_lengths)]
    else:
        samples = [align_on_end(attn,end,max_after) for attn,end in zip(samples,before_lengths)]

    samples = np.asarray(samples)
    support = np.count_nonzero(~np.isnan(samples),axis=0)
    sufficient = support >= 0.70*samples.shape[0]
    samples = samples[:,sufficient]
    domain = domain[:,sufficient]
    consensus = np.nanmean(samples,axis=0)
    return consensus,domain.ravel()

def build_consensus(parent):

    combined_file = "../Fa/refseq_combined_cds.csv.gz"
    cds_storage = load_CDS(combined_file,include_lnc=False)
    consensus = []

    for l in range(4):
        layer = parent+"_layer{}.enc_dec_attns".format(l)
        for h in range(8):
            tgt_head = "layer{}head{}".format(l,h)
            summary,domain  = summarize_head(cds_storage,layer,tgt_head,align_on ="start") 
            consensus.append(summary.reshape(-1,1))

    eps = 1e-64
    consensus = np.concatenate(consensus,axis=1)
        
    savefile = parent+"_consensus.npz"
    np.savez(savefile,consensus=consensus,domain=domain) 

def build_consensus_IG(ig_file,tgt):

    combined_file = "../Fa/refseq_combined_cds.csv.gz"
    cds_storage = load_CDS(combined_file,include_lnc=False)
    summary,domain  = summarize_head(cds_storage,ig_file,tgt,align_on ="start") 

    eps = 1e-64
        
    savefile = ig_file+"_consensus.npz"
    np.savez(savefile,consensus=summary,domain=domain) 

def plot_examples(consensus):   

    random.seed(30)
    example_indexes = random.sample(range(len(samples)),4)
    example_ids = [sample_ids[x] for x in example_indexes]
    examples = samples[example_indexes,:]
    filename = attn_prefix +"profile.pdf"

    with PdfPages(filename) as pdf:
        for i in range(len(example_indexes)):
            ex = examples[i,:]
            non_nan = ~np.isnan(ex)
            #ex_domain = domain[non_nan]
            #ex = ex[non_nan]
            
            plt.plot(domain,ex,'k',color='#CC4F1B')
            plt.plot(domain,consensus,'k',linestyle='dashed')
            ax = plt.gca()

            if plt_std_error:
                plt.fill_between(domain, consensus[non_nan]-2*error[non_nan], consensus[non_nan]+2*error[non_nan],alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            
            ax.set(xlabel = "Pos. Relative to CDS", ylabel = "Attention Score")
            ax.set_title(sample_ids[i])
  
            if plt_inset:
                axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
                axins.stem(domain,consensus,use_line_collection=True,basefmt=" ",markerfacecolor='w')
                axins.set_xlim(-10,13)
                ax.indicate_inset_zoom(axins)
            
            plt.tight_layout(rect=[0,0.03,1,0.95])
            pdf.savefig()
            plt.close()

def plot_line(domain,consensus,name,plot_type,plt_std_error=False):

    if plot_type == "stem":
        markerline, stemlines, baseline  = plt.stem(domain,consensus,use_line_collection=True,basefmt=" ")
    elif plot_type == "line":
        plt.plot(domain,consensus,'k',color='#CC4F1B')
        if plt_std_error:
            plt.fill_between(domain,consensus-2*error,consensus+2*error,alpha=0.5,edgecolor='#CC4F1B',facecolor='#FF9848')
    plt.axhline(y = 0, color = 'gray', linestyle = ':')     
    #plt.xlim(-10,13)
    plt.xlabel("Pos. Relative to CDS")
    plt.ylabel("Attention Score")
    plt.title(name)
    plt.tight_layout(rect=[0,0.03, 1, 0.95])
    plt.savefig(name+"codingprofile.pdf")
    plt.close()

def getLongestORF(mRNA):
    ORF_start = -1
    ORF_end = -1
    longestORF = 0
    for startMatch in re.finditer('ATG',mRNA):
        remaining = mRNA[startMatch.start():]
        if re.search('TGA|TAG|TAA',remaining):
            for stopMatch in re.finditer('TGA|TAG|TAA',remaining):
                ORF = remaining[0:stopMatch.end()]
                if len(ORF) % 3 == 0:
                    if len(ORF) > longestORF:
                        ORF_start = startMatch.start()
                        ORF_end = stopMatch.end()
                        longestORF = len(ORF)
                    break
    return ORF_start,ORF_end

def mean_by_mod(attn,savefile):

    idx = np.arange(attn.shape[0])
    zero = idx % 3 == 0
    one = idx % 3 == 1
    two = idx % 3 == 2

    storage = []
    for frame,mask in enumerate([zero,one,two]):
       slice = attn[mask]
       slice = slice[~np.isnan(slice)]
       print("frame {} , sum {}".format(frame,slice.sum()))
       for val in slice.tolist():
           entry = {"frame" : frame,"val" : val}
           storage.append(entry)

    df = pd.DataFrame(storage)
    result = stats.f_oneway(df['val'][df['frame'] == 0],df['val'][df['frame'] == 1],df['val'][df['frame'] == 2])
    means = [np.nanmean(attn[mask]) for mask in [zero,one,two]]
    
    textstr = '\n'.join((
    r'$F-statistic=%.3f$' % (result.statistic, ),
    r'$p-val=%.3f$' % (result.pvalue, )))

    print(textstr)

    '''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax = sns.barplot(x="frame",y="val",data=df)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    plt.xlabel("Pos. rel. to start mod 3")
    plt.ylabel("Attention")
    plt.title("Attention by Frame")
    prefix = savefile.split(".")[0]
    outfile = prefix+"attn_frames.pdf"
    plt.savefig(outfile)
    plt.close()
    '''

def load_CDS(combined_file,include_lnc=False):

    print("parsing",combined_file)

    df = pd.read_csv(combined_file,sep="\t")
    
    df['RNA_LEN'] = [len(x) for x in df['RNA'].values.tolist()]
    df = df[df['RNA_LEN'] < 1000]
    
    ids_list = df['ID'].values.tolist()
    cds_list = df['CDS'].values.tolist()
    rna_list = df['RNA'].values.tolist()

    if include_lnc:
        temp = []
        # name largest ORF as CDS for lncRNA
        for i in range(len(cds_list)):
            curr = cds_list[i]
            if curr == "-1":
                start,end = getLongestORF(rna_list[i])
                fake_cds = "{}:{}".format(start,end)
                temp.append(fake_cds)
            else:
                temp.append(curr)
        cds_list = temp

    return dict((x,y) for x,y in zip(ids_list,cds_list))

def align_on_start(attn,cds_start,max_start,):

    max_len = 999
    
    indices = list(range(len(attn)))
    indices = [x-cds_start for x in indices]

    left_remainder = max_start - cds_start
    prefix = [np.nan for x in range(left_remainder)]
    right_remainder = max_len - indices[-1] -1
    suffix = [np.nan for x in range(right_remainder)]
   
    eps = 1e-64
    min_information = 1/len(attn)
    
    attn = [x/min_information for x in attn]
    total = prefix+attn+suffix
    return total

def align_on_end(attn,cds_end,max_end):

    max_len = 999

    indices = list(range(len(attn)))
    indices = [x-cds_end for x in indices]

    left_remainder = max_len-cds_end
    prefix = [np.nan for x in range(left_remainder)]
    right_remainder = max_end - indices[-1]-1
    suffix = [np.nan for x in range(right_remainder)]

    total = prefix+attn+suffix
    return total

def plot_heatmap(consensus,title,heatmap_file):

    palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    sns.heatmap(consensus,cmap=palette)
    plt.suptitle(title)
    plt.savefig(heatmap_file)
    plt.close()

def plot_power_spectrum(consensus,title,spectrum_file):

    palette = sns.color_palette()

    freq,ps = signal.welch(consensus,axis=0,scaling="spectrum",)
    fig, ax1 = plt.subplots()
    n_freq_bins, n_heads = ps.shape
   
    x_label = "Period (Nuc.)" if domain == "period" else "Frequency (Cycles/Nuc.)"
    x_vals = 1.0 / freq if domain =="period" else freq    
    
    for i in range(n_heads):
        layer = i // 8
        label = layer if i % 8 == 0 else None
        ax1.plot(x_vals,ps[:,i],color=palette[layer],label=label,alpha=0.6)

    tick_labels = ["0",r'$\frac{1}{10}$']+[r"$\frac{1}{"+str(x)+r"}$" for x in range(5,1,-1)]
    tick_locs =[0,1.0/10]+ [1.0 / x for x in range(5,1,-1)]
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labels)
    #ax1.legend(title="Layer")
    
    ax1.set_xlabel(x_label)
    #ax1.set_ylabel("Power")

    plt.savefig(spectrum_file)
    plt.close()

def scale_min_max(consensus):

    mins = consensus.min(axis=0)
    maxes = consensus.max(axis=0)
    return  (consensus - mins) / (maxes - mins)

if __name__ == "__main__":
    
    plt.rcParams['font.family'] = "sans-serif"
    
    seq2seq_avg = "seq2seq_3_avg_pos_redo.ig"
    seq2seq_zero = "seq2seq_3_zero_pos_redo.ig"
    seq2seq_A = "seq2seq_3_A_pos.ig"
    seq2seq_C = "seq2seq_3_C_pos.ig"
    seq2seq_G = "seq2seq_3_G_pos.ig"
    seq2seq_T = "seq2seq_3_T_pos.ig"

    ED_classify_avg = "best_ED_classify_avg_pos.ig"
    ED_classify_zero = "best_ED_classify_zero_pos.ig"
    ED_classify_A = "best_ED_classify_A_pos.ig"
    ED_classify_C = "best_ED_classify_C_pos.ig"
    ED_classify_G = "best_ED_classify_G_pos.ig"
    ED_classify_T = "best_ED_classify_T_pos.ig"

    for base in ['avg','zero','A','C','G','T']:
        
        f = "best_ED_classify_"+base
        print(f)
        build_consensus_IG(f+'_pos.ig','summed_attr')
        loaded = np.load(f+'_pos.ig','_consensus.npz')
        consensus = loaded['consensus'].reshape(-1,1)
        domain = loaded['domain'].tolist()
        plot_line(domain,consensus,f,plot_type="line")    
        plot_power_spectrum(consensus,'Encoder Decoder Classify Spectrum Density',f+'IG_spectrum.pdf')
        
        f = "seq2seq_3_"+base
        print(f)
        build_consensus_IG(f+'_pos.ig','summed_attr')
        loaded = np.load(f+'_pos.ig','_consensus.npz')
        consensus = loaded['consensus'].reshape(-1,1)
        domain = loaded['domain'].tolist()
        plot_line(domain,consensus,f,plot_type="line")    
        plot_power_spectrum(consensus,"Seq2seq Spectrum Density",f+"IG_spectrum.pdf")
    
    '''
    loaded = np.load(ED_classify+"_consensus.npz")
    consensus = loaded['consensus']
    domain = loaded['domain'].tolist()
    #consensus = scale_min_max(consensus)
    
    #plot_heatmap(np.transpose(consensus),"Encoder Decoder Classifier","ED_classify_attn_heatmap.pdf")
    plot_power_spectrum(consensus,"Encoder Decoder Power Spectrum","ED_classify_attn_spectrum.pdf")

    loaded = np.load(seq2seq+"_consensus.npz")
    consensus = loaded['consensus'] 
    #consensus = scale_min_max(consensus)

    #plot_heatmap(np.transpose(consensus),"Seq2seq","seq2seq_attn_heatmap.pdf")
    plot_power_spectrum(consensus,"Seq2seq Power Spectrum","seq2seq_attn_spectrum.pdf")
    '''
