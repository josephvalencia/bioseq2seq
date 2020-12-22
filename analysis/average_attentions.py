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

    support = np.count_nonzero(~np.isnan(data),axis=0).tolist()
    alpha = 1e-5

    normality = []

    for i,count in enumerate(support):
        
        if count >= 0.75*n_samples:
            stat, p = stats.normaltest(data[:,i],nan_policy='omit')
            if p > alpha:
                #print('Sample looks Gaussian (fail to reject H0)')
                normality.append(1.0)
            else:
                #print('Sample does not look Gaussian (reject H0)')
                normality.append(0.0)

    return np.asarray(normality) 

def summarize_head(cds_storage,saved_file,tgt_head,align_on="start"):

    samples = []
    sample_ids = []
    before_lengths = []
    after_lengths = []

    with open(saved_file) as inFile:
        for l in inFile:

            fields = orjson.loads(l)
            #id_field = "TSCRIPT_ID"
            id_field = "ID"

            id = fields[id_field]
            sample_ids.append(id)
            
            if id in cds_storage: # and (id.startswith("XR_") or id.startswith("NR_")):
                cds = cds_storage[id]
                
                if cds != "-1" :
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    splits = [clean(x) for x in splits]
                    start,end = tuple([int(x) for x in splits])
                
                    attn = [float(x) for x in fields[tgt_head]]

                    if align_on == "start":
                        before_lengths.append(start)
                        after_lengths.append(len(attn) - start)
                    elif align_on == "end":
                        before_lengths.append(end)
                        after_lengths.append(len(attn) - end)
                    else:
                        raise ValueError("align_on must be 'start' or 'end'")
                    
                    samples.append(attn)

    percentiles = [10*x for x in range(11)]
    after_percentiles = np.percentile(after_lengths,percentiles)
    before_percentiles = np.percentile(before_lengths,percentiles)

    max_before = max(before_lengths)
    max_after = max(after_lengths)
    domain = np.asarray(list(range(-max_before,max_after)))

    if align_on == "start":
        samples = [align_on_start(attn,start,max_before) for attn,start in zip(samples,before_lengths)]
    else:
        samples = [align_on_end(attn,end,max_after) for attn,end in zip(samples,before_lengths)]

    samples = np.asarray(samples)
   
    support = np.count_nonzero(~np.isnan(samples),axis=0)
    sufficient = support >= 0.75*samples.shape[0]
     
    samples = samples[:,sufficient]
    domain = domain[sufficient]
    
    consensus = np.nanmean(samples,axis=0)
    return consensus,domain

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
                if head == 6 or head == 7:
                    axins = ax.inset_axes([0.1, 0.6, 0.35, 0.35])
                else:
                    axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])

                axins.stem(domain,consensus,use_line_collection=True,basefmt=" ",markerfacecolor='w')
                axins.set_xlim(-10,13)
                ax.indicate_inset_zoom(axins)

            plt.tight_layout(rect=[0,0.03,1,0.95])
            pdf.savefig()
            plt.close()


def plot_line(domain,consensus,name,plot_type):

    if plot_type == "stem":
        markerline, stemlines, baseline  = plt.stem(domain,consensus,use_line_collection=True,basefmt=" ")
    elif plot_type == "line":
        plt.plot(domain,consensus,'k',color='#CC4F1B')
        if plt_std_error:
            plt.fill_between(domain,consensus-2*error,consensus+2*error,alpha=0.5,edgecolor='#CC4F1B',facecolor='#FF9848')
   
    plt.xlabel("Pos. Relative to CDS")
    plt.ylabel("Attention Score")
    title = "{} Attention Profile".format(prefix)
    plt.title(title)
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
            if curr != "-1":
                start,end = getLongestORF(rna_list[i])
                fake_cds = "{}:{}".format(start,end)
                temp.append(curr)
            else:
                start,end = getLongestORF(rna_list[i])
                fake_cds = "{}:{}".format(start,end)
                temp.append(fake_cds)
        cds_list = temp

    return dict((x,y) for x,y in zip(ids_list,cds_list))

def align_on_start(attn,cds_start,max_start):

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

def plot_power_spectrum(consensus,title,spectrum_file,domain="freq"):

    palette = sns.color_palette()

    freq,ps = signal.welch(consensus,axis=0,scaling="spectrum")
   
    fig, ax1 = plt.subplots()

    n_freq_bins, n_heads = ps.shape
   
    x_label = "Period (Nuc.)" if domain == "period" else "Frequency (Cycles/Nuc.)"
    x_vals = 1.0 / freq if domain =="period" else freq    
    
    for i in range(n_heads):
        layer = i // 8
        label = layer if i % 8 == 0 else None
        ax1.plot(x_vals,ps[:,i],color=palette[layer],label=label,alpha=0.6)

    if domain == "freq":
        tick_labels = ["0",r'$\frac{1}{10}$']+[r"$\frac{1}{"+str(x)+r"}$" for x in range(5,1,-1)]
        tick_locs =[0,1.0/10]+ [1.0 / x for x in range(5,1,-1)]
        ax1.set_xticks(tick_locs)
        ax1.set_xticklabels(tick_labels)
        ax1.legend(title="Layer")
    
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Power")
    
    if domain == "period":

        ax2 = plt.axes([0,0,1,1])
        
        # place and mark inset 
        ip = InsetPosition(ax1, [0.20,0.35,0.5,0.5])
        ax2.set_axes_locator(ip)
        mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

        for i in range(n_heads):
            layer = i // 8
            label = layer if i % 8 ==0 else None
            ax2.plot(x_vals,ps[:,i],color=palette[layer],label=label,alpha=0.4)
       
        ax2.legend(title="Layer")
        ax2.set_xlim(2.9,3.1)

    plt.suptitle(title)
    plt.savefig(spectrum_file)
    plt.close()

def scale_min_max(consensus):

    mins = consensus.min(axis=0)
    maxes = consensus.max(axis=0)
    return  (consensus - mins) / (maxes - mins)

if __name__ == "__main__":
    
    ED_classify = "results/best_ED_classify/best_ED_classify"
    seq2seq = "results/best_seq2seq/best_seq2seq"

    ED_ig = "results/best_ED_classify/best_ED_classify.ig"
    seq2seq_ig = "results/best_seq2seq/best_seq2seq.ig"

    #build_consensus(ED_classify)
    #build_consensus(seq2seq)
    
    #build_consensus_IG(ED_ig,"summed_attr")
    #build_consensus_IG(seq2seq_ig, "summed_attr")

    plt.rcParams['font.family'] = "sans-serif"

    '''
    loaded = np.load(ED_ig+"_consensus.npz")
    consensus = loaded['consensus']
    mean_by_mod(consensus,"test")
    consensus = scale_min_max(consensus).reshape(-1,1)
    domain = loaded['domain'].tolist()
    
    #plot_heatmap(np.transpose(consensus),"Encoder Decoder Classifier","ED_classify_IG_heatmap.pdf")
    plot_power_spectrum(consensus,"Encoder Decoder Power Spectrum","ED_classify_IG_spectrum_period.pdf")

    loaded = np.load(seq2seq_ig+"_consensus.npz")
    consensus = loaded['consensus'] 
    mean_by_mod(consensus,"test")
    consensus = scale_min_max(consensus).reshape(-1,1)

    #plot_heatmap(np.transpose(consensus),"Seq2seq","seq2seq_IG_heatmap.pdf")
    plot_power_spectrum(consensus,"Seq2seq Power Spectrum","seq2seq_IG_spectrum_period.pdf")
    
    '''

    plt.rcParams['font.family'] = "sans-serif"

    loaded = np.load(ED_classify+"_consensus.npz")
    consensus = loaded['consensus']
    print("ED_classify") 
    for h in range(consensus.shape[1]):
        curr_slice = consensus[:,h]
        print(h)
        mean_by_mod(curr_slice,"test")

    print("______________________________")
    print("Seq2seq")
    consensus = scale_min_max(consensus)
    domain = loaded['domain'].tolist()
    
    #plot_heatmap(np.transpose(consensus),"Encoder Decoder Classifier","ED_classify_attn_heatmap.pdf")
    #plot_power_spectrum(consensus,"Encoder Decoder Power Spectrum","ED_classify_attn_spectrum_period.pdf",domain="period")

    loaded = np.load(seq2seq+"_consensus.npz")
    consensus = loaded['consensus'] 
    
    for h in range(consensus.shape[1]):
        curr_slice = consensus[:,h]
        print(h)
        mean_by_mod(curr_slice,"test")
    
    consensus = scale_min_max(consensus)

    #plot_heatmap(np.transpose(consensus),"Seq2seq","seq2seq_attn_heatmap.pdf")
    #plot_power_spectrum(consensus,"Seq2seq Power Spectrum","seq2seq_attn_spectrum_period.pdf",domain="period")

