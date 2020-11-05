import json
import sys
import gzip
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats , signal
import re,random

def load_enc_dec_attn(cds_storage,saved_file,align_on="start",plot_type="line",mode="attn",prefix=None,plt_std_error=False,plt_inset=False):

    samples = []
    sample_ids = []
    before_lengths = []
    after_lengths = []

    with open(saved_file) as inFile:
        for l in inFile:

            fields = json.loads(l)
            if mode == "attn":
                tgt = "layer_0_pos_0"
                id_field = "TSCRIPT_ID"
            else:
                tgt = "attr"
                id_field = "ID" 

            id = fields[id_field]
            sample_ids.append(id)
            
            if id in cds_storage:# and (id.startswith("XR_") or id.startswith("NR_")):
                cds = cds_storage[id]
                if cds != "-1" :
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    splits = [clean(x) for x in splits]
                    start,end = tuple([int(x) for x in splits])
                
                    if mode == "attn":
                        attn = keep_nonzero(fields[tgt])
                        attn = [float(x) for x in fields[tgt]]
                    else:
                        attn = [float(x) for x in fields[tgt]]

                    if align_on == "start":
                        before_lengths.append(start)
                        after_lengths.append(len(attn) - start)
                    elif align_on == "end":
                        before_lengths.append(end)
                        after_lengths.append(len(attn) - end)
                    else:
                        raise ValueError("align_on must be 'start' or 'end'")
                    samples.append(attn)

    percentiles = [10 * x for x in range(11)]
    after_percentiles = np.percentile(after_lengths,percentiles)
    before_percentiles = np.percentile(before_lengths,percentiles)

    if align_on == "start":
        max_before = max(before_lengths)
        domain = np.asarray(list(range(-max_before,999)))
        samples = [align_on_start(attn,start,max_before) for attn,start in zip(samples,before_lengths)]
    else:
        max_after = max(after_lengths)
        domain = np.asarray(list(range(-999,max_after)))
        samples = [align_on_end(attn,end,max_after) for attn,end in zip(samples,before_lengths)]

    attn_prefix = saved_file.split('.')[0]

    # mean and standard error over samples
    samples = np.asarray(samples)
    consensus = np.nanmean(samples,axis=0)
    error = np.nanstd(samples,axis=0) # / np.sqrt(samples.shape[0])
    value_by_mod(consensus[max_before:max_before+400],saved_file)

    if plot_type == "examples":
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

    elif plot_type == "stem" or plot_type =="line":
        if plot_type == "stem":
            markerline, stemlines, baseline  = plt.stem(domain,consensus,use_line_collection=True,basefmt=" ")
            #markerline.set_markerfacecolor('none')
        elif plot_type == "line":
            plt.plot(domain,consensus,'k',color='#CC4F1B')
            if plt_std_error:
                plt.fill_between(domain,consensus-2*error,consensus+2*error,alpha=0.5,edgecolor='#CC4F1B',facecolor='#FF9848')
       
        plt.xlabel("Pos. Relative to CDS")
        plt.ylabel("Attention Score")
        prefix = saved_file.split(".")[0]
        title = "{} Attention Profile".format(prefix)
        plt.title(title)
        plt.tight_layout(rect=[0,0.03, 1, 0.95])
        plt.savefig(prefix +"codingprofile.pdf")
        plt.close()

    elif plot_type == "spectrum":
        data = consensus[max_before:max_before+300]

        freq,ps = signal.welch(data)
        #periods = 1.0 / freq
        #idx = np.argsort(freq)
        plt.plot(freq,ps)
        
        plt.xlabel("Cycles/Nuc.")
        plt.ylabel("Power")
        title = "Layer {} Head {} Attention Power Spectrum".format(layer,head)
        plt.title(title)
        plt.tight_layout(rect=[0,0.03, 1, 0.95])
        plt.savefig(attn_prefix +"spectrum_periods.pdf")
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

def value_by_mod(attn,savefile):

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

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    #ax = sns.violinplot(x="frame",y="val",data=df)
    ax = sns.barplot(x="frame",y="val",data=df)
    #ax = sns.barplot(x=[0,1,2],y=means) 
    
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

def keep_nonzero(attn):
    nonzero = []
    for a in attn:
        f = float(a)
        if f == 0.0:
            break
        else:
            nonzero.append(f)
    return nonzero

def align_on_start(attn,cds_start,max_start):

    max_len = 999
    
    indices = list(range(len(attn)))
    indices = [x-cds_start for x in indices]

    left_remainder = max_start - cds_start
    prefix = [np.nan for x in range(left_remainder)]
    right_remainder = max_len - indices[-1] -1
    suffix = [np.nan for x in range(right_remainder)]
    
    min_information = -np.log2(1.0/len(attn))
    #attn = [min_information / -np.log2(x) for x in attn]

    total = prefix +attn+ suffix
    return total

def align_on_end(attn,cds_end,max_end):

    max_len = 999

    indices = list(range(len(attn)))
    indices = [x-cds_end for x in indices]

    left_remainder = max_len - cds_end
    prefix = [np.nan for x in range(left_remainder)]
    right_remainder = max_end - indices[-1] -1
    suffix = [np.nan for x in range(right_remainder)]

    total = prefix + attn + suffix
    return total

if __name__ == "__main__":
    
    combined_file = sys.argv[1]
    mode = sys.argv[2]

    cds_storage = load_CDS(combined_file,include_lnc=False)
    #load_enc_dec_attn(cds_storage,"classify.enc_dec_attns",align_on="start",mode=mode,plot_type="line") 
    
    for h in list(range(8)):
        layer = "medium/layer3head{}.enc_dec_attns".format(h)
        print(layer)
        load_enc_dec_attn(cds_storage,layer,align_on ="start",mode=mode,plot_type="examples")
