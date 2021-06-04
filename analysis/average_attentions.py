import orjson
import sys
import gzip
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Rectangle
from collections import Counter
from scipy import stats , signal
import re,random
from scipy.stats import pearsonr, kendalltau
from bioseq2seq.bin.batcher import train_test_val_split
from Bio.Seq import Seq
from Bio import motifs
from sklearn import preprocessing

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

def summarize_head(cds_storage,saved_file,tgt_head,mode="IG",align_on="start",coding=True):

    samples = []
    sample_ids = []
    before_lengths = []
    after_lengths = []

    with open(saved_file) as inFile:
        for l in inFile:
            fields = orjson.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"
            id = fields[id_field]

            is_pc = lambda x : x.startswith('NM_') or x.startswith('XM_')

            if id in cds_storage:
                # case 1 : protein coding  case 2 : non coding 
                if (coding and is_pc(id)) or (not coding and not is_pc(id)): 
                    cds = cds_storage[id]
                    if cds != "-1" :
                        splits = cds.split(":")
                        clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                        splits = [clean(x) for x in splits]
                        start,end = tuple([int(x) for x in splits])
                        
                        attn = [float(x) /1000 for x in fields[tgt_head]]

                        # IG has padding, strip it out
                        if mode == "IG":
                            src = fields['src'].split('<pad>')[0]
                            attn = attn[:len(src)]

                        #print("MDIG {} , start codon {}, scores {}".format(saved_file,src[start:start+3], attn[start:start+3]))

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
    print(samples.shape)
    support = np.count_nonzero(~np.isnan(samples),axis=0)
    sufficient = support >= 0.70*samples.shape[0]
    samples = samples[:,sufficient]
    domain = domain[:,sufficient]
    print(domain)
    consensus = np.nanmean(samples,axis=0)
    return consensus,domain.ravel()

def build_consensus_EDA(name,attn_file_list,coding=True):

    #combined_file = "../Fa/refseq_combined_cds.csv.gz"
    combined_file = "../Fa/test.csv"
    include_lnc = not coding
    cds_storage = load_CDS(combined_file,include_lnc=include_lnc)
    consensus = []

    for l in range(4):
        layer = attn_file_list[l] 
        for h in range(8):
            tgt_head = "layer{}head{}".format(l,h)
            summary,domain  = summarize_head(cds_storage,layer,tgt_head,mode="attn",align_on ="start",coding=coding) 
            consensus.append(summary.reshape(-1,1))

    consensus = np.concatenate(consensus,axis=1)
    suffix = "_PC" if coding else "_NC"
    savefile = name+suffix+"_EDA_consensus.npz"
    np.savez(savefile,consensus=consensus,domain=domain) 

def build_consensus_IG(ig_file,tgt,coding=True):

    combined_file = "../Fa/refseq_combined_cds.csv.gz"
    include_lnc = not coding
    cds_storage = load_CDS(combined_file,include_lnc=include_lnc)
    summary,domain  = summarize_head(cds_storage,ig_file,tgt,align_on ="start",coding=coding) 
    suffix = "_PC" if coding else "_NC"
    savefile = ig_file+suffix+"_consensus.npz"
    np.savez(savefile,consensus=summary,domain=domain) 

def build_consensus_multi_IG(name,ig_file_list,tgt,coding=True):

    #combined_file = "../Fa/refseq_combined_cds.csv.gz"
    combined_file = "../Fa/test.csv"
    include_lnc = not coding
    cds_storage = load_CDS(combined_file,include_lnc=include_lnc)
    consensus = []

    for layer in ig_file_list:
        summary,domain  = summarize_head(cds_storage,layer,tgt,align_on ="start",coding=coding) 
        consensus.append(summary.reshape(-1,1))

    consensus = np.concatenate(consensus,axis=1)
    suffix = "_PC" if coding else "_NC"
    savefile =name+suffix+"_multi_consensus.npz"
    np.savez(savefile,consensus=consensus,domain=domain) 

def build_example_multi_IG(name,ig_file_list,tgt,id_list):

    for tscript in id_list:
        total = []
        for layer in ig_file_list:
            summary  = example_attributions(layer,tgt,tscript) 
            total.append(summary.reshape(-1,1))
        
        total = np.concatenate(total,axis=1)
        savefile = 'examples/'+tscript+'_'+name+"_multi.npz"
        np.savez(savefile,total=total) 

def example_attributions(saved_file,tgt,transcript):

    with open(saved_file) as inFile:
        query = "{\"ID\": \""+transcript
        for l in inFile:
            if l.startswith(query):
                fields = orjson.loads(l)
                id_field = "ID"
                id = fields[id_field]
                return np.asarray(fields[tgt]) / 1000
    return None

def plot_line(domain,consensus,name,plot_type,plt_std_error=False,labels=None):

    palette = sns.color_palette()
    n_positions,n_heads = consensus.shape

    for i in range(n_heads): 
        if plot_type == "stem":
            markerline, stemlines, baseline  = plt.stem(domain,consensus[:,i],color=palette[i],label=labels,alpha=0.6,use_line_collection=True,basefmt=" ")
        elif plot_type == "line":
            plt.plot(domain,consensus[:,i],color=palette[i],label=labels[i],alpha=0.6)
            if plt_std_error:
                plt.fill_between(domain,consensus-2*error,consensus+2*error,alpha=0.5,edgecolor='#CC4F1B',facecolor='#FF9848')

    plt.axhline(y=0, color='gray', linestyle=':')     
    plt.xlabel("Pos. Relative to CDS")
    plt.ylabel("Mean IG Score")
    #plt.title(name)
    plt.legend()
    plt.tight_layout(rect=[0,0.03, 1, 0.95])
    plt.savefig(name+"codingprofile.svg")
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
                        ORF_end = startMatch.start()+stopMatch.end()
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
    outfile = prefix+"attn_frames.svg"
    plt.savefig(outfile)
    plt.close()
    '''

def load_CDS(combined_file,include_lnc=False):

    df = pd.read_csv(combined_file,sep="\t")
    df = df.set_index('ID')

    subset = []
    sub_file = "output/test/redundancy/test_reduced_80_ids.txt" 
    with open(sub_file) as inFile:
        for l in inFile:
            subset.append(l.rstrip())
    df = df.loc[subset]
    df = df.reset_index()

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
                rna = rna_list[i]
                start,end = getLongestORF(rna)
                cds = "{}:{}".format(start,end)
                temp.append(cds)
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
    
    #min_information = 1/len(attn)
    min_information = -np.log2(1.0/len(attn))
    #attn = [min_information / -np.log2(x) for x in attn]
    #attn = [x/min_information for x in attn]
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

def plot_heatmap(consensus,cds_start,title,heatmap_file,hist_file):

    plt.figure(figsize=(24, 6))
    palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

    b = 12 if cds_start > 12 else cds_start 

    consensus = consensus[:4,cds_start-b:cds_start+60]
    
    min_val = np.min(consensus)
    max_val = np.max(consensus) 
    print(min_val,max_val)

    domain = list(range(-b,60)) 
    #consensus = consensus[:4,:] 
    consensus_df = pd.DataFrame(data=consensus,index=['A','C','G','T'],columns=domain)
    
    #df_melted = consensus_df.T.melt(var_name='MDIG baseline')
    #sns.displot(df_melted,x='value',hue='MDIG baseline',common_bins=True,bins=np.arange(-0.05,0.025,0.001))
    #plt.savefig(hist_file)
    #plt.close()
    #quit()
    #ax = sns.heatmap(consensus_df,cmap='bwr',vmin=-.15,vmax=0.1,center=0,square=True,robust=True,xticklabels=3)
    
    ax = sns.heatmap(consensus_df,cmap='bwr',center=0,square=True,vmin=-0.08,vmax=0.04,robust=True,xticklabels=3)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 18)
    ax.tick_params(axis='x',labelsize=28)
    ax.axhline(y=0, color='k',linewidth=2.5)
    ax.axhline(y=consensus.shape[0], color='k',linewidth=2.5)
    ax.axvline(x=0, color='k',linewidth=2.5)
    ax.axvline(x=consensus.shape[1], color='k',linewidth=2.5)
    ax.add_patch(Rectangle((b,0),3, 4, fill=False, edgecolor='yellow', lw=2.5))
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=24)

    plt.savefig(heatmap_file)
    plt.close()

def plot_power_spectrum(consensus,title,spectrum_file,mode,units='freq'):

    palette = sns.color_palette()
    freq,ps = signal.welch(consensus,axis=0,scaling='density',average='median')
    fig, ax1 = plt.subplots()
    n_freq_bins, n_heads = ps.shape
   
    x_label = "Period (nt.)" if units == "period" else "Frequency (cycles/nt.)"
    x_vals = 1.0 / freq if units =="period" else freq    

    if mode == 'attn':
        for i in range(n_heads):
            layer = i // 8
            label = layer if i % 8 == 0 else None
            ax1.plot(x_vals,ps[:,i],color=palette[layer],label=label,alpha=0.6)
            #ax1.plot(x_vals,ps[:,i],label=i,alpha=0.6)
    else:
        #labels = ['A','C','G','T']
        #labels = ['mean','zero']
        #labels = ['A','C','G','mean','zero']
        #for i in range(len(labels)):
        #    ax1.plot(x_vals,ps[:,i],color=palette[i],label=labels[i],alpha=0.6)
    
        ax1.plot(x_vals,ps[:,4],color=palette[0],label='mean',alpha=0.6)
        ax1.plot(x_vals,ps[:,5],color=palette[1],label='zero',alpha=0.6)
   
    tick_labels = ["0",r'$\frac{1}{10}$']+[r"$\frac{1}{"+str(x)+r"}$" for x in range(5,1,-1)]
    tick_locs =[0,1.0/10]+ [1.0 / x for x in range(5,1,-1)]
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labels,fontsize=12)

    if mode == 'attn':
        ax1.legend(title=title+" attention layer")
        ax1.set_ylabel("Attention Power Spectrum")
    else:
        ax1.legend(title=title+' IG baseline')
        ax1.set_ylabel("IG Power Spectrum")
    
    ax1.set_xlabel(x_label)
    ax1.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    plt.savefig(spectrum_file)
    plt.close()

def optimize(consensus,name,rna,protein,cds_start,cds_end):

    print(name)
    tscript = name.split('/')[-1]
    coding = tscript.startswith('NM') or tscript.startswith('XM')
    print(coding)
    print(len(rna[cds_start:cds_end]),protein)
    nucs = consensus[:,:4]
    df = pd.DataFrame(data=nucs,columns=['A','C','G','T'])
    arg_maxes = np.argmax(nucs,axis=1)
    maxes = np.max(nucs,axis=1)
    outliers = (maxes >= 2).tolist()
    
    optimized = []
    vocab = {0:'A',1:'C',2:'G',3:'T'}
    n_mutations = 0
    n_nonsynonymous = 0

    largest = np.argsort(-maxes)
    #arg_maxes = arg_maxes[largest]
    #maxes = maxes[largest]
    print(cds_start,cds_end)
    
    n_steps = 5
    step = 0
    optimized = [c for c in rna]

    for i in largest:
        if i >=cds_start and i < cds_end:
            frame = (cds_start -i) % 3
            v = vocab[arg_maxes[i]]
            print("{}->{}, idx {}  score {} frame {}".format(rna[i],v,i,maxes[i],frame))
            optimized[i] = v
            step+=1
            if step == n_steps:
                break
    
    '''
    # make mutations based on high PM-DIG scores  
    for i in range(len(rna)):
        is_outlier = outliers[i] == 1
        if is_outlier:
            optimized.append(vocab[arg_maxes[i]])
            frame = (cds_start-i) % 3
            print("Changing pos {} (frame {}) from {} to {}".format(i,frame,rna[i],vocab[arg_maxes[i]]))
            if i >= cds_start and i < cds_end-3:
                n_mutations +=1
        else:
            optimized.append(rna[i])
    '''
   # correct nonsynonymous mutations
    for i,l in enumerate(range(cds_start,cds_end-3,3)):
        codon = ''.join(optimized[l:l+3])
        true = rna[l:l+3]
        aa = protein[i]
        candidate = Seq(codon).translate()
        if candidate != aa:
            #print("{}({})-> {}({}), idx {}".format(true,aa,codon,candidate,i)) 
            n_nonsynonymous +=1
            optimized[l:l+3] = [c for c in true]
    
    print("# mutations in CDS = {} , # synonymous = {}, len(protein) = {}".format(step,step-n_nonsynonymous,len(protein)))
    optimized = ''.join(optimized)
    cds = Seq(optimized[cds_start:cds_end])
    translation = cds.translate(to_stop=True)
    #print("Optimized RNA {} \n True RNA {}".format(optimized,rna))

def optimize2():

    n_iterations = 20
    for _ in range(n_iterations):

        mutations  = []
        for j in range(L):
            for b in 'ACGU':
                newTranscript = transcript[:j] + b + transcript[j+1:]
                score = score(newTranscript)
                # codon index
                i = (j-cdsStart) // 3
                #frame
                f = (j-cdsStart) % 3
                codon = transcript[i:i+3]
                newCodon = codon[:f] + b + codon[f+1:]
                
                if map[codon] == map[newCodon]:
                    mutations.append((score,j,b,codon,newCodon))

    return None

def scale_min_max(consensus):
    mins = consensus.min(axis=0)
    maxes = consensus.max(axis=0)
    return  (consensus - mins) / (maxes - mins)

if __name__ == "__main__":
    '''
    # ingest raw data
    data_file = "../Fa/refseq_combined_cds.csv.gz"
    dataframe = pd.read_csv(data_file,sep="\t",compression = "gzip")
    cds_storage = load_CDS(data_file,include_lnc=True)
     
    df_train,df_test,df_val = train_test_val_split(dataframe,1000,65)
    df_val = df_val.set_index('ID')
    df_train = df_train.set_index('ID')
    '''
    ''' 
    test_file = "../Fa/test.csv"
    train_file = "../Fa/train.csv"
    val_file = "../Fa/val.csv"

    cds_storage = load_CDS(test_file,include_lnc=True)
    df_test = pd.read_csv(test_file,sep='\t')
    df_test = df_test.set_index('ID')
    
    subset = []
    sub_file = "output/test/redundancy/test_reduced_80_ids.txt" 
    with open(sub_file) as inFile:
        for l in inFile:
            subset.append(l.rstrip())
    
    df_test = df_test.loc[subset]

    # IG data
    seq_bases = ['A','C','G','T','avg','zero']
    short_bases = ['avg','zero']
    
    seq_four_test_new = ['output/test/seq2seq/best_seq2seq_'+b+'_pos_test.ig' for b in seq_bases]
    seq_attn_file_list = ['output/test/seq2seq/best_seq2seq_test_layer{}.enc_dec_attns'.format(l) for l in range(4)]
    ED_file_list = ['output/test/ED_classify/best_ED_classify_'+b+'_pos_test.ig' for b in short_bases] 
    ED_attn_file_list = ['output/test/ED_classify/best_ED_classify_layer{}.enc_dec_attns'.format(l) for l in range(4)] 
    
    # select and save example transcripts
    np.random.seed(65)
    id_list = np.random.choice(df_test.index.values,size=35,replace=False)
    seqs = df_test.loc[id_list]['Protein'].values.tolist()
    rna = df_test.loc[id_list]['RNA'].values.tolist() 
    cds_list = [cds_storage[i] for i in id_list]
    starts = [x.split(':')[0] for x in cds_list]
    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
    starts = [int(clean(s)) for s in starts] 
    ends = [int(clean(x.split(':')[1])) for x in cds_list] 
    np.savez('example_ids.npz',ids=id_list,protein=seqs,rna=rna,starts=starts,ends=ends) 
    
    # build multi_IG examples
    #build_example_multi_IG('best_seq2seq_test',seq_four_test_new,'summed_attr',id_list)
    #build_example_multi_IG('best_ED_classify_test',ED_file_list,'summed_attr',id_list)

    # build EDA consensus
    build_consensus_EDA('best_seq2seq_test',seq_attn_file_list,coding=True)
    build_consensus_EDA('best_ED_classify_test',ED_attn_file_list,coding=True)
    
    # build multi_IG consensus
    #build_consensus_multi_IG('best_seq2seq_test_sum',seq_four_test_new,'summed_attr',coding=False)
    #build_consensus_multi_IG('best_ED_classify_test_sum',ED_file_list,'summed_attr',coding=False)
    #build_consensus_multi_IG('best_seq2seq_test_sum',seq_four_test_new,'summed_attr',coding=True)
    #build_consensus_multi_IG('best_ED_classify_test_sum',ED_file_list,'summed_attr',coding=True)
    '''
    # load example transcripts
    examples = np.load('example_ids.npz',allow_pickle=True)
    id_list = examples['ids'].tolist()
    starts = examples['starts'].tolist()
    ends = examples['ends'].tolist()
    proteins = examples['protein'].tolist()
    rna = examples['rna'].tolist()
    ED_transcripts = ['examples/'+t+'_best_ED_classify_test_multi.npz' for t in id_list]
    seq_transcripts = ['examples/'+t+'_best_seq2seq_test_multi.npz' for t in id_list]
    scaler = preprocessing.StandardScaler()
    
    '''
    # seq2seq transcript examples
    for f,s,e,r,p in zip(seq_transcripts,starts,ends,rna,proteins):
        name = f.split('.npz')[0]
        loaded = np.load(f)
        #consensus = scaler.fit_transform(-loaded['total'])
        consensus = -loaded['total']
        plot_heatmap(np.transpose(consensus),s,"",name+"_heatmap.svg")
        optimize(consensus,name,r,p,s,e)
   
    # ED_classify transcript examples
    for f,s,e,r,p in zip(ED_transcripts,starts,ends,rna,proteins):
        name = 'examples/'+f.split('.npz')[0]
        loaded = np.load(f)
        consensus = -loaded['total']
        plot_heatmap(np.transpose(consensus),s,"",name+"_heatmap.svg")
        plot_power_spectrum(consensus,"",name+"_spectrum.svg")
    '''
    consensus = np.load('best_seq2seq_test_PC_EDA_consensus.npz')['consensus']
    plot_power_spectrum(consensus,"bioseq2seq","best_seq2seq_test_PC_EDA_spectrum.svg",mode='attn')
    consensus = np.load('best_seq2seq_test_NC_EDA_consensus.npz')['consensus']
    plot_power_spectrum(consensus,"bioseq2seq","best_seq2seq_test_NC_EDA_spectrum.svg",mode='attn')
    
    #consensus = np.load('best_seq2seq_test_sum_PC_multi_consensus.npz')['consensus']
    #plot_power_spectrum(consensus,"bioseq2seq","best_seq2seq_test_sum_PC_IG_spectrum.svg",mode='IG')
    #plot_heatmap(-np.transpose(consensus),18,"bioseq2seq_test","best_seq2seq_test_PC_MDIG_heatmap.svg","best_seq2seq_test_PC_MDIG_hist.svg")
    
    #consensus = np.load('best_seq2seq_test_sum_NC_multi_consensus.npz')['consensus']
    #plot_power_spectrum(consensus,"bioseq2seq","best_seq2seq_test_sum_NC_IG_spectrum.svg",mode='IG')
    #plot_heatmap(-np.transpose(consensus),106,"bioseq2seq_test","best_seq2seq_test_NC_MDIG_heatmap.svg","best_seq2seq_test_NC_MDIG_hist.svg")
    
    #consensus = np.load('best_seq2seq_test_PC_multi_consensus.npz')['consensus']
    #plot_power_spectrum(consensus,"bioseq2seq","best_seq2seq_test_PC_IG_zero_spectrum.svg",mode='IG')

    consensus = np.load("best_ED_classify_test_PC_EDA_consensus.npz")['consensus']
    plot_power_spectrum(consensus,"EDC","best_EDC_test_PC_EDA_spectrum.svg",mode='attn')
   
    consensus = np.load("best_ED_classify_test_NC_EDA_consensus.npz")['consensus']
    plot_power_spectrum(consensus,"EDC","best_EDC_test_NC_EDA_spectrum.svg",mode='attn')

    #consensus = np.load("best_ED_classify_test_sum_NC_multi_consensus.npz")['consensus']
    #plot_power_spectrum(consensus,"EDC","best_EDC_test_sum_NC_IG_spectrum.svg",mode='IG')
    #plot_heatmap(-np.transpose(consensus),12,"EDC_test","best_EDC_test_PC_MDIG_heatmap.svg","best_EDC_test_PC_MDIG_hist.svg")

    #consensus = np.load("best_ED_classify_test_norm_NC_multi_consensus.npz")['consensus']
    #plot_power_spectrum(consensus,"EDC","best_EDC_test_norm_NC_IG_spectrum.svg",mode='IG')
    #plot_heatmap(-np.transpose(consensus),104,"EDC_test","best_EDC_test_NC_MDIG_heatmap.svg","best_EDC_test_NC_MDIG_hist.svg")
