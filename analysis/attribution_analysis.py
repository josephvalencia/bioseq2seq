import sys,random
import json
import os,re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr,kendalltau,ttest_ind
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import jensenshannon
from collections import defaultdict
from IPython.display import Image

from Bio.Data import CodonTable
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

from bioseq2seq.inputters.batcher import train_test_val_split

def plot_stem(name,array,labels):

    array = np.asarray(array)
    array = np.expand_dims(array,axis=1)
    array = array / np.linalg.norm(array,ord=2)

    ax = plt.stem(array,use_line_collection=True)
    plt.savefig(name+"_stemplot.png")
    plt.close()

def annotate_cds(start,end):

    width = 0.75
    color = "dimgray"

    plt.axvline(start,linestyle = "-.",linewidth = width,c=color)
    plt.axvline(end,linestyle = "-.",linewidth = width,c=color)

def plot_line(name,array,smoothed,cds_start=None,cds_end=None):
    
    array = array / np.linalg.norm(array)
    plt.plot(array,label="raw")
    #plt.plot(smoothed,label="smoothed")    
    plt.ylabel("Attention")
    plt.xlabel("Position")
    plt.legend()
    
    if not cds_start == None and not cds_end == None:
        annotate_cds(cds_start,cds_end)

    #plt.axhline(1/len(array),linestyle = "-.",linewidth = 0.75,c="black")
    plt.tight_layout()
    output = name+"_lineplot.pdf"
    plt.savefig(output)
    plt.close()

def get_top_k(array,k=15):

    k = k if k < len(array) else len(array)
    array = np.asarray(array)

    k_largest_inds = np.argpartition(array,-k)[-k:]
    k_largest_scores = array[k_largest_inds].tolist()
    k_largest_inds = k_largest_inds.tolist()

    return k_largest_scores,k_largest_inds

def get_min_k(array,k=15):

    k = k if k < len(array) else len(array)
    array = np.asarray(array)

    k_smallest_inds = np.argpartition(array,k)[:k]
    k_smallest_inds = k_smallest_inds.tolist()

    return k_smallest_inds

def top_indices(saved_file,tgt_field,coding_topk_file,noncoding_topk_file,mode= "attn"):
    
    df_storage = []
    coding_storage = []
    noncoding_storage = []

    out_name = saved_file.split(".")[0]
    
    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"

            id = fields[id_field]
            array = fields[tgt_field]
            array = np.asarray(array)

            name = id + "_" + mode
            
            window_size = 50
            
            #smoothed = uniform_filter1d(array,window_size,mode='constant',cval=0.0)
            #max_scores,max_idx = get_top_k(smoothed,1)
            
            max_idx = np.argmax(array).tolist()
            min_idx = np.argmax(-array).tolist()
            
            tscript_type = "<PC>" if (id.startswith("XM_") or id.startswith("NM_")) else "<NC>"
            storage = (id,max_idx)
            alt_storage = (id,min_idx)

            if tscript_type == "<PC>":
                coding_storage.append(storage)
            else:
                noncoding_storage.append(storage)

            #coding_storage.append(storage)
            #noncoding_storage.append(alt_storage)

    # save top indices
    with open(coding_topk_file,'w') as outFile:
        for tscript,idx in coding_storage:
            outFile.write("{},{}\n".format(tscript,idx))

    with open(noncoding_topk_file,'w') as outFile:
        for tscript,idx in noncoding_storage:
            outFile.write("{},{}\n".format(tscript,idx))

def smooth_array(array,window_size):

    half_size = window_size // 2
    running_sum = sum(array[:half_size]) 
    
    smoothed_scores = [0.0]*len(array)

    trail_idx = 0
    lead_idx = half_size

    for i in range(len(array)):
        gap_size = lead_idx - trail_idx + 1
        smoothed_scores[i] = running_sum / gap_size

        # advance lead until it reaches end
        if lead_idx < len(array)-1:
            running_sum += array[lead_idx]
            lead_idx +=1

        # advance trail when the gap is big enough or the lead has reached the end        
        if gap_size == window_size or lead_idx == len(array) -1:
            running_sum -= array[trail_idx]
            trail_idx+=1

    return smoothed_scores

def top_k_to_substrings(top_k_csv,motif_fasta,df):
    
    storage = []
    sequences = []
    
    # ingest top k indexes from attribution/attention
    with open(top_k_csv) as inFile:
        for l in inFile:
            fields = l.rstrip().split(",")
            id = fields[0]
            seq = df.loc[id,'RNA']

            substrings = []
            left_bound = 24
            right_bound = 26
            
            # get window around indexes
            for num,idx in enumerate(fields[1:]):
                idx = int(idx)

                # enforce uniform window
                if idx < left_bound:
                    idx = left_bound
                if idx > len(seq) - right_bound -1:
                    idx = len(seq) - right_bound -1 

                start = idx-left_bound if idx-left_bound >= 0 else 0
                end = idx+right_bound

                substr = seq[start:end]
                substrings.append(substr)
                
                if len(substr) > 7:
                    description = "loc[{}:{}]".format(start+1,end+1)
                    record = SeqRecord(Seq(substr),
                                            id=id+"_"+str(num),
                                            description=description)
                    sequences.append(record)

            entry = [id]+ substrings
            storage.append(entry)

    with open(motif_fasta,'w') as outFile:
        SeqIO.write(sequences, outFile, "fasta")

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

def make_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def codon_scores(saved_file,df,tgt_field,boxplot_file,scatterplot_file,significance_file,mode="attn"):

    extra = []
    
    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"
            id = fields[id_field]
            array = fields[tgt_field]
            seq = df.loc[id,'RNA']
            tscript_type = df.loc[id,'Type']

            if tscript_type == "<PC>":               
                # use provided CDS
                cds = df.loc[id,'CDS']
                if cds != "-1":
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    cds_start,cds_end = tuple([int(clean(x)) for x in splits])
                else:
                    cds_start,cds_end = getLongestORF(seq)
            else:
                # use start and end of longest ORF
                cds_start,cds_end = getLongestORF(seq)
            
            array = [float(x) / 1000 for x in array]
            inside = defaultdict(lambda : [0.0])
            outside = defaultdict(lambda : [0.0])

            disallowed = {'N','K','R','Y'}
            allowed = lambda codon : all([x not in disallowed for x in codon])
            inframe = lambda x : x >= cds_start and x<=cds_end-2 and (x-cds_start) % 3 == 0 

            total = sum(array)
            expectation = 3 * (total/len(array))
            uniform = 3 / len(array)

            # 5' UTR and out of frame and 3' UTR
            for i in range(0,len(array)-3):
                codon = seq[i:i+3]
                if allowed(codon) and not inframe(i):
                    score = sum(array[i:i+3])
                    outside[codon].append(score)
            
            # find average 
            for codon,scores in outside.items():
                avg = sum(scores) / len(scores) 
                info = {"tscript" : id ,"codon" : codon, "score" : avg , "status" : tscript_type, "segment" : "OUT"}
                extra.append(info)
            
            # inside CDS
            for i in range(cds_start,cds_end-3,3):
                codon = seq[i:i+3]
                if allowed(codon):
                    score = sum(array[i:i+3])
                    inside[codon].append(score)
           
           # average CDS
            for codon,scores in inside.items():
                avg = sum(scores) / len(scores)
                info = {"tscript" : id ,"codon" : codon, "score" : avg , "status" : tscript_type, "segment" : 'CDS'}
                extra.append(info)
            
        codon_df = pd.DataFrame(extra)
        num_coding = num_noncoding = num_out_coding = num_cds_coding = num_cds_noncoding = num_out_noncoding = 0
        cds_ratios = defaultdict(list)
        coding_status_ratios = defaultdict(list)

        with open(significance_file,'w') as outFile:
            for codon in codon_df.codon.unique():
                # compare coding and noncoding
                coding = codon_df[(codon_df.codon == codon) & (codon_df.status == "<PC>")]
                noncoding = codon_df[(codon_df.codon == codon) & (codon_df.status == "<NC>")]
                result = ttest_ind(coding.score,noncoding.score,equal_var=False)
                
                # count number of significant differences
                if result.pvalue < 0.01:
                    pc_score = coding.score.mean()
                    nc_score = noncoding.score.mean()
                    if pc_score > nc_score:
                        num_coding+=1
                    elif pc_score < nc_score:
                        num_noncoding+=1
                    diff = pc_score - nc_score
                    coding_status_ratios[codon].append(diff)
                
                # compare CDS vs UTR
                out = coding[coding.segment == "OUT"]
                cds = coding[coding.segment == "CDS"]
                result = ttest_ind(out.score,cds.score,equal_var=False)
                
                # count number of significant differences
                if result.pvalue < 0.01:
                    out_score = out.score.mean()
                    cds_score = cds.score.mean()
                    if cds_score > out_score:
                        num_cds_coding+=1
                    elif cds_score < out_score:
                        num_out_coding+=1
                    diff = cds_score - out_score 
                    cds_ratios[codon].append(diff) 
                
                # compare CDS vs UTR
                out = noncoding[noncoding.segment == "OUT"]
                cds = noncoding[noncoding.segment == "CDS"]
                result = ttest_ind(out.score,cds.score,equal_var=False)
                
                # count number of significant differences
                if result.pvalue < 0.01:
                    out_score = out.score.mean()
                    cds_score = cds.score.mean()
                    if cds_score > out_score:
                        num_cds_noncoding+=1
                    elif cds_score < out_score:
                        num_out_noncoding+=1

            outFile.write("Overall : {}/64 higher in coding , {}/64 higher in noncoding\n".format(num_coding,num_noncoding))
            outFile.write("Coding : {}/64 higher in CDS, {}/64 higher outside\n".format(num_cds_coding,num_out_coding))
            outFile.write("Noncoding: {}/64 higher in CDS, {}/64 higher outside\n".format(num_cds_noncoding,num_out_noncoding))
        
        sorted_ratios = {k: sum(v) / len(v) for k, v in sorted(coding_status_ratios.items(), key=lambda item: item[1],reverse=True)}
        plt.figure(figsize=(10,5))
        
        a = codon_df[codon_df['codon'].str.len() == 3]
        out = a[a['segment'] == 'OUT']
        cds = a[a['segment'] == 'CDS']
        nc = a[a['status'] == '<NC>']
        pc = a[a['status'] == '<PC>']

        out = out.groupby('codon').mean()
        cds = cds.groupby('codon').mean()
        pc = pc.groupby('codon').median()
        nc = nc.groupby('codon').median()
        
        '''
        by_status = pc.merge(nc,on='codon',suffixes=['_pc','_nc'])
        by_status['diff_coding'] = by_status['score_pc'] - by_status['score_nc'] 
        #by_status = by_status.sort_values('score_pc',ascending=False)
        #by_status.reset_index(inplace=True)
        by_segment = cds.merge(out,on='codon',suffixes=['_cds','_out'])
        
        combined = by_segment.merge(by_status,on='codon')
        combined['diff_cds'] = combined['score_cds'] - combined['score_out']
        combined['diff_coding'] = combined['score_pc'] - combined['score_nc']
        combined = combined.sort_values('diff_coding',ascending=False) 
        combined.reset_index(inplace=True)
        
        order = list(sorted_ratios.keys())
        '''
        pc = pc.sort_values('score',ascending=False)
        pc.reset_index(inplace=True)
        
        ax = sns.boxplot(x="codon",y="score",data=a[a['status'] == "<PC>"],order=pc['codon'].values.tolist(),linewidth=1,showfliers=False) 
        
        for item in ax.get_xticklabels():
            item.set_rotation(90)
        
        plt.title("Enrichment in Coding")
        plt.savefig(boxplot_file)
        plt.close()

        '''
        ax = sns.scatterplot(x='diff_cds',y='diff_coding',data=combined)
        for line in range(0,combined.shape[0]):
            ax.text(combined['diff_cds'][line]+0.001, combined['diff_coding'][line], combined['codon'][line], horizontalalignment='left',size='medium', color='black', weight='semibold')
        
        plt.savefig(scatterplot_file)
        plt.close()
        '''

def class_separation(saved_file):

    storage = []
    
    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "ID"
            tscript_id = fields[id_field]
            summed = np.asarray(fields['summed_attr']) / 1000
            status = "<PC>" if tscript_id.startswith("NM_") or tscript_id.startswith("XM_") else "<NC>"
            entry = {'transcript' : tscript_id , 'summed' : np.sum(summed), 'status' : status}
            print(entry)
            storage.append(entry)

    df = pd.DataFrame(storage)
    nc = df[df['status'] == '<NC>']
    pc = df[df['status'] == '<PC>']
    
    min_pc = np.round(pc['summed'].min(),decimals=3)
    max_pc = np.round(pc['summed'].max(),decimals=3)
    min_nc = np.round(nc['summed'].min(),decimals=3)
    max_nc = np.round(nc['summed'].max(),decimals=3)
    minimum = min(min_pc,min_nc)
    maximum = max(max_pc,max_nc)

    bins = np.arange(minimum,maximum,0.0005)
    pc_density,pc_bins = np.histogram(pc['summed'].values, bins=bins, density=True)
    pc_density = pc_density / pc_density.sum()
    nc_density,nc_bins = np.histogram(nc['summed'].values, bins=bins, density=True)
    nc_density = nc_density / nc_density.sum()
    jsd = jensenshannon(nc_density,pc_density)
    print('JS Divergence:',jsd)

    sns.histplot(df,x="summed",kde=False,hue="status",stat="density")
    #plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig(saved_file+'_class_hist.pdf')
    plt.close()

def IG_correlations(file_a,file_b):

    corrs = []
    storage = {}
    
    with open(file_a) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "ID"
            id = fields[id_field]
            summed = np.asarray(fields['summed_attr']) / 1000
            normed = np.asarray(fields['normed_attr']) / 1000
            storage[id] = summed

    with open(file_b) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "ID"
            id = fields[id_field]
            summed = np.asarray(fields['summed_attr']) / 1000
            normed = np.asarray(fields['normed_attr']) / 1000
            other = storage[id]
            v = kendalltau(summed,other)[0]
            corrs.append(v)

    corrs = np.asarray(corrs)
    print(np.nanmean(corrs))
    print(np.nanstd(corrs))

def run_attributions(saved_file,df_val,tgt_field,best_dir,mode="attn"):

    name = saved_file.split('.')[0]

    prefix = best_dir+"/"+name+'_'+tgt_field+"/"
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    # results files
    coding_indices_file = prefix+"coding_topk_idx.txt"
    noncoding_indices_file = prefix+"noncoding_topk_idx.txt"
    coding_motifs_file = prefix+"coding_motifs.fa"
    noncoding_motifs_file = prefix +"noncoding_motifs.fa"
    boxplot_file = prefix+"coding_boxplot.pdf"
    scatterplot_file = prefix+"coding_scatterplot.pdf"
    significance_file = prefix+"significance.txt"
    hist_file = prefix+"pos_hist.pdf"

    top_indices(saved_file,tgt_field,coding_indices_file,noncoding_indices_file,mode=mode)
    top_k_to_substrings(coding_indices_file,coding_motifs_file,df_val)
    top_k_to_substrings(noncoding_indices_file,noncoding_motifs_file,df_val)
    codon_scores(saved_file,df_val,tgt_field,boxplot_file,scatterplot_file,significance_file,mode)
    get_positional_bias(saved_file,df_val,tgt_field,hist_file,mode)

def get_positional_bias(saved_file,df,tgt_field,hist_file,mode):
    
    storage = []
    temp_idx = 0

    with open(saved_file) as inFile:
        for l in inFile:
            
            fields = json.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"
            id = fields[id_field]
            
            array = fields[tgt_field]
            seq = df.loc[id,'RNA']
            tscript_type = df.loc[id,'Type']
            argmax = np.argmax(array)

            if tscript_type == "<PC>":               
                # use provided CDS
                cds = df.loc[id,'CDS']
                if cds != "-1":
                    splits = cds.split(":")
                    clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
                    cds_start,cds_end = tuple([int(clean(x)) for x in splits])
                else:
                    cds_start,cds_end = getLongestORF(seq)

                entry = {"status" : "coding", "distance" : argmax-cds_start}
                storage.append(entry)
            else:
                # use start and end of longest ORF
                cds_start,cds_end = getLongestORF(seq)
                entry = {"status" : "noncoding", "distance" : argmax-cds_start}
                storage.append(entry)

    plt.figure()
    df = pd.DataFrame(storage)
    sns.histplot(df,x="distance",kde=False,hue="status",stat="density")
    
    plt.xlabel("Position relative to start/first AUG")
    plt.ylabel("Density")
    plt.title("Position of maximum attention")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig(hist_file)
    plt.close()

def visualize_attribution(data_file,attr_file):

    # ingest stored data
    dataframe = pd.read_csv(data_file,sep="\t",compression = "gzip")
    df_train,df_test,df_val = train_test_val_split(dataframe,1000,65)
    df_val = df_val.set_index("ID")

    print("Loading all attributions")
    with open(attr_file) as inFile:
        idx = 0
        for l in inFile:
            fields = json.loads(l)
            
            id = fields["ID"]
            
            if id == tgt_id:
                attr = np.asarray([float(x) / 1000 for x in fields["summed_attr"]])
                seq = df_val.loc[id,"RNA"]
               
               # storing couple samples in an array for visualization purposes
                vis = viz.VisualizationDataRecord(
                                        attr,
                                        0.90,
                                        25,
                                        25,
                                        25,
                                        attr.sum(),
                                        seq,
                                        100)
                
                display = viz.visualize_text([vis])
                html = display.data
                
            with open('html_file.html', 'w') as f:
                f.write(html)
                
if __name__ == "__main__":
    
    #plt.style.use('ggplot')
    sns.set_style("whitegrid")
   
    # ingest stored data
    data_file = "../Fa/refseq_combined_cds.csv.gz"
    dataframe = pd.read_csv(data_file,sep="\t",compression = "gzip")
    df_train,df_test,df_val = train_test_val_split(dataframe,1000,65)
    df_val = df_val.set_index("ID")
    
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

    for base in ['avg','A','C','G','T']:
        f = "best_ED_classify_"+base+"_pos.ig"
        run_attributions(f,df_val,"summed_attr","attributions/", "IG")
        f = "seq2seq_3_"+base+"_pos.ig"
        if f != seq2seq_zero:
            run_attributions(f,df_val,"summed_attr","attributions/", "IG")
    
    '''
    for l in range(4):
        layer = "results/best_ED_classify/best_ED_classify_layer"+str(l)+".enc_dec_attns"
        for h in range(8):
            tgt_head = "layer{}head{}".format(l,h)
            print("tgt_head: ",tgt_head)
            run_attributions(layer,df_val,tgt_head,"results/best_ED_classify","attn")

    for l in range(4):
        layer = "results/best_seq2seq/best_seq2seq_layer"+str(l)+".enc_dec_attns"
        for h in range(8):
            tgt_head = "layer{}head{}".format(l,h)
            print("tgt_head: ",tgt_head)
    ''' 
