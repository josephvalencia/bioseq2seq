import sys,random
import json
import os,re
#import configargparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr,kendalltau,ttest_ind,entropy
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import jensenshannon
from collections import defaultdict
from IPython.display import Image

from Bio.Data import CodonTable
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

from utils import parse_config, add_file_list, getLongestORF

def parse_args():
    """parse required and optional configuration arguments.""" 
    
    parser = configargparse.ArgParser()
    
    # required args
    parser.add_argument("--train",help = "file containing rna to protein dataset")
    parser.add_argument("--val",help = "file containing rna to protein dataset")
    
    # optional args
    parser.add_argument("--save-directory","--s", default = "checkpoints/", help = "name of directory for saving model checkpoints")
    parser.add_argument("--learning-rate","--lr", type = float, default = 1e-3,help = "optimizer learning rate")
    parser.add_argument("--max-epochs","--e", type = int, default = 100000,help = "maximum number of training epochs" )
    parser.add_argument("--report-every",'--r', type = int, default = 750, help = "number of iterations before calculating statistics")
    parser.add_argument("--mode", default = "combined", help = "training mode. classify for binary classification. translate for full translation")
    parser.add_argument("--num_gpus","--g", type = int, default = 1, help = "number of gpus to use on node")
    parser.add_argument("--accum_steps", type = int, default = 4, help = "number of batches to accumulate gradients before update")
    parser.add_argument("--max_tokens",type = int , default = 4500, help = "max number of tokens in training batch")
    parser.add_argument("--patience", type = int, default = 5, help = "maximum epochs without improvement")
    parser.add_argument("--address",default =  "127.0.0.1",help = "ip address for master process in distributed training")
    parser.add_argument("--port",default = "6000",help = "port for master process in distributed training")
    parser.add_argument("--n_enc_layers",type=int,default = 6,help= "number of encoder layers")
    parser.add_argument("--n_dec_layers",type=int,default = 6,help="number of decoder layers")
    parser.add_argument("--model_dim",type=int,default = 64,help ="size of hidden context embeddings")
    parser.add_argument("--max_rel_pos",type=int,default = 8,help="max value of relative position embedding")

    # optional flags
    parser.add_argument("--checkpoint", "--c", help = "name of .pt model to initialize training")
    parser.add_argument("--finetune",action = "store_true", help = "reinitialize generator")
    parser.add_argument("--verbose",action="store_true")

    return parser.parse_args()


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
    output = name+"_lineplot.svg"
    plt.savefig(output)
    plt.close()

def get_top_k(array,k=1):

    k = k if k < len(array) else len(array)
    array = np.asarray(array)
    k_largest_inds = np.argpartition(array,-k)[-k:]
    k_largest_scores = array[k_largest_inds].tolist()
    k_largest_inds = k_largest_inds.tolist()

    return k_largest_scores,k_largest_inds

def get_min_k(array,k=1):

    k = k if k < len(array) else len(array)
    array = np.asarray(array)
    k_smallest_inds = np.argpartition(array,k)[:k]
    k_smallest_inds = k_smallest_inds.tolist()

    return k_smallest_inds

def top_indices(saved_file,tgt_field,positive_topk_file,negative_topk_file,groups,metrics,mode="attn"):
    
    df_storage = []
    negative_storage = []
    positive_storage = []
    val_list = []

    out_name = saved_file.split(".")[0]

    with open(saved_file) as inFile:
        for l in inFile:
            fields = json.loads(l)
            id_field = "TSCRIPT_ID" if mode == "attn" else "ID"
            id = fields[id_field]
            
            #src = fields['src'].split('<pad>')[0]
            #array = [float(x) for x in fields[tgt_field][:len(src)]]
            array = [float(x) for x in fields[tgt_field]]
            L = len(array)
            array = np.asarray(array) 
            
            name = id + "_" + mode
            window_size = 50
            
            # find various indices of interest
            max_idx = np.argmax(array).tolist()
            min_idx = np.argmin(array).tolist()
            smallest_magnitude_idx = np.argmin(np.abs(array)).tolist()
            largest_magnitude_idx = np.argmax(np.abs(array)).tolist()
            other_idx = [x for x in range(L) if x != max_idx]
            random_idx = random.sample(other_idx,1)[0]
            
            val_list.append(np.max(array))

            coding = True if (id.startswith('XM_') or id.startswith('NM_')) else False 
            both_storage = [positive_storage,negative_storage]

            for g,m,dataset in zip(groups,metrics,both_storage): 
                result = None
                # partition based on args
                if g == 'PC' and coding:
                    if m == 'max':
                        result = (id,max_idx)
                    elif m == 'min':
                        result = (id,min_idx)
                    elif m == 'random':
                        result = (id,random_idx)
                elif g == 'NC' and not coding:
                    if m == 'max':
                        result = (id,max_idx)
                    elif m == 'min':
                        result = (id,min_idx)
                    elif m == 'random':
                        result = (id,random_idx)
                if result is not None:
                    dataset.append(result) 

    val_mean = np.mean(val_list)
    val_std = np.std(val_list)
    print(f'Max val: mean={val_mean}, std={val_std}')
    
    # save top indices
    with open(positive_topk_file,'w') as outFile:
        for tscript,idx in positive_storage:
            outFile.write("{},{}\n".format(tscript,idx))

    with open(negative_topk_file,'w') as outFile:
        for tscript,idx in negative_storage:
            outFile.write("{},{}\n".format(tscript,idx))


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
            #left_bound = 24
            #right_bound = 26
            left_bound = 9
            right_bound = 11

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
            
            array = [float(x)  for x in array]
            inside = defaultdict(lambda : [])
            outside = defaultdict(lambda : [])

            legal_chars = {'A','C','G','T'}
            allowed = lambda codon : all([x in legal_chars for x in codon])
            inframe = lambda x : x >= cds_start and x<=cds_end-2 and (x-cds_start) % 3 == 0 

            # 5' UTR and out of frame and 3' UTR
            #outside_range = list(range(0,cds_start-3)) + list(range(cds_end,len(array) -3))
            for i in range(0,len(array)-3):
                #for i in outside_range:
                codon = seq[i:i+3]
                if allowed(codon) and not inframe(i):
                    score = sum(array[i:i+3])
                    outside[codon].append(score)
            
            # find average 
            for codon,scores in outside.items():
                avg = sum(scores) / len(scores) 
                argmax = np.argmax(np.abs(scores))
                avg = scores[argmax]
                info = {"tscript" : id ,"codon" : codon, "score" : avg/len(array), "status" : tscript_type, "segment" : "OUT"}
                extra.append(info)
            
            # inside CDS
            for i in range(cds_start,cds_end-3,3):
                codon = seq[i:i+3]
                if allowed(codon):
                    score = sum(array[i:i+3])
                    inside[codon].append(score)
           
           # average CDS
            for codon,scores in inside.items():
                #avg = sum(scores) / len(scores)
                argmax = np.argmax(np.abs(scores))
                avg = scores[argmax]
                info = {"tscript" : id ,"codon" : codon, "score" : avg/len(array), "status" : tscript_type, "segment" : 'CDS'}
                extra.append(info)
            
        codon_df = pd.DataFrame(extra)
        
        # valid codons
        a = codon_df[codon_df['codon'].str.len() == 3]
        scatter_plot(a,scatterplot_file)
        #boxplot_all(a,boxplot_file)
        
        # coding CDS only
        coding_cds = a[(a['segment'] == 'CDS') & (a['status'] == '<PC>')]
        no_start_stop_file = boxplot_file.split('.')[0]+'_no_start_stop.svg'
        boxplot_no_start_stop(coding_cds,no_start_stop_file)
        
def boxplot_no_start_stop(data,boxplot_file,**kws):

    plt.figure(figsize=(12,8))

    #start_stop = ['ATG','TAA','TAG','TGA']
    #data = data[~data['codon'].isin(start_stop)]
    medians = data.groupby('codon').median().sort_values(by='score',ascending=False)
    medians = medians.reset_index()
    order = medians['codon'].values.tolist()
    #median_scores = medians['score'].values.tolist() 
    
    #means = data.groupby('codon').mean().sort_values(by='score',ascending=False)
    #means = means.reset_index()
    #order = means['codon'].values.tolist()
    
    ax = sns.boxplot(x="codon",y="score",data=data,order=order,showfliers=False,linewidth=0.6,orient='v',palette='coolwarm_r') 
    #ax = sns.pointplot(x='codon', y='score',data=data,order=order)
    #ax = sns.barplot(x='codon',y='score',data=data,order=order)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 12)
    ax.set_ylabel('Max IG score per transcript')

    '''
    meds = [str(np.round(s,2)) for s in median_scores]
    ind = 0
    for tick in range(len(ax.get_xticklabels())):
        ax.text(tick+.2, median_scores[ind]+1, meds[ind], horizontalalignment='center', color='w', weight='semibold')
        ind += 1    
    ''' 
    plt.savefig(boxplot_file)
    plt.close()

def boxplot_all(data,boxplot_file):

    ax = sns.FacetGrid(data, row="status",col='segment', sharex=False,sharey=False)
    ax.map_dataframe(boxplot_fn)

    axes = ax.axes.flatten()
    for a in axes:
        a.set_title('')

    plt.tight_layout()
    plt.savefig(boxplot_file)
    plt.close()

def boxplot_fn(data, **kws):

    medians = data.groupby('codon').median().sort_values(by='score',ascending=False)
    medians = medians.reset_index()
    order = medians['codon'].values.tolist()
    x_order = order[:8] + order[-8:]
    
    ax = sns.boxplot(x="codon",y="score",data=data,order=x_order,showfliers=False,width=0.8,linewidth=0.6) 
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    palette = sns.color_palette()

    for item in ax.get_xticklabels():
        item.set_rotation(90)
    
    for i in range(len(ax.artists)):
        mybox = ax.artists[i]
        if i < 8:
            mybox.set_facecolor(palette[0])
        else:
            mybox.set_facecolor(palette[1])

def scatter_plot(a,scatterplot_file):
    
    #start_stop = ['ATG','TAA','TAG','TGA']
    #a = a[~a['codon'].isin(start_stop)]
    out = a[a['segment'] == 'OUT']
    cds = a[a['segment'] == 'CDS']
    nc = a[a['status'] == '<NC>']
    pc = a[a['status'] == '<PC>']

    out = out.groupby('codon').mean()
    cds = cds.groupby('codon').mean()
    pc = pc.groupby('codon').mean()
    nc = nc.groupby('codon').mean()
    
    by_status = pc.merge(nc,on='codon',suffixes=['_pc','_nc'])
    by_status['diff_coding'] = by_status['score_pc'] - by_status['score_nc'] 
    by_segment = cds.merge(out,on='codon',suffixes=['_cds','_out'])
    
    combined = by_segment.merge(by_status,on='codon')
    combined['diff_cds'] = combined['score_cds'] - combined['score_out']
    combined['diff_coding'] = combined['score_pc'] - combined['score_nc']
    #combined = combined.sort_values('diff_coding',ascending=False) 
    combined.reset_index(inplace=True)
    
    pc = pc.sort_values('score',ascending=False)
    pc.reset_index(inplace=True)
  
    #x_var = 'score_nc'
    #y_var = 'score_pc'
    x_var = 'score_nc'
    y_var = 'score_pc'

    min_x = np.min(combined[x_var].values)
    max_x = np.max(combined[x_var].values)
    min_y = np.min(combined[y_var].values)
    max_y = np.max(combined[y_var].values)

    ax = sns.scatterplot(x=x_var,y=y_var,data=combined)
    #ax = sns.lmplot(x='score_nc',y='score_pc',data=combined)
    
    x_spread = max_x - min_x
    y_spread = max_y - min_y
    ax.set_xlim(min_x - x_spread*0.05,max_x + x_spread*0.05)
    ax.set_ylim(min_y - y_spread*0.05,max_y + y_spread*0.05)
    
    for line in range(0,combined.shape[0]):
        ax.text(combined[x_var][line]+0.005*x_spread, combined[y_var][line]+0.005*y_spread, \
                combined['codon'][line], horizontalalignment='left',size='medium', color='black', weight='semibold')
    
    #ax = sns.pairplot(combined)
    plt.savefig(scatterplot_file)
    plt.close()


def significance(codon_df,significance_file):

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


def run_attributions(saved_file,df,tgt_field,parent_dir,groups,metrics,mode="attn"):

    attr_name = os.path.split(saved_file)[1]
    attr_name = attr_name.split('.')[0]
    prefix = f'{parent_dir}/{attr_name}_{tgt_field}/' 
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    print(parent_dir,tgt_field)

    # results files
    positive_indices_file = prefix+"positive_topk_idx.txt"
    negative_indices_file = prefix+"negative_topk_idx.txt"
    positive_motifs_file = prefix+"positive_motifs.fa"
    negative_motifs_file = prefix +"negative_motifs.fa"
    boxplot_file = prefix+"codon_boxplot.svg"
    scatterplot_file = prefix+"codon_scatterplot.svg"
    significance_file = prefix+"significance.txt"
    hist_file = prefix+"pos_hist.svg"

    top_indices(saved_file,tgt_field,positive_indices_file,negative_indices_file,groups,metrics,mode=mode)
    top_k_to_substrings(positive_indices_file,positive_motifs_file,df)
    top_k_to_substrings(negative_indices_file,negative_motifs_file,df)
    #get_positional_bias(coding_indices_file,noncoding_indices_file,df,hist_file)
    #codon_scores(saved_file,df,tgt_field,boxplot_file,scatterplot_file,significance_file,mode)

def get_positional_bias(coding_indices_file,noncoding_indices_file,df_data,hist_file):

    storage = []

    pc = pd.read_csv(coding_indices_file,names=['ID','start'])
    nc = pd.read_csv(noncoding_indices_file,names=['ID','start'])
    df_attn = pd.concat([pc,nc])
    print(df_attn) 
    df_data['cds_start'] = [get_CDS_start(cds,seq) for cds,seq in zip(df_data['CDS'].values.tolist(),df_data['RNA'].values.tolist())]
    df = pd.merge(df_attn,df_data,on='ID')
    df['rel_start'] = df['start'] - df['cds_start']-1
    df = df.drop(df.columns.difference(['Type','rel_start']),1)
    print(df)

    bins = np.arange(-750,1000,10)
    g = sns.displot(data=df,x='rel_start',col='Type',kind='hist',stat='density',bins=bins,element='step')
    #sns.histplot(data=pc,x='rel_start',hue='motif_id',binwidth=10,common_bins=True,ax=axs[1],legend=False,element='step')

    axes = g.axes.flatten()
    axes[0].set_title("")
    axes[0].set_xlabel("Position of max IG val rel. start")
    axes[0].set_ylabel("Density")
    axes[1].set_title("")
    axes[1].set_xlabel("Position of min IG val rel. to start longest ORF")
    axes[1].set_ylabel("")
    
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
                attr = np.asarray([float(x) for x in fields["summed_attr"]])
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


def attribution_loci_pipeline(): 

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

    # load attribution files from config
    best_seq_EDA = add_file_list(args.best_seq_EDA,'layers')
    best_EDC_EDA = add_file_list(args.best_EDC_EDA,'layers')
    best_seq_IG = add_file_list(args.best_seq_IG,'bases')
    best_EDC_IG = add_file_list(args.best_EDC_IG,'bases')
    best_seq_MDIG = add_file_list(args.best_seq_MDIG,'bases')
    best_EDC_MDIG = add_file_list(args.best_EDC_MDIG,'bases')
    
    groups = [['PC','NC'],['PC','PC'],['NC','NC']]
    metrics = [['max','max'],['max','min'],['max','random'],['min','random'],['random','random']]
    
    for g in groups:
        for m in metrics:
            # g[0] and g[1] are transcript type for pos and neg sets
            # m[0] and m[1] are loci of interest for pos and neg sets
            a = f'{g[0]}-{m[0]}'
            b = f'{g[1]}-{m[1]}'
            # ensure comparison groups are different 
            if a != b:
                trial_name = f'{a}_{b}'
                # build directories
                best_EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/'
                if not os.path.isdir(best_EDC_dir):
                    os.mkdir(best_EDC_dir)
                best_seq_dir = f'{attr_dir}/best_seq2seq_{trial_name}/'
                if not os.path.isdir(best_seq_dir):
                    os.mkdir(best_seq_dir)
                
                # run all IG bases for both models 
                for f in best_seq_IG['path_list']:
                    run_attributions(f,df_test,'summed_attr',best_seq_dir,g,m,'IG')
                for f in best_EDC_IG['path_list']: 
                    run_attributions(f,df_test,'summed_attr',best_EDC_dir,g,m,'IG')
                
                # run all EDA layers for both models
                for l,f in enumerate(best_seq_EDA['path_list']):
                    for h in range(8):
                        tgt_head = f'layer{l}head{h}'
                        run_attributions(f,df_test,tgt_head,best_seq_dir,g,m,'attn')
                for l,f in enumerate(best_EDC_EDA['path_list']):
                    for h in range(8):
                        tgt_head = f'layer{l}head{h}'
                        run_attributions(f,df_test,tgt_head,best_EDC_dir,g,m,'attn')

if __name__ == "__main__":
    
    attribution_loci_pipeline() 
