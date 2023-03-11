import yaml
import configargparse
import os,re
import pandas as pd

from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import ScalarFormatter

class ScalarFormatterClass(ScalarFormatter):
       def _set_format(self):
                 self.format = "%1.1f"

def setup_fonts():
    
    # manually add Helvetica for CentOS, if it fails just let matplotlib define defaults 
    try:
        font_path = os.path.join(Path.home(),'.fonts','Helvetica.ttc') 
        font_manager.fontManager.addfont(font_path) 
    except:
        print('Helvetica.ttc not found')

    mpl.rcParams['font.family'] = 'Helvetica'
    plt.rcParams.update({'font.family':'Helvetica'})
    sns.set_theme(font='Helvetica',context='paper',style='ticks')

def parse_config():

    p = configargparse.ArgParser() 
    
    p.add('--c','--config',required=False,is_config_file=True,help='path to config file')
    p.add('--test_prefix',help='test dataset (.csv)' )
    p.add('--data_dir',help='data directory' )
    p.add('--val_prefix',help='validation dataset (.csv)')
    p.add('--train_prefix',help='train dataset (.csv)')
    p.add('--competitors_results',help='competitors results (.csv)')
    p.add('--bioseq2seq_results',help='bioseq2seq results (.csv)')
    p.add('--best_BIO_DIR',help ='best bioseq2seq encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_EDC_DIR',help ='best EDC encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_BIO_chkpt',help ='best bioseq2seq encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_EDC_chkpt',help ='best EDC encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--all_BIO_replicates',help ='best bioseq2seq encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--all_EDC_replicates',help ='best EDC encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--all_EDC_small_replicates',help ='best EDC encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_BIO_EDA',help ='best bioseq2seq encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_EDC_EDA',help ='best EDC encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--reference_class',help ='class type')
    p.add('--position',help ='class type')
    return p.parse_known_args()

def grad_simplex_correction(input_grad):
    # Macdandzic et .al 2022 https://doi.org/10.1101/2022.04.29.490102
    input_grad -= input_grad.mean(axis=-1,keepdims=True)
    return input_grad

def build_EDA_file_list(args,parent):
    
    file_list = [f'{parent}EDA_layer{l}.npz' for l in range(args['n_layers'])] 
    args['path_list'] = file_list
    return args

def get_CDS_loc(cds,rna):
    
    # use provided CDS for mRNA 
    if cds != "-1": 
        splits = cds.split(":")
        clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
        splits = [clean(x) for x in splits]
        start,end = tuple([int(x) for x in splits])
    # impute longest ORF as CDS for lncRNA
    else:
        start,end = getLongestORF(rna)
    
    return start,end
        
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

def getFirstORF(mRNA):
    ORF_start = -1
    ORF_end = -1
    longestORF = 0
    is_found = False
    for startMatch in re.finditer('ATG',mRNA):
        remaining = mRNA[startMatch.start():]
        if re.search('TGA|TAG|TAA',remaining):
            for stopMatch in re.finditer('TGA|TAG|TAA',remaining):
                ORF = remaining[0:stopMatch.end()]
                if len(ORF) % 3 == 0 and len(ORF) > 30:
                    ORF_start = startMatch.start()
                    ORF_end = startMatch.start()+stopMatch.end()
                    return ORF_start, ORF_end 
    return ORF_start,ORF_end

def load_CDS(combined_file):

    df = pd.read_csv(combined_file,sep="\t")
    ids_list = df['ID'].values.tolist()
    cds_list = df['CDS'].values.tolist()
    rna_list = df['RNA'].values.tolist()

    temp = []
    # identify largest ORF as CDS for lncRNA
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
