import yaml
import configargparse
import re
import pandas as pd

def parse_config():

    p = configargparse.ArgParser() 
    
    p.add('--c','--config',required=False,is_config_file=True,help='path to config file')
    p.add('--test_csv',help='test dataset (.csv)' )
    p.add('--val_csv',help='validation dataset (.csv)')
    p.add('--train_csv',help='train dataset (.csv)')
    p.add('--competitors_results',help='competitors results (.csv)')
    p.add('--bioseq2seq_results',help='bioseq2seq results (.csv)')
    p.add('--EDC_results',help='EDC results (.csv)')
    p.add('--best_BIO_EDA',help ='best bioseq2seq encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_EDC_EDA',help ='best EDC encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_BIO_DIR',help ='best bioseq2seq encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_EDC_DIR',help ='best EDC encoder-decoder attention (.enc_dec_attn)',type=yaml.safe_load)
    p.add('--best_BIO_grad_PC',help = 'best bioseq2seq Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_EDC_grad_PC',help = 'best EDC Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_BIO_grad_NC',help = 'best bioseq2seq Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_EDC_grad_NC',help = 'best EDC Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_BIO_inputXgrad_PC',help = 'best bioseq2seq Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_EDC_inputXgrad_PC',help = 'best EDC Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_BIO_inputXgrad_NC',help = 'best bioseq2seq Integrated Gradients (.ig)',type=yaml.safe_load)
    p.add('--best_EDC_inputXgrad_NC',help = 'best EDC Integrated Gradients (.ig)',type=yaml.safe_load)
    return p.parse_known_args()

def grad_simplex_correction(input_grad):
    # Macdandzic et .al 2022 https://doi.org/10.1101/2022.04.29.490102
    input_grad -= input_grad.mean(axis=-1,keepdims=True)
    return input_grad

def build_EDA_file_list(args,parent):
    
    file_list = [f'{parent}EDA_layer{l}.npz' for l in range(args['n_layers'])] 
    args['path_list'] = file_list
    return args

def get_CDS_start(cds,rna):
    
    # use provided CDS for mRNA 
    if cds != "-1": 
        splits = cds.split(":")
        clean = lambda x : x[1:] if x.startswith("<") or x.startswith(">") else x
        splits = [clean(x) for x in splits]
        start,end = tuple([int(x) for x in splits])
    # impute longest ORF as CDS for lncRNA
    else:
        start,end = getLongestORF(rna)
    
    return start
        
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
