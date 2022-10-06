import os
import pandas as pd
from utils import parse_config, add_file_list, getLongestORF, get_CDS_start

def aggregate_motifs(streme_files,summary_name):
    all_files = ' '.join(streme_files)
    os.system(f'/local/cluster/meme_4.12.0/bin/meme2meme {all_files} > {summary_name}')

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
       
        # positive class inputXgrad
        EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/inputXgrad_PC/'
        BIO_dir = f'{attr_dir}/best_seq2seq_{trial_name}/inputXgrad_PC/'
        BIO_PC = f'streme -p {BIO_dir}/positive_motifs.fa -n {BIO_dir}/negative_motifs.fa -oc {BIO_dir}/streme_out \
                -rna -minw 6 -maxw 15 -pvt 1e-5 -patience 0' 
        EDC_PC = f'streme -p {EDC_dir}/positive_motifs.fa -n {EDC_dir}/negative_motifs.fa -oc {EDC_dir}/streme_out \
                -rna -minw 6 -maxw 15 -pvt 1e-5 -patience 0'
        os.system(BIO_PC)
        os.system(EDC_PC)
        
        # negative class inputXgrad
        EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/inputXgrad_NC/'
        BIO_dir = f'{attr_dir}/best_seq2seq_{trial_name}/inputXgrad_NC/'
        BIO_NC = f'streme -p {BIO_dir}/positive_motifs.fa -n {BIO_dir}/negative_motifs.fa -oc {BIO_dir}/streme_out \
                -rna -minw 6 -maxw 15 -pvt 1e-5 -patience 0' 
        EDC_NC = f'streme -p {EDC_dir}/positive_motifs.fa -n {EDC_dir}/negative_motifs.fa -oc {EDC_dir}/streme_out \
                -rna -minw 6 -maxw 15 -pvt 1e-5 -patience 0'
        os.system(BIO_NC)
        os.system(EDC_NC)
    
        for l,f in enumerate(best_BIO_EDA['path_list']):
            heads = []
            for h in range(8):
                BIO_dir = f'{attr_dir}best_seq2seq_{trial_name}/layer{l}head{h}/'
                cmd1 = f'streme -p {BIO_dir}positive_motifs.fa -n {BIO_dir}negative_motifs.fa -oc {BIO_dir}streme_out \
                        -rna -minw 6 -maxw 15 -pvt 1e-5 -patience 0' 
                os.system(cmd1)
                xml = f'{BIO_dir}streme_out/streme.xml'
                heads.append(xml)
            summary_file = f'{attr_dir}/best_seq2seq_{trial_name}/layer{l}_summary.txt'
            aggregate_motifs(heads,summary_file)
        
        for l,f in enumerate(best_EDC_EDA['path_list']):
            heads = []
            for h in range(8):
                EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/layer{l}head{h}/'
                cmd1 = f'streme -p {EDC_dir}/positive_motifs.fa -n {EDC_dir}/negative_motifs.fa -oc {EDC_dir}/streme_out \
                        -rna -minw 6 -maxw 15 -pvt 1e-5 -patience 0'
                os.system(cmd1)
                xml = f'{EDC_dir}streme_out/streme.xml'
                heads.append(xml)
            summary_file = f'{attr_dir}/best_EDC_{trial_name}/layer{l}_summary.txt'
            aggregate_motifs(heads,summary_file)
