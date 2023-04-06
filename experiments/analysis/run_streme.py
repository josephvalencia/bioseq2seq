import os
import pandas as pd
from utils import parse_config, build_EDA_file_list, getLongestORF, get_CDS_start

if __name__ == "__main__":

    args, unknown_args = parse_config()
    
    test_file = os.path.join(args.data_dir,args.test_prefix+'.csv')
    train_file = os.path.join(args.data_dir,args.train_prefix+'.csv')
    val_file = os.path.join(args.data_dir,args.val_prefix+'.csv')
    df_test = pd.read_csv(test_file,sep='\t').set_index('ID')
    df_train = pd.read_csv(train_file,sep='\t').set_index('ID')
    
    # load attribution files from config
    best_BIO_EDA = build_EDA_file_list(args.best_BIO_EDA,args.best_BIO_DIR)
    best_EDC_EDA = build_EDA_file_list(args.best_EDC_EDA,args.best_EDC_DIR)

    # build output directory
    config = args.c
    config_prefix = config.split('.yaml')[0]
    output_dir  =  f'results_{config_prefix}'
    attr_dir  =  f'{output_dir}/attr'
    if not os.path.isdir(output_dir):
        print("Building directory ...")
        os.mkdir(output_dir)
    # make subdir for attribution loci results  
    if not os.path.isdir(attr_dir):
        os.mkdir(attr_dir)

    # load attribution files from config
    best_BIO_EDA = build_EDA_file_list(args.best_BIO_EDA,args.best_BIO_DIR)
    best_EDC_EDA = build_EDA_file_list(args.best_EDC_EDA,args.best_EDC_DIR)
    prefix = args.train_prefix.replace('train','train_RNA')
    best_BIO_mdig = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.MDIG.npz')
    best_EDC_mdig = os.path.join(args.best_EDC_DIR,f'{prefix}.{args.reference_class}.{args.position}.MDIG.npz')
    best_BIO_grad = os.path.join(args.best_BIO_DIR,f'{prefix}.{args.reference_class}.{args.position}.grad.npz')
    best_EDC_grad = os.path.join(args.best_EDC_DIR,f'{prefix}.{args.reference_class}.{args.position}.grad.npz')
    
    groups = [['PC','NC'],['NC','PC'],['PC','PC'],['NC','NC']]
    cross_metrics = [['max','max'],['max','min'],['min','max'],['rolling-abs','rolling-abs'],['random','random']]
    same_metrics = [['max','min'],['max','random'],['min','random'],['rolling-abs','random']]
    
    cmd = 'streme -p {}positive_motifs.fa -n {}negative_motifs.fa -oc {}streme_out -rna -minw 6 -maxw 15 -pvt 1e-5 -patience 0' 

    for i,g in enumerate(groups):
        metrics = same_metrics if i>1 else cross_metrics
        for m in metrics:
            # g[0] and g[1] are transcript type for pos and neg sets
            # m[0] and m[1] are loci of interest for pos and neg sets
            a = f'{g[0]}-{m[0]}'
            b = f'{g[1]}-{m[1]}'
            assert a != b
            trial_name = f'{a}_{b}'
          
            # grad
            EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/{prefix}_{args.reference_class}_{args.position}_grad/'
            BIO_dir = f'{attr_dir}/best_seq2seq_{trial_name}/{prefix}_{args.reference_class}_{args.position}_grad/'
            BIO_grad = cmd.format(*[BIO_dir]*3)
            EDC_grad = cmd.format(*[EDC_dir]*3)
            print(BIO_grad)
            print(EDC_grad)
            
            # MDIG
            EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/{prefix}_{args.reference_class}_{args.position}_MDIG/'
            BIO_dir = f'{attr_dir}/best_seq2seq_{trial_name}/{prefix}_{args.reference_class}_{args.position}_MDIG/'
            BIO_mdig = cmd.format(*[BIO_dir]*3)
            EDC_mdig = cmd.format(*[EDC_dir]*3)
            print(BIO_mdig)
            #print(EDC_mdig)
       
            '''
            for l,f in enumerate(best_BIO_EDA['path_list']):
                heads = []
                for h in range(8):
                    BIO_dir = f'{attr_dir}best_seq2seq_{trial_name}/layer{l}head{h}/'
                    cmd1 = cmd.format(*[BIO_dir]*3)
                    run(cmd1)
                    xml = f'{BIO_dir}streme_out/streme.xml'
                    heads.append(xml)
                summary_file = f'{attr_dir}/best_seq2seq_{trial_name}/layer{l}_summary.txt'
                #aggregate_motifs(heads,summary_file)
            
            for l,f in enumerate(best_EDC_EDA['path_list']):
                heads = []
                for h in range(8):
                    EDC_dir = f'{attr_dir}/best_EDC_{trial_name}/layer{l}head{h}/'
                    cmd1 = cmd.format(*[EDC_dir]*3)
                    run(cmd1)
                    xml = f'{EDC_dir}streme_out/streme.xml'
                    heads.append(xml)
                summary_file = f'{attr_dir}/best_EDC_{trial_name}/layer{l}_summary.txt'
                #aggregate_motifs(heads,summary_file)
            '''
