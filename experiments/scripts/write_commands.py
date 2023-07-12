import os
import argparse

def bio_train(models):
    for i,m in enumerate(models):
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$TRAIN_BIO --name bioseq2seq_{i+1} --random_seed {m}'
        print(cmd)

def edc_train(models):
    for i,m in enumerate(models):
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$TRAIN_EDC --name EDC_{i+1} --random_seed {m}'
        print(cmd)

def edc_small_train(models):
    for i,m in enumerate(models):
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$TRAIN_EDC_EQ --name EDC_eq_{i+1} --random_seed {m}'
        print(cmd)

def bio_pred_with_attn(models,outfile,parent='.'):
    with open(f'{parent}/{outfile}','w') as outFile:
        for j in range(2):
            for i,m in enumerate(models): 
                outname = m.split('.pt')[0].replace('/','')
                cmd = f'$PRED_TEST_BIO_CLASS --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/ --attn_save_layer {j}'
                outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)*2} commands') 

def edc_pred_with_attn(models,outfile,parent='.'):
    with open(f'{parent}/{outfile}','w') as outFile:
        for j in range(16):
            for i,m in enumerate(models):
                outname = m.split('.pt')[0].replace('/','')
                cmd = f'$PRED_TEST_EDC --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/ --attn_save_layer {j}'
                outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)*16} commands') 

def start_pred(models,outfile,parent='.'):
    with open(f'{parent}/{outfile}','w') as outFile:
        for i,m in enumerate(models): 
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_START --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands') 

def bio_pred(models,outfile,parent='.'):
    with open(f'{parent}/{outfile}','w') as outFile:
        for i,m in enumerate(models): 
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_BIO_CLASS --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands') 

def bio_full_pred(models,outfile,parent='.'):
    with open(f'{parent}/{outfile}','w') as outFile:
        for i,m in enumerate(models): 
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_BIO --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands') 

def edc_pred(models,outfile,parent='.'):
    with open(f'{parent}/{outfile}','w') as outFile:
        for i,m in enumerate(models):
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_EDC --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands') 

def start_pred(models,outfile,parent='.'):
    with open(f'{parent}/{outfile}','w') as outFile:
        for i,m in enumerate(models):
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_START --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
        print(f'wrote {outfile} with {len(models)} commands')
            
def attr(models,inf_mode,attr_mode,dataset,outfile,parent='.',alpha=0.5):
    with open(f'{parent}/{outfile}','w') as outFile:
        for i,m in enumerate(models): 
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$ATTR_{dataset}_{inf_mode} --checkpoint ${{CHKPT_DIR}}/{m} --name ${{OUT_DIR}}/{outname}/ --attribution_mode {attr_mode} --tgt_class PC --tgt_pos 1'
            if attr_mode == 'MDIG':
                cmd += f' --sample_size 32 --max_alpha {alpha}'
            elif attr_mode == 'IG':
                cmd += ' --sample_size 128'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands')

def mdig_val_attr(models,inf_mode,attr_mode,dataset,outfile,parent='.'):
    with open(f'{parent}/{outfile}','w') as outFile:
        for a in [0.1,0.25,0.5,0.75,1.0]: 
            for i,m in enumerate(models): 
                outname = m.split('.pt')[0].replace('/','')
                cmd = f'$ATTR_{dataset}_{inf_mode} --checkpoint ${{CHKPT_DIR}}/{m} --name ${{OUT_DIR}}/{outname}/ --attribution_mode {attr_mode} --tgt_class PC --tgt_pos 1'
                cmd += f' --sample_size 32 --max_alpha {a}'
                outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)*5} commands')

def build_all_pred_scripts(bio_rep_file,edc_rep_file,edc_small_rep_file,cnn_rep_file,start_rep_file,weighted_rep_file,parent):

    def load_files(rep_file):
        # ingest model replicate .pt files
        with open(rep_file) as inFile: 
            replicates = [m.strip() for m in inFile.readlines()]
        return replicates

    bio_replicates = load_files(bio_rep_file)
    edc_replicates = load_files(edc_rep_file)
    edc_small_replicates = load_files(edc_small_rep_file)
    cnn_replicates = load_files(cnn_rep_file)
    start_replicates = load_files(start_rep_file)
    weighted_replicates = load_files(weighted_rep_file)

    if not os.path.isdir(parent):
        os.mkdir(parent)

    # regular test set predictions
    bio_full_pred(bio_replicates,'pred_bioseq2seq.txt',parent=parent)
    bio_full_pred(cnn_replicates,'pred_class_bioseq2seq_CNN.txt',parent=parent)
    bio_full_pred(weighted_replicates,'pred_class_bioseq2seq_weighted.txt',parent=parent)
    
    bio_pred(bio_replicates,'pred_class_bioseq2seq.txt',parent=parent)
    bio_pred(cnn_replicates,'pred_class_bioseq2seq_CNN.txt',parent=parent)
    bio_pred(weighted_replicates,'pred_class_bioseq2seq_weighted.txt',parent=parent)
    
    edc_pred(edc_replicates,'pred_EDC.txt',parent=parent)
    edc_pred(edc_small_replicates,'pred_EDC_small.txt',parent=parent)
    start_pred(start_replicates,'pred_start.txt',parent=parent)

    # test set preds with encoder-decoder attention
    bio_pred_with_attn(bio_replicates,'pred_with_attn_bioseq2seq.txt',parent=parent)
    bio_pred_with_attn(weighted_replicates,'pred_with_attn_bioseq2seq_weighted.txt',parent=parent)
    bio_pred_with_attn(cnn_replicates,'pred_with_attn_bioseq2seq_CNN.txt',parent=parent)
    edc_pred_with_attn(edc_replicates,'pred_with_attn_EDC.txt',parent=parent)

    # verified validation set attributions
    dataset = 'VAL_VERIFIED'
    reps = [bio_replicates,weighted_replicates,cnn_replicates,edc_replicates]
    for attr_mode in ['MDIG','IG','grad','ISM']:
        for replicates,inf_mode in zip(reps, ['BIO','BIO','BIO','EDC']):
            suffix = 'bioseq2seq' if inf_mode == 'BIO' else inf_mode 
            prefix = 'uniform_ig' if attr_mode == 'IG' else attr_mode.lower()
            filename = f'{prefix}_{dataset.lower()}_{suffix}.txt'
            if attr_mode == 'MDIG':
                mdig_val_attr(replicates,inf_mode,attr_mode,dataset,filename,parent=parent)
            else: 
                attr(replicates,inf_mode,attr_mode,dataset,filename,parent=parent)
        
    # verified test set attributions
    attr(bio_replicates,'BIO','ISM','TEST_VERIFIED','ism_test_verified_bioseq2seq.txt',parent=parent)
    attr(bio_replicates,'BIO','grad','TEST_VERIFIED','grad_test_verified_bioseq2seq.txt',parent=parent)
    attr(bio_replicates,'BIO','MDIG','TEST_VERIFIED','mdig_test_verified_bioseq2seq.txt',alpha=0.5,parent=parent)
    
    attr(cnn_replicates,'BIO','ISM','TEST_VERIFIED','ism_test_verified_bioseq2seq_cnn.txt',parent=parent)
    attr(cnn_replicates,'BIO','grad','TEST_VERIFIED','grad_test_verified_bioseq2seq_cnn.txt',parent=parent)
    attr(cnn_replicates,'BIO','MDIG','TEST_VERIFIED','mdig_test_verified_bioseq2seq_cnn.txt',alpha=0.5,parent=parent)
    
    attr(weighted_replicates,'BIO','ISM','TEST_VERIFIED','ism_test_verified_bioseq2seq_weighted.txt',parent=parent)
    attr(weighted_replicates,'BIO','grad','TEST_VERIFIED','grad_test_verified_bioseq2seq_weighted.txt',parent=parent)
    attr(weighted_replicates,'BIO','MDIG','TEST_VERIFIED','mdig_test_verified_bioseq2seq_weighted.txt',alpha=0.5,parent=parent)
    
    attr(edc_replicates,'EDC','ISM','TEST_VERIFIED','ism_test_verified_EDC.txt',parent=parent) 
    attr(edc_replicates,'EDC','grad','TEST_VERIFIED','grad_test_verified_EDC.txt',parent=parent) 
    attr(edc_replicates,'EDC','MDIG','TEST_VERIFIED','mdig_test_verified_EDC.txt',alpha=0.1,parent=parent) 
    
    # bioseq2seq attributions only on larger datsets
    attr(bio_replicates,'BIO','MDIG','TEST_FULL','mdig_test_full_bioseq2seq.txt',alpha=0.5,parent=parent) 
    attr(weighted_replicates,'BIO','MDIG','TEST_FULL','mdig_test_full_bioseq2seq_weighted.txt',alpha=0.5,parent=parent) 
    attr(cnn_replicates,'BIO','MDIG','TEST_FULL','mdig_test_full_bioseq2seq_cnn.txt',alpha=0.5,parent=parent) 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bio',type=str,required=True)
    parser.add_argument('--edc',type=str,required=True)
    parser.add_argument('--edc_small',type=str,required=True)
    parser.add_argument('--cnn_bio',type=str,required=True)
    parser.add_argument('--start',type=str,required=True)
    parser.add_argument('--weighted_bio',type=str,required=True)
    parser.add_argument('--out_dir',default='.',type=str,required=False)
    args = parser.parse_args()
    
    build_all_pred_scripts(args.bio,
                        args.edc,
                        args.edc_small,
                        args.cnn_bio,
                        args.start,
                        args.weighted_bio,
                        args.out_dir)
