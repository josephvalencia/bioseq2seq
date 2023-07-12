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

def bio_pred_with_attn(models,outfile):
    with open(outfile,'w') as outFile:
        for j in range(2):
            for i,m in enumerate(models): 
                outname = m.split('.pt')[0].replace('/','')
                cmd = f'$PRED_TEST_BIO_CLASS --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/ --attn_save_layer {j}'
                outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)*2} commands') 

def edc_pred_with_attn(models,outfile):
    with open(outfile,'w') as outFile:
        for j in range(16):
            for i,m in enumerate(models):
                outname = m.split('.pt')[0].replace('/','')
                cmd = f'$PRED_TEST_EDC --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/ --attn_save_layer {j}'
                outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)*16} commands') 

def start_pred(models,outfile):
    with open(outfile,'w') as outFile:
        for i,m in enumerate(models): 
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_START --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands') 

def bio_pred(models,outfile):
    with open(outfile,'w') as outFile:
        for i,m in enumerate(models): 
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_BIO_CLASS --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands') 

def bio_full_pred(models,outfile):
    with open(outfile,'w') as outFile:
        for i,m in enumerate(models): 
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_BIO --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands') 

def edc_pred(models,outfile):
    with open(outfile,'w') as outFile:
        for i,m in enumerate(models):
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_EDC --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands') 

def start_pred(models,outfile):
    with open(outfile,'w') as outFile:
        for i,m in enumerate(models):
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_START --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
            outFile.write(cmd+'\n')
        print(f'wrote {outfile} with {len(models)} commands')
            
def attr(models,inf_mode,attr_mode,dataset,outfile,alpha=0.5):
    with open(outfile,'w') as outFile:
        for i,m in enumerate(models): 
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$ATTR_{dataset}_{inf_mode} --checkpoint ${{CHKPT_DIR}}/{m} --name ${{OUT_DIR}}/{outname}/ --attribution_mode {attr_mode} --tgt_class PC --tgt_pos 1'
            if attr_mode == 'MDIG':
                cmd += f' --sample_size 32 --max_alpha {alpha}'
            elif attr_mode == 'IG':
                cmd += ' --sample_size 128'
            outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)} commands')

def mdig_val_attr(models,inf_mode,attr_mode,dataset,outfile):
    with open(outfile,'w') as outFile:
        for a in [0.1,0.25,0.5,0.75,1.0]: 
            for i,m in enumerate(models): 
                outname = m.split('.pt')[0].replace('/','')
                cmd = f'$ATTR_{dataset}_{inf_mode} --checkpoint ${{CHKPT_DIR}}/{m} --name ${{OUT_DIR}}/{outname}/ --attribution_mode {attr_mode} --tgt_class PC --tgt_pos 1'
                cmd += f' --sample_size 32 --max_alpha {a}'
                outFile.write(cmd+'\n')
    print(f'wrote {outfile} with {len(models)*5} commands')

def build_all_pred_scripts(bio_rep_file,edc_rep_file,edc_small_rep_file,cnn_rep_file,start_rep_file):

    # ingest model replicate .pt files
    with open(bio_rep_file) as inFile: 
        bio_replicates = [m.strip() for m in inFile.readlines()]
    with open(edc_rep_file) as inFile: 
        edc_replicates = [m.strip() for m in inFile.readlines()]
    with open(edc_small_rep_file) as inFile: 
        edc_small_replicates = [m.strip() for m in inFile.readlines()]
    with open(cnn_rep_file) as inFile:
        cnn_replicates = [m.strip() for m in inFile.readlines()]
    with open(start_rep_file) as inFile:
        start_replicates = [m.strip() for m in inFile.readlines()]

    # regular test set predictions
    bio_full_pred(bio_replicates,'pred_bioseq2seq.txt')
    bio_pred(bio_replicates,'pred_class_bioseq2seq.txt')
    start_pred(start_replicates,'pred_start.txt')
    edc_pred(edc_replicates,'pred_EDC.txt')
    edc_pred(edc_small_replicates,'pred_EDC_small.txt')
    start_pred(start_replicates,'pred_start.txt')
    bio_pred(cnn_replicates,'pred_class_bioseq2seq_CNN.txt')

    # test set preds with encoder-decoder attention
    bio_pred_with_attn(bio_replicates,'pred_with_attn_bioseq2seq.txt')
    edc_pred_with_attn(edc_replicates,'pred_with_attn_EDC.txt')

    # verified validation set attributions
    dataset = 'VAL_VERIFIED'
    for attr_mode in ['MDIG','IG','grad','ISM']:
        for inf_mode in ['BIO','EDC']:
            suffix = 'bioseq2seq' if inf_mode == 'BIO' else inf_mode 
            prefix = 'uniform_ig' if attr_mode == 'IG' else attr_mode.lower()
            filename = f'{prefix}_{dataset.lower()}_{suffix}.txt'
            replicates = bio_replicates if inf_mode == 'BIO' else edc_replicates
            if attr_mode == 'MDIG':
                mdig_val_attr(replicates,inf_mode,attr_mode,dataset,filename)
            else: 
                attr(replicates,inf_mode,attr_mode,dataset,filename)
        
    # verified test set attributions
    attr(bio_replicates,'BIO','ISM','TEST_VERIFED','ism_test_verified_bioseq2seq.txt')
    attr(edc_replicates,'EDC','ISM','TEST_VERIFED','ism_test_verified_EDC.txt') 
    attr(bio_replicates,'BIO','grad','TEST_VERIFED','grad_test_verified_bioseq2seq.txt')
    attr(edc_replicates,'EDC','grad','TEST_VERIFED','grad_test_verified_EDC.txt') 
    attr(bio_replicates,'BIO','MDIG','TEST_VERIFED','mdig_test_verified_bioseq2seq.txt',alpha=0.5)
    attr(edc_replicates,'EDC','MDIG','TEST_VERIFED','mdig_test_verified_EDC.txt',alpha=0.1) 
    
    # bioseq2seq attributions only on larger datsets
    attr(bio_replicates,'BIO','MDIG','TEST_FULL','mdig_test_full_bioseq2seq.txt',alpha=0.5) 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bio_rep_file',type=str,required=True)
    parser.add_argument('--edc_rep_file',type=str,required=True)
    parser.add_argument('--edc_small_rep_file',type=str,required=True)
    parser.add_argument('--cnn_bio_rep_file',type=str,required=True)
    parser.add_argument('--start_rep_file',type=str,required=True)
    args = parser.parse_args()
    
    build_all_pred_scripts(args.bio_rep_file,
                        args.edc_rep_file,
                        args.edc_small_rep_file,
                        args.cnn_bio_rep_file,
                        args.start_rep_file)
