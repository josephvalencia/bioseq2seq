import sys

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

def bio_grad(models):
    
    for i,m in enumerate(models):
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$GRAD_TEST_BIO --checkpoint ${{CHKPT_DIR}}/{m} --name ${{OUT_DIR}}/{outname}/ --tgt_class PC --tgt_pos 1'
        print(cmd)

def EDC_grad(models):
    
    for i,m in enumerate(models): 
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$GRAD_TEST_EDC --checkpoint ${{CHKPT_DIR}}/{m} --name ${{OUT_DIR}}/{outname}/'
        print(cmd)

def bio_pred(models):
    
    for j in range(2):
        for i,m in enumerate(models): 
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_BIO_CLASS --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/ --attn_save_layer {j}'
            print(cmd)

def bio_full_pred(models):
    
    for i,m in enumerate(models): 
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$PRED_TEST_BIO --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
        print(cmd)
        
def EDC_pred(models):
    
    for j in range(16):
        for i,m in enumerate(models):
            outname = m.split('.pt')[0].replace('/','')
            cmd = f'$PRED_TEST_EDC --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/ --attn_save_layer {j}'
            print(cmd)

def bio_designed_pred(models):
    
    for i,m in enumerate(models): 
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$PRED_DESIGNED_BIO_NC --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/'
        print(cmd)
        
def EDC_designed_pred(models):
    
    for i,m in enumerate(models):
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$PRED_DESIGNED_EDC_NC --checkpoint ${{CHKPT_DIR}}/{m} --output_name ${{OUT_DIR}}/{outname}/ '
        print(cmd)

def bio_attr(models,mode,dataset):
  
    sample_size = None
    if mode == 'MDIG':
        sample_size = 32
    elif mode == 'IG':
        sample_size = 128
    elif mode == 'EG':
        sample_size = 512


    for i,m in enumerate(models): 
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$ATTR_{dataset}_BIO --checkpoint ${{CHKPT_DIR}}/{m} --name ${{OUT_DIR}}/{outname}/ --attribution_mode {mode} --tgt_class PC --tgt_pos 1 --max_alpha 1.0'
        if sample_size:
            cmd += f' --sample_size {sample_size}'
        print(cmd)

def EDC_attr(models,mode,dataset):
    
    sample_size = None
    if mode == 'MDIG':
        sample_size = 32
    elif mode == 'IG':
        sample_size = 128
    elif mode == 'EG':
        sample_size = 512

    for i,m in enumerate(models): 
        outname = m.split('.pt')[0].replace('/','')
        cmd = f'$ATTR_{dataset}_VERIFIED_EDC --checkpoint ${{CHKPT_DIR}}/{m} --name ${{OUT_DIR}}/{outname}/ --attribution_mode {mode} --tgt_class PC --tgt_pos 1'
        if sample_size:
            cmd += f' --sample_size {sample_size}'
        print(cmd)
    pass

if __name__ == "__main__":
    
    models_file = sys.argv[1]
    
    with open(models_file) as inFile: 
        models = [m.strip() for m in inFile.readlines()]
    
    #EDC_pred(models)
    bio_attr(models,'ISM','TEST')
    #EDC_attr(models,'ISM','VAL')
