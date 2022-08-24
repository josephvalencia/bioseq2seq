import sys

def bio_grad():
    pass
def EDC_grad():
    pass
def bio_pred():
    pass
def bio_full_pred():
    pass

def EDC_pred(fname):
    with open(fname) as inFile: 
        models = [m.strip() for m in inFile.readlines()]
    for j in range(12):
        for i,m in enumerate(models):
            a = '$PRED_TEST_EDC --checkpoint ${CHKPT_DIR}'
            b = '--output_name ${out_dir}/'
            c = f'EDC_large_{i+1}_test --rank {i} --attn_save_layer {j}'
            cmd = f'{a}{m} {b}{c}'
            print(cmd)

def bio_EG():
    pass
def EDC_EG():
    pass


if __name__ == "__main__":
    
    models_file = sys.argv[1]
    EDC_pred(models_file)
