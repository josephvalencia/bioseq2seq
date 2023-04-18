import subprocess
import numpy as np
import os, sys, re
import pandas as pd
        
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

def pad_to_max_len(array,max_len):

    diff = max_len - array.shape[1]
    zeros = np.zeros((4,diff),dtype=array.dtype)
    padded = np.concatenate([array,zeros],axis=1)
    return padded

def save_transposed(parent,attr_type,mode='FULL'):
  
    df = pd.read_csv("data/mammalian_200-1200_train_balanced.csv",sep='\t').set_index('ID')
    #df = pd.read_csv("data/mammalian_200-1200_test_nonredundant_80.csv",sep='\t').set_index('ID')
    onehot_file = np.load(f"{parent}.onehot.npz")
    attribution_file = np.load(f"{parent}.{attr_type}.npz")
    logits_file = np.load(f"{parent}.wildtype_logit.npz")
     
    attr_storage = []
    onehot_storage = []
    min_len = 50 
    longest = min_len
    
    for tscript,array in attribution_file.items():
        onehot = onehot_file[tscript][:,2:6]
        seq = df.loc[tscript]['RNA']
        if attr_type == "grad":
            array = array[:,2:6]
        if attr_type == "MDIG" or attr_type == "ISM":
            array += logits_file[tscript][0]

        # center around zero 
        array = array - array.mean(axis=1,keepdims=1) 
        
        # save by area of interest
        s,e = getLongestORF(seq)
        if mode == "5-prime":
            onehot = onehot[:s,:]
            array = array[:s,:]
            dist = s 
        elif mode == "3-prime":
            onehot = onehot[e:,:]
            array = array[e:,:]
            dist = len(seq)-e
        elif mode == 'CDS':
            onehot = onehot[s:e,:]
            array = array[s:e,:]
            dist = e-s
        else:
            dist = len(seq) 
        if dist > longest:
            longest = dist
        if dist > min_len: 
            attr_storage.append(array)
            onehot_storage.append(onehot)
    
    # pad to longest
    print(f"longest {mode} is {longest}")
    attr_storage = [pad_to_max_len(array.T,longest) for array in attr_storage]
    onehot_storage = [pad_to_max_len(onehot.T,longest) for onehot in onehot_storage]
    
    # save in format required by TF-MoDisco 
    combined_onehot = np.stack(onehot_storage)
    combined_attr = np.stack(attr_storage)
    tempfile_onehot = f'{parent}.onehot.{mode}.modisco'
    tempfile_attr = f'{parent}.{attr_type}.{mode}.modisco'
    print(f'saving temporary files {tempfile_onehot} ({combined_onehot.shape}) and {tempfile_attr} ({combined_attr.shape})') 
    np.savez_compressed(tempfile_onehot,combined_onehot)
    np.savez_compressed(tempfile_attr,combined_attr)

def cleanup(tempfile):
    
    if 'modisco' in tempfile:
        os.remove(tempfile)
    else:
        print(f'Refusing to delete {tempfile}') 
def discover(parent,attr_type,mode):
   
    temp_onehot = f'{parent}.onehot.{mode}.modisco.npz'
    temp_attr = f'{parent}.{attr_type}.{mode}.modisco.npz'
    if not (os.path.exists(temp_onehot) and os.path.exists(temp_attr)):
        save_transposed(parent,attr_type,mode)
    
    result_name = f'{parent}.{attr_type}.{mode}.modisco_results.h5'
    cmd = ['modisco','motifs','-s',temp_onehot,'-a',temp_attr,'-n', '2000','-o',result_name,'-v']
    print(' '.join(cmd))
    subprocess.run(cmd)  
    #cleanup(temp_onehot)
    #cleanup(temp_attr)

if __name__ == "__main__":

    discover(sys.argv[1],sys.argv[2],sys.argv[3])
