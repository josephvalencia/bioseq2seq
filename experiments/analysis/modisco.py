import subprocess
import numpy as np
import os, sys

def pad_to_max_len(array,max_len):

    diff = max_len - array.shape[1]
    zeros = np.zeros((4,diff),dtype=array.dtype)
    padded = np.concatenate([array,zeros],axis=1)
    return padded

def save_transposed(filename,subset=True):
    
    saved = np.load(filename)
    storage = []

    for tscript,array in saved.items():
        if subset:
            array = array[:,2:6]
        padded_array = pad_to_max_len(array.T,1200)
        storage.append(padded_array)
   
    combined = np.stack(storage)
    print(filename,combined.shape) 
    filename = filename.replace('.npz','')
    tempfile = f'{filename}.modisco'
    np.savez_compressed(tempfile,combined)
    return tempfile+'.npz'

def cleanup(filename):
    
    filename = filename.replace('.npz','')
    tempfile = f'{filename}.modisco.npz'
    os.remove(tempfile)

def discover(onehot_file,attribution_file):

    temp_onehot = save_transposed(onehot_file)
    temp_attr = save_transposed(attribution_file,subset=False)
    result_name = attribution_file.replace('.npz','_modisco_results.h5') 
    
    cmd = ['modisco','motifs','-s',temp_onehot,'-a',temp_attr,'-n', '2000','-o',result_name]
    subprocess.run(cmd)  
    
    cleanup(onehot_file)
    cleanup(attribution_file)

if __name__ == "__main__":

    discover(sys.argv[1],sys.argv[2])
