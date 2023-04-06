import torch
import numpy as np
import sys,os,re
from experiments.analysis.utils import grad_simplex_correction

def pad_to_max_len(array,max_len):

    diff = max_len - array.shape[1]
    zeros = np.zeros((4,diff),dtype=array.dtype)
    padded = np.concatenate([array,zeros],axis=1)
    return padded

def make_consensus(parent,attr,model_prefix,class_type,target_pos):

    modelstring = f"{model_prefix}_(.*).step_(\d*)"
    results =  "{}.{}.{}.npz".format(class_type,target_pos,attr)
    replicates = [os.path.join(os.path.join(parent,x),results) for x in os.listdir(parent) if re.search(modelstring,x)]
    replicates = [x for x in replicates if os.path.isfile(x)]
    print(replicates) 
    quit() 
    saved = [np.load(x) for x in replicates]
    seq_file = "{}.{}.{}.npz".format(class_type,target_pos,'onehot')
    
    sequences = np.load('experiments/output/bioseq2seq_4_Jun25_07-51-41_step_10500/PC.1.onehot.npz') 
    driver = saved.pop()
    consensus = {}
    all_attr = []
    all_seq = []
    max_len = 1200
    
    for tscript,grad in driver.items():
        all_grads = [grad]
        onehot = sequences[tscript]
        for rep in saved:
            all_grads.append(rep[tscript])
        total = np.stack(all_grads,axis=0)
        mean = total.mean(axis=0)
        std = total.std(axis=0)
        coeff_variation = std / np.abs(mean)
        coeff_variation = coeff_variation[~np.isnan(coeff_variation)]
        consensus[tscript] = mean
        padded_attr = pad_to_max_len(mean[:,2:6].T,1200)
        padded_seq = pad_to_max_len(onehot[:,2:6].T,1200)
        all_attr.append(padded_attr)
        all_seq.append(padded_seq)
        big_var = np.count_nonzero(coeff_variation > 0.25)
        #print(f'tscript = {tscript}, mean(consensus) = {mean.shape}, large variance = {big_var}/{coeff_variation.size}') 

    modisco_attr = np.stack(all_attr)
    modisco_seq = np.stack(all_seq)
    print(f'attr = {modisco_attr.shape} , seq = {modisco_seq.shape}')
    consensus_file = "{}/{}.CONSENSUS.{}.{}.{}.npz".format(parent,model_prefix,class_type,target_pos,attr)
    modisco_attr_file = "{}/{}.CONSENSUS.{}.{}.{}.npz".format(parent,model_prefix,class_type,target_pos,attr)
    modisco_seq_file = "{}/{}.CONSENSUS.{}.{}.{}.npz".format(parent,model_prefix,class_type,target_pos,'onehot')
    np.savez_compressed(consensus_file,**consensus)
    np.savez_compressed(modisco_attr_file,modisco_attr)
    np.savez_compressed(modisco_seq_file,modisco_seq)

if __name__ == "__main__":
    
    make_consensus(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
