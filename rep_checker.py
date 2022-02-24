import numpy as np
import os
from scipy.stats import pearsonr, kendalltau 

def find_pair(prefix):
    
    translator_file = f'{prefix}_trainer_2.npz'
    training_file = f'{prefix}_trainer.npz'
    if os.path.exists(training_file) and os.path.exists(translator_file):
        a = np.load(translator_file)
        b = np.load(training_file)
        return a,b
    else:
        return None

special = ['XR_949580.2', 'XR_001748355.1', 'XR_001707416.2', 'XR_003029405.1', 'XR_922291.3', 'XM_015134081.2', 'XM_032910311.1', 'NM_001375259.1', 'XR_002007359.1', 'XR_003726903.1']

for p in special:
    pair = find_pair(p)
    if pair is not None:
        a,b = pair
        #vals_in_common = set(a.files+b.files)
        vals_in_common = ['src','enc_states','memory_bank']
        for v in list(vals_in_common):
            a_val = a[v]
            b_val = b[v]
            if a_val.shape[0] != b_val.shape[0]:
                shorter = min(a_val.shape[0],b_val.shape[0])
                a_val = a_val[:shorter,:]
                b_val = b_val[:shorter,:]
            if v == 'src':
                print(f'tscript :{p} , array : {v}, translator : {a_val.shape} , trainer :{b_val.shape}, is_close : {np.allclose(a_val,b_val)}\n')
            else:
                print(f'tscript :{p} , array : {v}, translator : {a_val.shape} , trainer :{b_val.shape}, is_close : {np.isclose(a_val,b_val)}\n')
                different = np.where(~np.isclose(a_val,b_val))
                d = 0
                for i in range(different[0].shape[0]):
                    if i < 10:
                        print('i = {}, j = {}'.format(different[0][i],different[1][i]))
                    d+=1
                corr = pearsonr(a_val.reshape(-1),b_val.reshape(-1)) 
                print(f'% different ={d/a_val.size}, corr = {corr}')
        print('_________________')
