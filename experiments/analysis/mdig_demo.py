import torch
import numpy as np
import sys
from utils import parse_config,add_file_list,load_CDS
from scipy.spatial.distance import cosine
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def similarity_scores(a,b):

    frob = np.linalg.norm(a-b,ord='fro')
    cos_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(a),torch.from_numpy(b),dim=0).numpy()
    print(cos_sim.shape) 
    mean_cos_sim = cos_sim.mean()
    a = a.ravel()
    b = b.ravel()
    
    #cos_dist = cosine(a,b)
    #cos_sim = (cos_dist - 1) / -1
    corr = pearsonr(a,b)
    print(f'Pearson corr = {corr[0]:.3f} (p={corr[1]:.3f}), mean position-wise cosine sim. = {mean_cos_sim:.3f}, Frobenius dist. = {frob:.3f}')

def compare(prefix):

    embed_file = np.load(f'{prefix}.MDIG-embed.npz') 
    onehot_file = np.load(f'{prefix}.MDIG.npz') 
    ism_file = np.load(f'{prefix}.ISM.npz') 
    test_file = '/home/bb/valejose/home/bioseq2seq/data/mammalian_200-1200_test_nonredundant_80.csv'
    test_cds = load_CDS(test_file)
    
    for tscript,mdig in onehot_file.items():

        cds_loc = [int(x) for x in test_cds[tscript].split(':')] 
        s,e = tuple(cds_loc) 
        #mdig = mdig[:,2:6].T
        sep = '_________________________________________'
        print(tscript)
        print('Full transcript MDIG (onehot) vs MDIG (embedding)') 
        similarity_scores(mdig,embed_file[tscript])
        labels = ['A','C','G','T']
        if s>=12 and s+60 < mdig.shape[1]:
            cds_mdig = mdig[:,s-12:s+60]
            domain = list(range(-12,60))
            ism = ism_file[tscript]
            print(f'Start codon -12 to +60 MDIG (idx={s-12}:{s+60}) vs ISM') 
            similarity_scores(-cds_mdig,ism.T) 
        print(sep)

if __name__ == "__main__":
    
    compare(sys.argv[1])
