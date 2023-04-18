import numpy as np
from scipy.stats import pearsonr
import torch
import re,os
import itertools

def metrics(a,b):

    frob_norm = np.linalg.norm(b-a,ord='fro') / np.linalg.norm(a,ord='fro')
    cos_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(a),torch.from_numpy(b),dim=0).numpy()
    a = a.ravel()
    b = b.ravel()
    corr = pearsonr(a,b)
    print(frob_norm,cos_sim.mean(axis=0),corr[0])

def compare(a,b):
    first = np.load(a)
    second = np.load(b)
    for tscript,grad in first.items():
        other_grad = second[tscript]
        metrics(grad,other_grad)

if __name__ == "__main__":

    sep = '____________________________'
    #for parent in ['coding','noncoding']:
    for parent in ['noncoding']:
        modelstring = f"PC.1.MDIG.(\d*)_steps.npz"
        print(sep)
        print(parent)
        print(sep)
        replicates = []
        for x in os.listdir(parent):
            match = re.search(modelstring,x)
            if match is not None:
                fname = os.path.join(parent,x)
                replicates.append((fname,int(match.group(1))))
        replicates.sort(key = lambda x: x[1])
        print(replicates)
        for a,b in itertools.combinations(replicates,2):
            print(f'steps : {a[1]} vs. {b[1]}')
            compare(a[0],b[0])

