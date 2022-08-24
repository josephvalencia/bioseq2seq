import numpy as np

def parse(filename):

    saved = np.load(filename)
    count = 0
    for k,v in saved.items():
        corrected = v - v.mean(axis=1)[:,None]
        print(k,corrected.shape)
parse('output/bioseq2seq_1_val/all.PC.grad.rank_0.npz')
