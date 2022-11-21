import torch
import time

''' 11/08/2022. Pseudocode sketch of how to form diagonal + low rank estimate of inverse Fisher information 
    matrix and compute natural gradient descent direction. Currently handles up to 24 samples at typical
    sequence lengths for bioseq2seq. NGD time overhead is equivalent to ~1 sample worth, appears not to
    alter asymptotic time complexity because input gradient calculation dominates. Memory usage increases
    quadratically with params due to cost of instantiating FIM.
'''

N_SAMPLES = 24
# obtain oracle grads wrt. input (via STE or Gumbel-SM) and pack vocab and sequence dimensions into one
input_grads = calc_input_grads(likelihood,onehot_src)
G = input_grads.reshape(N_SAMPLES,-1)
# diagonalize with economy SVD
U,S,Vh = torch.linalg.svd(G,full_matrices=False)
# square to convert G singular vals to G^TG eigvals, then invert 
S_inv = torch.diag(S.pow(-2)) 

# placeholder for closed-form diagonal Fisher of independent categoricals
diag_inv_col = G.new_ones(G.shape[1],1)

# broadcast to avoid matmul with diagonal, ~order of magnitude faster
V_scaled = diag_inv_col * Vh.T
# only an N_SAMPLES x N_SAMPLES  matrix to invert
inner_inv = torch.linalg.inv(S_inv + Vh @ V_scaled) 
# Woodbury update of diagonal inverse with low rank term
woodbury_inv_fisher = torch.diag(diag_inv_col.squeeze()) - V_scaled @ inner_inv @ V_scaled.T 
# TODO: add special case for rank-1 updates with Sherman-Morrison, no SVD needed

# use averaged grads as descent direction 
avg_grad = G.sum(dim=0) / G.shape[0]
# precondition to obtain NGD step
natural_grad = woodbury_inv_fisher @ avg_grad
# return to proper input shape
natural_grad = natural_grad.reshape(*input_grads.shape)
