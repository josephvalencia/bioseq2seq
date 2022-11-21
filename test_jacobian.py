import torch
from functorch import grad,jacfwd, jacrev
import time

def affine(x,weight,bias):
    return torch.matmul(x,weight)+bias 

def f(x,weight,bias):
    result = affine(x,weight,bias)
    return result,result

def weights_jacobian(f):
    return jacrev(f,argnums=1,has_aux=True)

def input_fisher(x,weight,bias):

    J_w,thetas = weights_jacobian(f)(x,W,b)
    multiple = (1/thetas).reshape(-1,1,1) * J_w
    multiple = multiple.reshape(2,-1)
    J_w = J_w.permute(1,2,0).reshape(-1,2)
    F_wt = torch.matmul(J_w,multiple)
    F_wt = F_wt.reshape(4,2,4,2)

    '''    
    F_theta = torch.diag(1/thetas)
    inner = torch.matmul(J_w.permute(1,2,0),F_theta)
    inner = inner.reshape(-1,4)
    final_J_w = J_w.permute(0,1,2).reshape(4,-1)
    F_wx = torch.matmul(inner, final_J_w)
    F_wx = F_wx.reshape(64,4,64,4)
    '''
    return F_wt,F_wt

def uncertainty(x,W,b,Z):

    F_wt,alt = input_fisher(x,W,b)
    
    vectorized = F_wt.reshape(-1)
    alt_vectorized = Z.reshape(-1)
    unc = torch.dot(vectorized,alt_vectorized)
    return unc,unc

W = torch.randn(4,2,requires_grad=True)
b = torch.randn(2,requires_grad=True)
x = torch.rand(4,requires_grad=True)
Z = 10*torch.rand(4,2,4,2)

dJ_fw = jacfwd(jacrev(affine,argnums=0),argnums=1)(x,W,b)
#grads = jacrev(affine,argnums=0)(x,W,b)

J_w,thetas = weights_jacobian(f)(x,W,b)
'''
print(f'W = {W}')
print(f'x = {x}')
print(f'thetas = {thetas.shape}')
print(f'J_w = {J_w.shape}')
'''
MAX_ITER = 5000
lr = 1e-5

for i in range(MAX_ITER):
    print(i)
    d_unc,unc = grad(uncertainty,has_aux=True)(x,W,b,Z) 
    x = x-lr*d_unc
    print(f'iter={i}, uncertainty = {unc}, {x.shape}')

#dF_wx, F_wt = jacfwd(input_fisher,argnums=0,has_aux=True)(x,W,b)
