import torch
from functorch import grad,jacfwd, jacrev
import time

def affine(x,weight,bias):
    return torch.matmul(x,weight.T) + bias

def f(x,weight,bias):
    result = affine(x,weight,bias)
    return result,result

def weights_jacobian(f):
    return jacrev(f,argnums=1,has_aux=True)

def input_fisher(x,weight,bias):

    J_w,thetas = weights_jacobian(f)(x,W,b)
    print(f'J_w.T={J_w.permute(1,2,0).shape}')
    F_theta = torch.diag(1/thetas)
    inner = torch.matmul(J_w.permute(1,2,0),F_theta)
    print(f'inner={inner.shape}')
    print(f'J_w={J_w.shape}')
    F_wx = torch.matmul(inner, J_w)
    print(f'F_wx={F_wx.shape}')
    return F_wx

W = torch.randn(32,64,requires_grad=True)
b = torch.randn(32,requires_grad=True)
x = torch.rand(64,requires_grad=True)

F_wx = input_fisher(x,W,b)
