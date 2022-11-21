import torch
import time

dim1 = 64
dim2 = 10000
A = torch.randn(dim1,dim2).cuda()
col = torch.randn(dim1).cuda()

s1 = time.time()
ans1 = torch.diag(col) @ A
e1 = time.time()

s2 = time.time()
ans2 = col.unsqueeze(1) * A 
e2 = time.time()

print(f'matmul time = {e1-s1}, broadcast time = {e2-s2}, equal = {torch.allclose(ans1,ans2)}')
