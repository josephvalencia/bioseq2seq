import pywt
import torch
from pywt import WaveletPacket2D
from itertools import product

wavelet = pywt.Wavelet('db1')
x = torch.randn(10, 30, 512)
wp = WaveletPacket2D(data=x,wavelet=wavelet,mode='reflect')
wp_keys = list(product(["a", "h", "v", "d"], repeat=1))
np_lst = []
for key in wp_keys:
    item = torch.tensor(wp["".join(key)].data)
    print(key,item.shape)
    np_lst.append(item)
viz = torch.cat([torch.cat([np_lst[0], np_lst[1]], -2),torch.cat([np_lst[2], np_lst[3]], -2)], -1)
print(viz.shape)
