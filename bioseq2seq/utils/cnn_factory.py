"""
Implementation of "Convolutional Sequence to Sequence Learning"
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import bioseq2seq.modules
from torch import linalg as LA

SCALE_WEIGHT = 0.5 ** 0.5

def shape_transform(x):
    """ Tranform the size of the tensors to fit for conv input. """
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)


class GatedConv(nn.Module):
    """ Gated convolution for CNN class """

    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        
        '''
        self.conv = bioseq2seq.modules.WeightNormConv2d(
            input_size, 2 * input_size, kernel_size=(width, 1), stride=(1, 1),
            padding=(width // 2 * (1 - nopad), 0))
        ''' 
        self.conv = nn.Conv1d(
            input_size, 2 * input_size, kernel_size=(width, 1), stride=(1, 1),
            padding=(width // 2 * (1 - nopad), 0))

        #init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout))**0.5)
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.conv.weight)
        gain = (4 * (1 - dropout)/fan_in)**0.5
        init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout))**0.5)
        #init.normal_(self.conv.weight,mean=0.0,std=(4 * (1 - dropout)/fan_in)**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var):
        
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        is_clean =  lambda q: torch.all(~torch.isnan(q))
        #print(f'is_clean(x_var) = {is_clean(x_var)}')
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        #print(f'is_clean(out) = {is_clean(out)}')
        #print(f'is_clean(gate) = {is_clean(gate)}')
        old_out = torch.ravel(out)
        out = out * torch.sigmoid(gate)
        #print(f'is_clean(out) *after gate* = {is_clean(out)}')
        #print('gate',torch.sigmoid(gate).ravel()[:20])
        return out

class StackedCNN(nn.Module):
    """ Stacked CNN class """

    def __init__(self, num_layers, input_size, cnn_kernel_width=3,
                 dropout=0.2):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GatedConv(input_size, cnn_kernel_width, dropout))
        #self.batch_norm = nn.BatchNorm2d(input_size, eps=1e-6)
    
    def forward(self, x):
        is_clean =  lambda q: torch.all(~torch.isnan(q))
        for i,conv in enumerate(self.layers):
            
            #x = self.batch_norm(x)
            #x_norm = LA.norm(x.permute(0,3,1,2),ord='fro',dim=(2,3)) 
            #print(f'layer={i}, is_clean(x)={is_clean(x)}, shape(x)={x.shape}, norm(x,2)={x_norm}')
            #x = torch.fft.fft2(x.float(),dim=(1,2)).real.half()
            x = x + conv(x)
            x *= SCALE_WEIGHT
            #x = torch.fft.fft2(x,dim=(1,2)).real
            #x = self.batch_norm(x)
        return x
