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

    def __init__(self, input_size, width=3, dropout=0.2, nopad=False,dilation=1):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(
            input_size, 2 * input_size, kernel_size=(width, 1), stride=(1, 1),
            dilation=dilation,padding='same')
        '''
        self.conv = bioseq2seq.modules.WeightNormConv2d(
            input_size, 2 * input_size, kernel_size=(width, 1), stride=(1, 1),dilation=dilation,
            padding='same')
        ''' 
        padding = dilation * (width //2) * (1-nopad)
        
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.conv.weight)
        gain = (4 * (1 - dropout)/fan_in)**0.5
        
        init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout))**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var):
        
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        old_out = torch.ravel(out)
        out = out * torch.sigmoid(gate)
        return out

class StackedCNN(nn.Module):
    """ Stacked CNN class """

    def __init__(self, num_layers, input_size, cnn_kernel_width=3,
                 dropout=0.2,dilation_factor=1):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        dilation = 1
        for _ in range(num_layers):
            self.layers.append(
                GatedConv(input_size, cnn_kernel_width,dropout,dilation=dilation))
            dilation *= dilation_factor
    
    def forward(self, x):
        is_clean =  lambda q: torch.all(~torch.isnan(q))
        
        for i,conv in enumerate(self.layers):
            y = conv(x)
            x = x + y
            x *= SCALE_WEIGHT
        return x
