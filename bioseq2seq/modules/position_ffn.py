"""Position feed-forward network from "Attention is All You Need"."""

import torch
import torch.nn as nn

class ComplexLinear(nn.Module):
    
    def __init__(self,n_blocks,d_per_block):
        super(ComplexLinear, self).__init__()
        
        self.weight = nn.Parameter(torch.randn(n_blocks,d_per_block,d_per_block,2)*0.02) 
        self.bias = nn.Parameter(torch.randn(n_blocks,1,d_per_block,2)*0.02) 

    def forward(self,x):
        x = torch.matmul(x,torch.view_as_complex(self.weight.contiguous()))
        x = x + torch.view_as_complex(self.bias.contiguous())
        return x


class ComplexBlockwiseMLP(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self,d_model, n_blocks, d_per_block,dropout=0.1):
        super(ComplexBlockwiseMLP, self).__init__()
       
        self.w_1 = ComplexLinear(n_blocks,d_per_block)
        self.w_2 = ComplexLinear(n_blocks,d_per_block)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        inputs = x 
        x = self.w_1(x)
        x = self.relu(torch.view_as_real(x))
        x = torch.view_as_complex(self.dropout_1(x).contiguous())
        x = torch.view_as_real(self.w_2(x))
        x = torch.view_as_complex(self.dropout_2(x).contiguous())
        return inputs + x

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout
