"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from bioseq2seq.encoders.encoder import EncoderBase
from bioseq2seq.modules import MultiHeadedAttention
from bioseq2seq.modules.position_ffn import PositionwiseFeedForward
from bioseq2seq.utils.misc import sequence_mask

class GlobalFilterEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
 
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_spatial, d_ff, dropout, attention_dropout):
        
        super(GlobalFilterEncoderLayer, self).__init__()
        
        filter_hidden_size = math.floor(d_model/2)+1
        # global filter is complex, so dimension of size 2 holds real and imaginary parts
        self.global_filter =  nn.Parameter(torch.randn(filter_hidden_size,2,d_spatial)*0.02) 
        self.input_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs,mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        x = self.input_norm(inputs)
        mixed_tokens = self.mix_interpolated_filter(x,mask) 
        #mixed_tokens = self.mix_decimate_frequencies(x,mask) 
        x = self.feed_forward_norm(mixed_tokens)
        ff_output = self.feed_forward(x) 
        return inputs+self.dropout(ff_output).masked_fill(mask,0.0)

    def mix_decimate_frequencies(self,x,mask):

        axes = (1,2)
        # pad with trailing zeros
        is_pad = torch.where(mask,torch.ones_like(mask,dtype=torch.long),0)
        x = x.masked_fill(mask,0.0)
        num_padding_tokens = torch.count_nonzero(is_pad,dim=1)
        batch_size,seq_len,model_dim = x.shape 
        # FFT
        freq_domain =  torch.fft.rfft2(x,dim=axes,norm='ortho')
        batch_freq,len_freq,dim_freq = freq_domain.shape 
        hidden,_,spatial = self.global_filter.shape 
        # downsample frequency domain to match filter 
        freq_domain = torch.view_as_real(freq_domain).reshape(batch_freq,len_freq,2*dim_freq)
        freq_domain = F.interpolate(freq_domain.permute(0,2,1),spatial,mode='linear',align_corners=True).permute(0,2,1)
        freq_domain = freq_domain.reshape(batch_freq,spatial,dim_freq,2)
        # elementwise multplication 
        upscaled_filter = self.global_filter.permute(2,0,1) 
        x = torch.view_as_complex(freq_domain.contiguous()) * torch.view_as_complex(upscaled_filter.contiguous())
        # interpolate result back to input size
        x = torch.view_as_real(x).reshape(batch_freq,spatial,dim_freq*2)
        x = F.interpolate(x.permute(0,2,1),len_freq,mode='linear',align_corners=True).permute(0,2,1) 
        x = x.reshape(batch_freq,len_freq,dim_freq,2)
        x = torch.view_as_complex(x.contiguous())
        # inverse FFT
        mixed_tokens =  torch.fft.irfft2(x,dim=axes,s=(seq_len,model_dim),norm='ortho')
        return mixed_tokens

    def mix_interpolated_filter(self,x,mask):

        axes = (1,2)
        # pad with trailing zeros
        is_pad = torch.where(mask,torch.ones_like(mask,dtype=torch.long),0)
        x = x.masked_fill(mask,0.0)
        num_padding_tokens = torch.count_nonzero(is_pad,dim=1)
        batch_size,seq_len,model_dim = x.shape 
        # FFT
        freq_domain =  torch.fft.rfft2(x,dim=axes,norm='ortho')
        batch_freq,len_freq,dim_freq = freq_domain.shape 
        # elementwise multiplication by resampled filter 
        upscaled_filter = F.interpolate(self.global_filter,len_freq,mode='linear',align_corners=True).permute(2,0,1)
        x = freq_domain * torch.view_as_complex(upscaled_filter.contiguous())
        #complex_weight = torch.view_as_complex(upscaled_filter.contiguous())
        #complex_weight = complex_weight[:,0].reshape(1,-1,1)
        #x = freq_domain * complex_weight
        #print(freq_domain.shape,complex_weight.shape)
        # inverse FFT 
        mixed_tokens =  torch.fft.irfft2(x,dim=axes,s=(seq_len,model_dim),norm='ortho')
        return mixed_tokens

    def update_dropout(self, dropout):
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

class FNetEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
 
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout):
        super(FNetEncoderLayer, self).__init__()

        self.fourier_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, inputs,mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        axes = (1,2)
        mixed_tokens = torch.fft.fft2(inputs,dim=axes).real
        x = self.fourier_norm(mixed_tokens + inputs)
        ff_output = self.feed_forward(x) 
        return self.feed_forward_norm(ff_output+x)

    def update_dropout(self, dropout):
        self.feed_forward.update_dropout(dropout)

class FNetEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, d_filter, d_ff, dropout,
                 attention_dropout, embeddings,global_filter=False):
        super(FNetEncoder, self).__init__()

        self.embeddings = embeddings
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        
        if global_filter:
            self.fnet = nn.ModuleList(
                [GlobalFilterEncoderLayer(
                    d_model, d_filter, d_ff, dropout, attention_dropout)
                 for i in range(num_layers)])
        else:
            self.fnet = nn.ModuleList(
                [FNetEncoderLayer(
                    d_model, d_filter, d_ff, dropout, attention_dropout)
                 for i in range(num_layers)])

    def forward(self, src, lengths=None, attn_debug = True,grad_mode=False):
        """See :func:`EncoderBase.forward()`"""
        
        emb = self.embeddings(src,grad_mode=grad_mode)
        if grad_mode:
            out = emb.contiguous()
        else:
            out = emb.transpose(0,1).contiguous()

        mask = ~sequence_mask(lengths).unsqueeze(-1)

        # Run the forward pass of every layer of the tranformer.
        for i,layer in enumerate(self.fnet):
            out = layer(out,mask)
        
        out = self.layer_norm(out)
        self_attns = None
        return emb, out.transpose(0, 1).contiguous(), lengths , self_attns

    def update_dropout(self, dropout, attention_dropout):
        print(f'updating Encoder dropout ={dropout}, {attention_dropout}')
        self.embeddings.update_dropout(dropout)
        for layer in self.fnet:
            layer.update_dropout(dropout, attention_dropout)
