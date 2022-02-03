"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
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

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout):
        
        super(GlobalFilterEncoderLayer, self).__init__()
        N_FREQ_BINS = 100
        filter_size = math.floor(d_model/2)+1
        self.global_filter =  nn.Parameter(torch.randn(filter_size,2,N_FREQ_BINS)*0.02) 
        self.input_norm = nn.LayerNorm(d_model, eps=1e-12)
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
        batch,seq_len,model_dim = inputs.shape
        x = self.input_norm(inputs)
        freq_domain =  torch.fft.rfft2(x,dim=axes,norm='ortho')
        upscaled_filter = F.interpolate(self.global_filter,freq_domain.shape[1],mode='linear',align_corners=True).permute(2,0,1)
        W_prime = upscaled_filter.permute(1,2,0) 
        x = freq_domain * torch.view_as_complex(upscaled_filter.contiguous())
        mixed_tokens =  torch.fft.irfft2(x,dim=axes,s=(seq_len,model_dim),norm='ortho')
        x = self.feed_forward_norm(mixed_tokens)
        ff_output = self.feed_forward(x) 
        return ff_output+inputs

    def update_dropout(self, dropout):
        self.feed_forward.update_dropout(dropout)

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

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings,global_filter=False):
        super(FNetEncoder, self).__init__()

        self.embeddings = embeddings
        
        if global_filter:
            self.fnet = nn.ModuleList(
                [GlobalFilterEncoderLayer(
                    d_model, heads, d_ff, dropout, attention_dropout)
                 for i in range(num_layers)])
        else:
            self.fnet = nn.ModuleList(
                [FNetEncoderLayer(
                    d_model, heads, d_ff, dropout, attention_dropout)
                 for i in range(num_layers)])

    def forward(self, src, lengths=None, attn_debug = True,grad_mode=False):
        """See :func:`EncoderBase.forward()`"""
        
        emb = self.embeddings(src,grad_mode=grad_mode)
        if grad_mode:
            out = emb.contiguous()
        else:
            out = emb.transpose(0,1).contiguous()

        mask = ~sequence_mask(lengths).unsqueeze(1)

        # Run the forward pass of every layer of the tranformer.
        for i,layer in enumerate(self.fnet):
            out = layer(out,mask)

        self_attns = None
        return emb, out.transpose(0, 1).contiguous(), lengths , self_attns

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.fnet:
            layer.update_dropout(dropout, attention_dropout)
