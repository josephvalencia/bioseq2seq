import torch.nn as nn
import torch
import torchaudio
import math
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from bioseq2seq.encoders.encoder import EncoderBase
from bioseq2seq.modules import MultiHeadedAttention
from bioseq2seq.modules.position_ffn import PositionwiseFeedForward, ComplexBlockwiseMLP
from bioseq2seq.utils.misc import sequence_mask
from collections import defaultdict

class FourierEncoderLayer(nn.Module):
    """
    Base class for Fourier token mixing encoder.
 
    Args:
        d_model (int): the model dimension, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self,d_model,d_ff,dropout,lambd_L1=0.5):

        super(FourierEncoderLayer, self).__init__()
        self.input_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.softshrink = nn.Softshrink(lambd=lambd_L1) 

    def forward(self,inputs,mask):
        
        x = self.input_norm(inputs)
        # ensure sequence is zero-padded to interpolate in frequency domain
        is_pad = torch.where(mask,torch.ones_like(mask,dtype=torch.long),0)
        x = x.masked_fill(mask,0.0)
        mixed_tokens, enc_cache = self.mix_tokens(x) 
        x = self.feed_forward_norm(mixed_tokens)
        ff_output = self.feed_forward(x) 
        return inputs+self.dropout(ff_output).masked_fill(mask,0.0), enc_cache

    def mix_tokens(self,x):
        raise NotImplementedError

    def update_dropout(self, dropout):
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

    def complex_softshrink(self,x):
        return torch.view_as_complex(self.softshrink(torch.view_as_real(x)))

class GlobalFilterEncoderLayer(FourierEncoderLayer):

    """
    Based on (Rao et. al, 2021) https://proceedings.neurips.cc/paper/2021/file/07e87c2f4fc7f7c96116d8e2a92790f5-Paper.pdf
    """
    def __init__(self, d_model, d_ff, dropout,d_spatial,two_dim=True,share_hidden_weights=False,lambd_L1=0.5):
        
        super(GlobalFilterEncoderLayer, self).__init__(d_model,d_ff,dropout,lambd_L1=lambd_L1)
        
        self.two_dim = two_dim

        # 2D global conv requires weight for each embedding dim, optional in 1D case
        if two_dim or not share_hidden_weights:
            filter_hidden_size = math.floor(d_model/2)+1
        else:
            filter_hidden_size = 1 
        
        # for compatibility with F.interpolate use old format for complex num
        # last dim of size 2 for real and imaginary parts
        self.global_filter =  nn.Parameter(torch.randn(filter_hidden_size,2,d_spatial)*0.02) 
    
    def mix_tokens(self,x):

        # save input dims 
        batch_size,seq_len,model_dim = x.shape 
        
        # FFT
        if self.two_dim:
            freq_domain = torch.fft.rfft2(x,dim=(1,2),norm='ortho')
        else:
            freq_domain = torch.fft.rfft(x,dim=1,norm='ortho')
        
        batch_freq,len_freq,dim_freq = freq_domain.shape 
        # elementwise multiplication by resampled filter 
        upscaled_filter = F.interpolate(self.global_filter,len_freq,mode='linear',align_corners=True).permute(2,0,1)
        x = freq_domain * torch.view_as_complex(upscaled_filter.contiguous())
       
        # promote sparsity
        x = self.complex_softshrink(x)
        
        enc_cache = {'mod_freq' : x.detach().cpu().abs() , 'original_freq': freq_domain.detach().cpu().abs()}

        # inverse FFT 
        if self.two_dim:
            mixed_tokens = torch.fft.irfft2(x,dim=(1,2),s=(seq_len,model_dim),norm='ortho')
        else:
            mixed_tokens = torch.fft.irfft(x,dim=1,n=seq_len,norm='ortho')
        
        return mixed_tokens, enc_cache 

class LocalFilterEncoderLayer(FourierEncoderLayer):
    """
    Modification of Global Filter Network for Short-Time Fourier Transform
    """

    def __init__(self, d_model, d_ff, dropout,window_size=300,share_hidden_weights=False,lambd_L1=0.5):
        
        super(LocalFilterEncoderLayer, self).__init__(d_model,d_ff,dropout,lambd_L1=lambd_L1)
        
        spatial_size = math.floor(window_size/2)+1
        filter_hidden_size = 1 if share_hidden_weights else d_model

        # global filter is complex
        self.global_filter =  nn.Parameter(torch.randn(spatial_size,filter_hidden_size,dtype=torch.cfloat)*0.02) 
        self.stft = torchaudio.transforms.Spectrogram(n_fft=window_size,power=None,pad_mode='constant')
        self.inv_stft = torchaudio.transforms.InverseSpectrogram(n_fft=window_size)

    def mix_tokens(self,x):

        # save input dims 
        batch_size,seq_len,model_dim = x.shape 
        # STFT
        spect = self.stft(x.transpose(1,2))
        enc_cache = dict()
        # pack STFT windows into batch dim
        spect = spect.permute(0,3,2,1)
        s_0,s_1,s_2,s_3 = spect.shape
        spect = spect.reshape(-1,s_2,s_3) 
        # elementwise multiplication
        x = spect * self.global_filter
        # restore shape
        x = x.reshape(s_0,s_1,s_2,s_3)
        x = self.complex_softshrink(x)
        x = x.permute(0,3,2,1)
        enc_cache['mod_freq'] = x.detach().abs().pow(2)[:,31,:,:] #.sum(dim=1) 

        # inverse FFT 
        mixed_tokens = self.inv_stft(x,seq_len).transpose(1,2)
        enc_cache['mod_space'] = mixed_tokens.detach()
        return mixed_tokens, enc_cache

class AFNOEncoderLayer(FourierEncoderLayer):
    """
    Based on (Guibas et. al, 2021) https://arxiv.org/abs/2111.13587
    """
    def __init__(self, d_model, d_ff, dropout, n_blocks, sparsity):
        
        super(AFNOEncoderLayer, self).__init__(d_model,d_ff,dropout)
        
        self.input_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.n_blocks = n_blocks
        self.d_per_block = d_model // n_blocks
        self.mlp = ComplexBlockwiseMLP(d_model,self.n_blocks,self.d_per_block,dropout)
        self.sparsity = sparsity

    def soft_shrink(self,x):
       
        a = torch.sgn(x)
        b = torch.abs(x)- self.sparsity
        c = torch.zeros_like(x,dtype=float)
        mask = b < c
        x = x.masked_fill(mask,0.0+0.0j) 
        return a * x
    
    def mix_tokens(self,x):

        # save input dims 
        batch_size,seq_len,model_dim = x.shape 
        # FFT
        freq_domain = torch.fft.rfft(x,dim=1,norm='ortho')
        b,l,d = freq_domain.shape
        freq_domain = freq_domain.reshape(b,l,self.n_blocks,1,self.d_per_block)
        # two-layer perceptron
        x = self.mlp(freq_domain)
        x = x.reshape(b,l,d)
        x = self.complex_softshrink(x) 
        # inverse FFT 
        mixed_tokens = torch.fft.irfft(x,dim=1,n=seq_len,norm='ortho')
        return mixed_tokens

class FNetEncoderLayer(FourierEncoderLayer):
    """
    Based on (Lee-Thorp et al 2021) https://arxiv.org/abs/2105.03824
    """

    def __init__(self, d_model,d_ff, dropout):
        super(FNetEncoderLayer, self).__init__(d_model,d_ff,dropout)

    def mix_tokens(self, inputs):
        
        axes = (1,2)
        mixed_tokens = torch.fft.fft2(inputs,dim=axes).real
        return mixed_tokens

class FourierEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    Args:
        model_template_fn (function): encoder stack consists of idential layers returned by a function
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self,model_template_fn,d_model,num_layers,embeddings):
        super(FourierEncoder, self).__init__()

        self.embeddings = embeddings
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.fnet = nn.ModuleList([model_template_fn(i) for i in range(num_layers)])

    def forward(self, src, lengths=None, attn_debug = True,grad_mode=False):
        """See :func:`EncoderBase.forward()`"""
        
        emb = self.embeddings(src,grad_mode=grad_mode)
        if grad_mode:
            out = emb.contiguous()
        else:
            out = emb.transpose(0,1).contiguous()

        mask = ~sequence_mask(lengths).unsqueeze(-1)
        
        freq_data = []
        space_data = []
        # Run the forward pass of every layer 
        for i,layer in enumerate(self.fnet):
            out, layer_cache = layer(out,mask)
            #  save all
            freq_data.append(layer_cache['mod_freq'])
            space_data.append(layer_cache['mod_space'])
       
        mod_freq = torch.stack(freq_data,dim=1)
        mod_space = torch.stack(space_data,dim=1)
        all_cache = {'mod_freq' : mod_freq , 'mod_space' : mod_space}

        out = self.layer_norm(out)
        return emb, out.transpose(0, 1).contiguous(), lengths , all_cache

    def update_dropout(self, dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.fnet:
            layer.update_dropout(dropout)
