""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        #print(f'tgt.shape ={tgt.shape}, start={tgt[:2,:]}, end ={tgt[-1]}')
        dec_in = tgt[:-1]  # exclude last target from inputs
        #print(f'tgt shape (<eos> removed) = {dec_in.shape}')
        #print(f'dec_in.shape={dec_in.shape}')
        enc_state, memory_bank, lengths, enc_self_attn = self.encoder(src, lengths)
        #print(memory_bank)
        #print(enc_state.shape,memory_bank.shape)
        is_clean =  lambda q: torch.all(~torch.isnan(q))
        
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, enc_dec_attn = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        if not self.training:
            src_cache = { "src" : src, "enc_states" : enc_state, "memory_bank" : memory_bank}
            return dec_out, enc_self_attn, enc_dec_attn, src_cache
        else:
            return dec_out, enc_self_attn, enc_dec_attn

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
