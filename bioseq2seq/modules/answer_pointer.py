import torch.nn as nn
import torch.nn.functional as F
import torch
from bioseq2seq.decoders.decoder import DecoderBase

class PointerGenerator(nn.Module):
    '''Simple attention output over encoder inputs'''
    def __init__(self,hidden_size):
        super(PointerGenerator,self).__init__()
        self.encoder_proj = nn.Linear(hidden_size,1,bias=False)

    def forward(self,embeds):
        # dot product between embeds and encoder_proj params 
        similarity = self.encoder_proj(embeds)
        attn = F.log_softmax(similarity,dim=0).squeeze(-1)
        return attn

class PadDecoder(DecoderBase):
    '''Right-pad for NULL index, just to comply with NMTModel interface'''
    def __init__(self):
        super(PadDecoder,self).__init__(attentional=False)
        self.state = None

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        pass

    def forward(self, tgt, memory_bank, step=None,grad_mode=False, **kwargs): 
        ''' memory_bank : (seq_len, batch_size, hidden_size)'''
        # pad with zeros for out of range
        L,B,H = memory_bank.size()
        #padded_embeddings = torch.cat([memory_bank,memory_bank.new_zeros(1,B,H)],dim=0)
        #return padded_embeddings,None
        return memory_bank, None
