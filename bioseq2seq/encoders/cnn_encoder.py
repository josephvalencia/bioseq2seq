"""
Implementation of "Convolutional Sequence to Sequence Learning"
"""
import torch.nn as nn

from bioseq2seq.encoders.encoder import EncoderBase
from bioseq2seq.utils.cnn_factory import shape_transform, StackedCNN

SCALE_WEIGHT = 0.5 ** 0.5

class CNNEncoder(EncoderBase):
    """Encoder based on "Convolutional Sequence to Sequence Learning"
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """

    def __init__(self, num_layers, hidden_size,
                 cnn_kernel_width, dropout, embeddings,dilation_factor=1):
        super(CNNEncoder, self).__init__()

        self.embeddings = embeddings
        input_size = embeddings.embedding_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size,
                              cnn_kernel_width,dropout,dilation_factor=dilation_factor)


    def forward(self, input, lengths=None, hidden=None,grad_mode=False):
        """See :class:`bioseq2seq.modules.EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        # s_len, batch, emb_dim = emb.size()
        emb = self.embeddings(input,grad_mode=grad_mode)
        if grad_mode:
            emb = emb.contiguous()
        else:
            emb = emb.transpose(0, 1).contiguous()
        
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        #return emb_remap.squeeze(3).permute(2,0,1).contiguous(), \
        #    out.squeeze(3).permute(2,0,1).contiguous(), lengths, None
        return emb_remap.squeeze(3).transpose(0, 1).contiguous(), \
            out.squeeze(3).transpose(0, 1).contiguous(), lengths, None

    def update_dropout(self, dropout):
        self.cnn.dropout.p = dropout
