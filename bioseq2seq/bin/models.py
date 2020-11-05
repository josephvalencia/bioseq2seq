import torch.nn as nn
import torch.nn.functional as F

from bioseq2seq.models import NMTModel
from bioseq2seq.encoders import TransformerEncoder
from bioseq2seq.decoders import TransformerDecoder
from bioseq2seq.modules import Embeddings

class Generator(nn.Module):
    '''Fully connected + log-softmax over target vocab'''

    def __init__(self, d_model, vocab):

        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self,x):

        logits = F.log_softmax(self.proj(x), dim=-1)
        return logits

class Classifier(nn.Module):

    def __init__(self,encoder,generator):

        super(Classifier,self).__init__()
        self.encoder = encoder
        self.generator = generator

    def forward(self,src,lengths):
        
        enc_state, memory_bank, lengths, enc_self_attn = self.encoder(src,lengths)
        pooled = memory_bank.mean(dim=0)
        
        logits = self.generator(pooled)
        return logits

def make_transformer_seq2seq(n_enc=4,n_dec=4,model_dim=128,dim_ff=2048, heads=8, dropout=0.1,max_rel_pos=10):

    '''construct Transformer encoder-decoder from hyperparameters'''

    attention_dropout = 0.1
    NUM_INPUT_CLASSES = 19
    NUM_OUTPUT_CLASSES = 29

    nucleotide_embeddings = Embeddings(word_vec_size = model_dim,
                                       word_vocab_size = NUM_INPUT_CLASSES,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = model_dim,
                                    word_vocab_size = NUM_OUTPUT_CLASSES,
                                    word_padding_idx = 1,
                                    position_encoding = True)

    encoder_stack = TransformerEncoder(num_layers = n_enc,
                                       d_model = model_dim,
                                       heads = heads,
                                       d_ff = dim_ff,
                                       dropout = dropout,
                                       embeddings = nucleotide_embeddings,
                                       max_relative_positions = max_rel_pos,
                                       attention_dropout = attention_dropout)

    decoder_stack = TransformerDecoder(num_layers = n_dec,
                                       d_model = model_dim,
                                       heads = heads,
                                       d_ff = dim_ff,
                                       dropout = dropout,
                                       embeddings = protein_embeddings,
                                       self_attn_type = 'scaled-dot',
                                       copy_attn = False,
                                       max_relative_positions = max_rel_pos,
                                       aan_useffn = False,
                                       attention_dropout = attention_dropout,
                                       full_context_alignment = False,
                                       alignment_heads = None,
                                       alignment_layer = None)

    generator = Generator(model_dim,NUM_OUTPUT_CLASSES)

    model = NMTModel(encoder_stack,decoder_stack)
    model.generator = generator

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def make_transformer_classifier(n_enc=4,model_dim=128,dim_ff=2048, heads=8, dropout=0.1,max_rel_pos=10):

    NUM_INPUT_CLASSES = 19
    NUM_OUTPUT_CLASSES = 29

    nucleotide_embeddings = Embeddings(word_vec_size = model_dim,
                                       word_vocab_size = NUM_INPUT_CLASSES,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    encoder_stack = TransformerEncoder(num_layers = n_enc,
                                    d_model = model_dim,
                                    heads = heads,
                                    d_ff = dim_ff,
                                    dropout = dropout,
                                    embeddings = nucleotide_embeddings,
                                    max_relative_positions = max_rel_pos,
                                    attention_dropout = dropout)

    generator = Generator(model_dim,NUM_OUTPUT_CLASSES)
    model = Classifier(encoder_stack,generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
