import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from bioseq2seq.models import NMTModel
from bioseq2seq.encoders import TransformerEncoder, CNNEncoder, FNetEncoder
from bioseq2seq.decoders import TransformerDecoder, CNNDecoder
from bioseq2seq.modules import Embeddings

class Generator(nn.Module):
    '''Fully connected + log-softmax over target vocab'''

    def __init__(self, d_model, vocab):

        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self,x,softmax=True):

        linear = self.proj(x)
        if softmax:
            return F.log_softmax(linear,dim=-1)
        else:
            return linear

def make_cnn_seq2seq(n_input_classes,n_output_classes,n_enc=4,n_dec=4,model_dim=128,dropout=0.1):

    '''construct Transformer encoder-decoder from hyperparameters'''

    attention_dropout = 0.1

    nucleotide_embeddings = Embeddings(word_vec_size = model_dim,
                                       word_vocab_size = n_input_classes,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = model_dim,
                                    word_vocab_size = n_output_classes,
                                    word_padding_idx = 1,
                                    position_encoding = True)

    encoder_kernel_size = 3
    decoder_kernel_size = 3 

    encoder_stack = CNNEncoder(num_layers = n_enc,
                                       hidden_size = model_dim,
                                       cnn_kernel_width = encoder_kernel_size,
                                       dropout = dropout,
                                       embeddings = nucleotide_embeddings)
    
    decoder_stack = CNNDecoder(num_layers = n_dec,
                                       hidden_size = model_dim,
                                       cnn_kernel_width = decoder_kernel_size,
                                       dropout = dropout,
                                       embeddings = protein_embeddings)

    generator = Generator(model_dim,n_output_classes)
    model = NMTModel(encoder_stack,decoder_stack)
    model.generator = generator

    model.apply(cnn_init_weights)

    return model

def cnn_init_weights(m):
    
    if isinstance(m,nn.Embedding):
        init.normal_(m.weight,mean=0.0,std=0.1)
    elif isinstance(m,nn.modules.linear.Linear):
        f_in,f_out = init._calculate_fan_in_and_fan_out(m.weight)
        init.normal_(m.weight,mean=0.0,std=(1/f_in)**0.5)
    else:
        init.xavier_uniform_(m)

def make_transformer_seq2seq(n_input_classes,n_output_classes,n_enc=4,n_dec=4,model_dim=128,dim_ff=2048,heads=8, dropout=0.1,max_rel_pos=8):

    '''construct Transformer encoder-decoder from hyperparameters'''

    attention_dropout = 0.1

    nucleotide_embeddings = Embeddings(word_vec_size = model_dim,
                                       word_vocab_size = n_input_classes,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = model_dim,
                                    word_vocab_size = n_output_classes,
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

    generator = Generator(model_dim,n_output_classes)
    model = NMTModel(encoder_stack,decoder_stack)
    model.generator = generator

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def make_hybrid_seq2seq(n_input_classes,n_output_classes,n_enc=4,n_dec=4,model_dim=128,dim_ff=2048,dim_filter=100,heads=8,dropout=0.1,max_rel_pos=8):

    '''construct Transformer encoder-decoder from hyperparameters'''

    attention_dropout = 0.1

    nucleotide_embeddings = Embeddings(word_vec_size = model_dim,
                                       word_vocab_size = n_input_classes,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = model_dim,
                                    word_vocab_size = n_output_classes,
                                    word_padding_idx = 1,
                                    position_encoding = True)
    
    encoder_stack = FNetEncoder(num_layers = n_enc,
                                       d_model = model_dim,
                                       d_filter = dim_filter,
                                       d_ff = dim_ff,
                                       dropout = dropout,
                                       embeddings = nucleotide_embeddings,
                                       attention_dropout = attention_dropout,
                                       global_filter=True)
   
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
    
    generator = Generator(model_dim,n_output_classes)
    model = NMTModel(encoder_stack,decoder_stack)
    model.generator = generator

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

