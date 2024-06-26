import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from bioseq2seq.models import NMTModel
from bioseq2seq.encoders import TransformerEncoder, CNNEncoder, FourierEncoder, GlobalFilterEncoderLayer,\
        LocalFilterEncoderLayer,FNetEncoderLayer,AFNOEncoderLayer
from bioseq2seq.decoders import TransformerDecoder, CNNDecoder
from bioseq2seq.modules import Embeddings

from bioseq2seq.modules import PadDecoder, PointerGenerator

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

def make_cnn_transformer_seq2seq(n_input_classes,n_output_classes,
                                 n_enc=4,n_dec=4,model_dim=128,
                                 dropout=0.1,encoder_kernel_size=3,
                                 encoder_dilation_factor=1,
                                 dim_ff=2048,heads=8,max_rel_pos=8):
    
    '''construct CNN-Transformer encoder-decoder from hyperparameters'''

    attention_dropout = 0.1
    
    nucleotide_embeddings = Embeddings(word_vec_size = model_dim,
                                       word_vocab_size = n_input_classes,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = model_dim,
                                    word_vocab_size = n_output_classes,
                                    word_padding_idx = 1,
                                    position_encoding = True)
    
    encoder_stack = CNNEncoder(num_layers = n_enc,
                                       hidden_size = model_dim,
                                       cnn_kernel_width = encoder_kernel_size,
                                       dropout = dropout,
                                       embeddings = nucleotide_embeddings,
                                       dilation_factor=encoder_dilation_factor)
    
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

    model.apply(cnn_init_weights)

    return model


def make_cnn_seq2seq(n_input_classes,n_output_classes,n_enc=4,n_dec=4,
                     model_dim=128,dropout=0.1,encoder_kernel_size=3,encoder_dilation_factor=1,
                     decoder_kernel_size=3,decoder_dilation_factor=1):
    
    '''construct CNN encoder-decoder from hyperparameters'''

    attention_dropout = 0.1
    
    nucleotide_embeddings = Embeddings(word_vec_size = model_dim,
                                       word_vocab_size = n_input_classes,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = model_dim,
                                    word_vocab_size = n_output_classes,
                                    word_padding_idx = 1,
                                    position_encoding = True)
    encoder_stack = CNNEncoder(num_layers = n_enc,
                                       hidden_size = model_dim,
                                       cnn_kernel_width = encoder_kernel_size,
                                       dropout = dropout,
                                       embeddings = nucleotide_embeddings,
                                       dilation_factor=encoder_dilation_factor)
    
    decoder_stack = CNNDecoder(num_layers = n_dec,
                                       hidden_size = model_dim,
                                       cnn_kernel_width = decoder_kernel_size,
                                       dropout = dropout,
                                       embeddings = protein_embeddings,
                                       dilation_factor=decoder_dilation_factor)
    
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

def make_transformer_seq2seq(n_input_classes,n_output_classes,n_enc=4,n_dec=4,model_dim=128,dim_ff=2048,heads=8,dropout=0.1,max_rel_pos=8):

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

def make_lfnet_cnn_seq2seq(n_input_classes,n_output_classes,n_enc=4,
                            n_dec=4,model_dim=128,dim_ff=2048,dropout=0.1,
                            window_size=200,lambd_L1=0.5,
                            decoder_kernel_size=3,decoder_dilation_factor=1):
    
    attention_dropout = 0.1

    nucleotide_embeddings = Embeddings(word_vec_size = model_dim,
                                       word_vocab_size = n_input_classes,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = model_dim,
                                    word_vocab_size = n_output_classes,
                                    word_padding_idx = 1,
                                    position_encoding = True)

    model_template_fn = lambda x : LocalFilterEncoderLayer(model_dim,dim_ff,dropout,window_size,share_hidden_weights=False,lambd_L1=lambd_L1)
    
    encoder_stack = FourierEncoder(model_template_fn,
                                    num_layers = n_enc,
                                    d_model = model_dim,
                                    embeddings = nucleotide_embeddings)
    
    decoder_stack = CNNDecoder(num_layers = n_dec,
                                       hidden_size = model_dim,
                                       cnn_kernel_width = decoder_kernel_size,
                                       dropout = dropout,
                                       embeddings = protein_embeddings,
                                       dilation_factor=decoder_dilation_factor)

    generator = Generator(model_dim,n_output_classes)
    model = NMTModel(encoder_stack,decoder_stack)
    model.generator = generator

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
    
def make_hybrid_seq2seq(n_input_classes,n_output_classes,n_enc=4,n_dec=4,model_dim=128,dim_ff=2048,dropout=0.1,\
        fourier_type='LFNet',dim_filter=100,heads=8,max_rel_pos=8,sparsity=1e-2,num_blocks=8,window_size=200,lambd_L1=0.5):

    '''construct Fourier encoder + Transformer decoder from hyperparameters'''

    attention_dropout = 0.1

    nucleotide_embeddings = Embeddings(word_vec_size = model_dim,
                                       word_vocab_size = n_input_classes,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = model_dim,
                                    word_vocab_size = n_output_classes,
                                    word_padding_idx = 1,
                                    position_encoding = True)

    if fourier_type == 'GFNet':
        model_template_fn =  lambda x : GlobalFilterEncoderLayer(model_dim,dim_ff,dropout,dim_filter,lambd_L1=lambd_L1)
    elif fourier_type == 'FNet':
        model_template_fn =  lambda x : FNetEncoderLayer(model_dim,dim_ff,dropout)
    elif fourier_type == 'LFNet':
        model_template_fn = lambda x : LocalFilterEncoderLayer(model_dim,dim_ff,dropout,window_size,share_hidden_weights=False,lambd_L1=lambd_L1)
    elif fourier_type == 'AFNO':
        model_template_fn = lambda x :  AFNOEncoderLayer(model_dim,dim_ff,dropout,num_blocks,sparsity)
    
    encoder_stack = FourierEncoder(model_template_fn,
                                    num_layers = n_enc,
                                    d_model = model_dim,
                                    embeddings = nucleotide_embeddings)

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

def attach_pointer_output(model,model_dim):
    decoder_stack = PadDecoder()
    generator = PointerGenerator(model_dim)
    model.decoder = decoder_stack
    model.generator = generator
    	
def restore_seq2seq_model(checkpoint,machine,opts):
    ''' Restore a seq2seq model from .pt
    Args:
        checkpoint : path to .pt saved model
        machine : torch device
    Returns:
        restored model'''
    
    vocab_fields = checkpoint['vocab'] 
    
    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
    n_input_classes = len(src_vocab.stoi)
    n_output_classes = len(tgt_vocab.stoi)
    n_output_classes = 28 
    
    if opts.model_type == 'Transformer':
        model = make_transformer_seq2seq(n_input_classes,
                                        n_output_classes,
                                        n_enc=opts.n_enc_layers,
                                        n_dec=opts.n_dec_layers,
                                        model_dim=opts.model_dim,
                                        max_rel_pos=opts.max_rel_pos)
    elif opts.model_type == 'CNN':
        model = make_cnn_seq2seq(n_input_classes,
                                n_output_classes,
                                n_enc=opts.n_enc_layers,
                                n_dec=opts.n_dec_layers,
                                model_dim=opts.model_dim,
                                encoder_kernel_size=opts.encoder_kernel_size,
                                decoder_kernel_size=opts.decoder_kernel_size,
                                encoder_dilation_factor=opts.encoder_dilation_factor,
                                decoder_dilation_factor=opts.decoder_dilation_factor)
    elif opts.model_type == 'CNN-Transformer':
        model = make_cnn_transformer_seq2seq(n_input_classes,
                                n_output_classes,
                                n_enc=opts.n_enc_layers,
                                n_dec=opts.n_dec_layers,
                                model_dim=opts.model_dim,
                                encoder_kernel_size=opts.encoder_kernel_size,
                                encoder_dilation_factor=opts.encoder_dilation_factor,
                                max_rel_pos=opts.max_rel_pos)

    elif opts.model_type == 'LFNet-CNN':
        model = make_lfnet_cnn_seq2seq(n_input_classes,
                                        n_output_classes,
                                        n_enc=opts.n_enc_layers,
                                        n_dec=opts.n_dec_layers,
                                        model_dim=opts.model_dim,
                                        window_size=opts.window_size,
                                        lambd_L1=opts.lambd_L1,
                                        dropout=opts.dropout,
                                        decoder_kernel_size=opts.decoder_kernel_size)
    else:
        model = make_hybrid_seq2seq(n_input_classes,
                                        n_output_classes,
                                        n_enc=opts.n_enc_layers,
                                        n_dec=opts.n_dec_layers,
                                        fourier_type=opts.model_type,
                                        model_dim=opts.model_dim,
                                        max_rel_pos=opts.max_rel_pos,
                                        dim_filter=opts.filter_size,
                                        window_size=opts.window_size,
                                        lambd_L1=opts.lambd_L1,
                                        dropout=opts.dropout)
    if opts.mode == 'start':
        attach_pointer_output(model,opts.model_dim)
    
    model.load_state_dict(checkpoint['model'],strict = False)
    model.generator.load_state_dict(checkpoint['generator'])
    return model
