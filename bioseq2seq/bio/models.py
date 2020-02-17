import torch as torch
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import pandas as pd
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from batcher import iterator_from_dataset, dataset_from_df

from bioseq2seq.models import NMTModel
from bioseq2seq.encoders import TransformerEncoder
from bioseq2seq.decoders import TransformerDecoder
from bioseq2seq.modules import Embeddings
from bioseq2seq.utils.loss import NMTLossCompute, build_loss_compute
from bioseq2seq.utils.optimizers import Optimizer
from bioseq2seq.utils.report_manager import build_report_manager, ReportMgr

from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):
    '''Fully connected + log-softmax over target vocab'''

    def __init__(self, d_model, vocab):

        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):

        logits = F.log_softmax(self.proj(x), dim=-1)
        return logits

def make_transformer_model(n=4,dim_model=128, dim_ff=2048, heads=8, dropout=0.1):
    ''' construct Transformer encoder-decoder from hyperparameters '''

    max_relative_positions = 10
    attention_dropout = 0.1
    NUM_INPUT_CLASSES = 6 # 4 nucleotides + <pad> + <unk>
    NUM_OUTPUT_CLASSES = 26 # 20 amino acids + <pad> + <unk> + <sos> + <eos>

    nucleotide_embeddings = Embeddings(word_vec_size = dim_model,
                                       word_vocab_size = NUM_INPUT_CLASSES,
                                       word_padding_idx = 1,
                                       position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = dim_model,
                                    word_vocab_size = NUM_OUTPUT_CLASSES,
                                    word_padding_idx = 1,
                                    position_encoding = True)

    encoder_stack = TransformerEncoder(num_layers = n,
                                       d_model = dim_model,
                                       heads = heads,
                                       d_ff = dim_ff,
                                       dropout = dropout,
                                       embeddings = nucleotide_embeddings,
                                       max_relative_positions = max_relative_positions,
                                       attention_dropout = attention_dropout)

    decoder_stack = TransformerDecoder(num_layers = n,
                                       d_model = dim_model,
                                       heads = heads,
                                       d_ff = dim_ff,
                                       dropout = dropout,
                                       embeddings = protein_embeddings,
                                       self_attn_type = 'scaled-dot',
                                       copy_attn = False,
                                       max_relative_positions = max_relative_positions,
                                       aan_useffn = False,
                                       attention_dropout = attention_dropout,
                                       full_context_alignment = False,
                                       alignment_heads = None,
                                       alignment_layer = None)

    generator = Generator(dim_model,NUM_OUTPUT_CLASSES)

    model = NMTModel(encoder_stack,decoder_stack)
    model.generator = generator

    for p in model.parameters():

        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def make_loss_function(device,generator,rank,num_gpus):

    criterion = nn.NLLLoss(ignore_index=1, reduction='sum')
    nmt_loss = NMTLossCompute(criterion,generator,rank,num_gpus)

    return nmt_loss

