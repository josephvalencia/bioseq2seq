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
from batcher import iterator_from_dataset, dataset_from_csv

from bioseq2seq.models import NMTModel
from bioseq2seq.encoders import TransformerEncoder
from bioseq2seq.decoders import TransformerDecoder
from bioseq2seq.modules import Embeddings
from bioseq2seq.utils.loss import NMTLossCompute, build_loss_compute
from bioseq2seq.utils.optimizers import Optimizer
from bioseq2seq.trainer import Trainer
from bioseq2seq.utils.report_manager import build_report_manager, ReportMgr

from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):

        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):

        logits = F.log_softmax(self.proj(x), dim=-1)
        return logits

class EncoderDecoder(NMTModel):
    """ Architecture-independent Encoder-Decoder. """

    def __init__(self, encoder, decoder,generator):

        super(EncoderDecoder, self).__init__(encoder,decoder)
        self.generator = generator

    def parallelize(self,device_ids,output_device):

        self.encoder = DDP(self.encoder,device_ids = device_ids, output_device = output_device)
        #self.decoder = DDP(self.decoder,device_ids = device_ids, output_device = output_device)
        self.generator = DDP(self.generator,device_ids = device_ids, output_device = output_device)

def make_transformer_model(n=4,d_model=128, d_ff=2048, h=8, dropout=0.1):

    "construct Transformer encoder-decoder from hyperparameters."

    NUM_INPUT_CLASSES = 6
    NUM_OUTPUT_CLASSES = 26

    nucleotide_embeddings = Embeddings(word_vec_size = d_model,word_vocab_size = NUM_INPUT_CLASSES,\
                                       word_padding_idx = 1,position_encoding = True)

    protein_embeddings = Embeddings(word_vec_size = d_model,word_vocab_size =NUM_OUTPUT_CLASSES,\
                                    word_padding_idx = 1,position_encoding = True)

    encoder_stack = TransformerEncoder(num_layers = n,d_model = d_model,heads = h, d_ff = d_ff,\
                                       dropout = dropout, embeddings = nucleotide_embeddings,\
                                       max_relative_positions = 10,attention_dropout = 0.1)

    decoder_stack = TransformerDecoder(num_layers = n,d_model = d_model,heads = h, d_ff = d_ff,\
                                       dropout = dropout, embeddings = protein_embeddings,\
                                       self_attn_type = 'scaled-dot',copy_attn = False,\
                                       max_relative_positions = 10,aan_useffn = False,attention_dropout = 0.1,\
                                       full_context_alignment=False,alignment_heads=None,alignment_layer=None)

    generator = Generator(128,26)

    model = EncoderDecoder(encoder_stack,decoder_stack,generator)

    for p in model.parameters():

        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def make_loss_function(device,generator):

    criterion = nn.NLLLoss(ignore_index=1, reduction='sum')
    nmt_loss = NMTLossCompute(criterion,generator)

    return nmt_loss

