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

    def parallelize(self):

        self.encoder = DataParallel(self.encoder)
        self.decoder = DataParallel(self.decoder)
        self.generator = DataParallel(self.generator)



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

if __name__ ==  "__main__":

    translation_record = sys.argv[1]

    data = pd.read_csv(translation_record,index_col = 0)
    data_iterator = iterator_from_csv(data)

    seq2seq = make_transformer_model()
    seq2seq.to(device = data_iterator.device)

    loss_computer = make_loss_function(device = data_iterator.device, generator = seq2seq.generator)

    adam = Adam(seq2seq.parameters())
    optim = Optimizer(adam, learning_rate = 1e-3)

    batch_count = 10000

    report_manager = ReportMgr(report_every = batch_count,tensorboard_writer = SummaryWriter())

    trainer = Trainer(seq2seq,train_loss = loss_computer,valid_loss = loss_computer,\
                      optim = optim,report_manager = report_manager )

    num_iterations = 1000

    print("Beginning training")

    for i in range(num_iterations):

        s = time.time()
        stats = trainer.train(data_iterator,1000000)
        e = time.time()
        print("Epoch: "+str(i)+" | Time elapsed: "+str(e-s))

    print("Concluding training")
