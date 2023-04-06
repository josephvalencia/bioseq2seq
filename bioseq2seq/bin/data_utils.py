import yaml
import torch
import torch.nn as nn
from argparse import Namespace
from collections import defaultdict, Counter
from bioseq2seq.inputters.inputter import _load_vocab, _build_fields_vocab, get_fields, IterOnDevice
from bioseq2seq.inputters.corpus import ParallelFastaCorpus, ParallelTextCorpus
from bioseq2seq.inputters.dynamic_iterator import DynamicDatasetIter
from bioseq2seq.translate import GNMTGlobalScorer, Translator, TranslationBuilder
from bioseq2seq.utils.misc import set_random_seed
from Bio import SeqIO
import bioseq2seq.bin.transforms as xfm
import copy
import numpy as np
import os
import time
import random

def std_vocab():

    counters = defaultdict(Counter)
    counters['src'] =  Counter({'A': 6, 'C': 5, 'G': 4, 'T': 3, 'N': 2, 'R': 1})

    counters['tgt'] = Counter({'<PC>': 24, '<NC>': 23, 'A': 22, 'C': 21, 'D': 20,\
                                'E': 19, 'F': 18, 'G': 17, 'H': 16,'I': 15, 'K': 14,\
                                'L': 13, 'M': 12, 'N': 11, 'P': 10, 'Q': 9,'R': 8,\
                                'S': 7, 'T': 6, 'U': 5, 'V': 4, 'W': 3, 'X': 2, 'Y': 1})
    return counters

def build_standard_vocab(src_vocab_path=None,tgt_vocab_path=None):
    
    if src_vocab_path is not None and tgt_vocab_path is not None:
        # load vocabularies
        counters = defaultdict(Counter)
        _src_vocab, _src_vocab_size = _load_vocab(src_vocab_path,'src',counters)
        _tgt_vocab, _tgt_vocab_size = _load_vocab(tgt_vocab_path,'tgt',counters)
    else: 
        counters = std_vocab()
   
   # initialize fields
    src_nfeats, tgt_nfeats = 0, 0 # do not support word features for now
    fields = get_fields('text', src_nfeats, tgt_nfeats)

    # build fields vocab
    share_vocab = False
    vocab_size_multiple = 1
    src_vocab_size = 8
    tgt_vocab_size = 28
    src_words_min_frequency = 1
    tgt_words_min_frequency = 1
    vocab_fields = _build_fields_vocab(fields, counters, 'text', share_vocab,
                                        vocab_size_multiple,
                                        src_vocab_size, src_words_min_frequency,
                                        tgt_vocab_size, tgt_words_min_frequency)

    return vocab_fields

def iterator_from_fasta(src,tgt,vocab_fields,mode,is_train,max_tokens,external_transforms=None,rank=0,world_size=1):

    # build the ParallelCorpus
    corpus_name = "train" if is_train else "valid"
    corpus = ParallelFastaCorpus("train",src,tgt,mode)
    print('Corpus made!') 
    transforms = {"attach_class_label" : xfm.AttachClassLabel(opts=None),'omit_peptide' : xfm.OmitPeptide(opts=None)}
    
    if mode == "bioseq2seq":
        transform_names = ["attach_class_label"]
    else:
        transform_names = ["attach_class_label","omit_peptide"] 
    # account for externally passed transforms
    if external_transforms is not None:
        transforms.update(external_transforms)
    if external_transforms is not None:
        transform_names += list(external_transforms.keys())
    
    corpora_info = {corpus_name: {"weight": 1 , "transforms": transform_names}}

    offset = rank if world_size > 1 else 0 
    # build the training iterator
    iterator = DynamicDatasetIter(corpora={corpus_name: corpus},
                                    corpora_info=corpora_info,
                                    transforms=transforms,
                                    fields=vocab_fields,
                                    is_train=is_train,
                                    batch_type="tokens",
                                    batch_size=max_tokens,
                                    bucket_size=1000,
                                    pool_factor=1000,
                                    batch_size_multiple=1,
                                    data_type="text",
                                    stride=world_size,
                                    offset=offset)
    return iterator

def test_effective_batch_size(iterator,dataset_size):

    start_time = time.time()
    n_examples = 0  
    percentages = []
    batch_sizes = []
    iterator = iter(iterator)
    
    for b in iterator:
        src = b.src[0]
        lengths = b.src[1]
        tgt = b.tgt
        num_padding = torch.eq(src,1).sum()
        tokens = src.shape[0] * src.shape[1]
        pad_pct = num_padding / tokens
        percentages.append(pad_pct.cpu())
        batch_sizes.append(src.shape[1])
        n_examples += src.shape[1]
        if n_examples >= dataset_size:
            break
    end_time = time.time()
    print(f'time to iterate {end_time-start_time}')
    print(f'% src pad tokens = {np.mean(percentages)} +- {np.std(percentages)}')
    print(f'% batch sizes = {np.mean(batch_sizes)} +- {np.std(batch_sizes)}')


