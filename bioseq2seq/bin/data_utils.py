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
from bioseq2seq.transforms import Transform
import numpy as np
import os
import time

class AttachClassLabel(Transform):
    '''Pre-pend class label based on FASTA sequence '''
    def apply(self, example, is_train=False, stats=None, **kwargs):
        
        curr_tgt = example['tgt'] 
        if curr_tgt[0] == '[NONE]':
            example['tgt'] = ['<NC>']
        else:
            example['tgt'] = ['<PC>'] + curr_tgt
        return example

class OmitPeptide(Transform):
    '''Remove amino acid sequence'''
    
    def apply(self, example, is_train=False, stats=None, **kwargs):
        
        curr_tgt = example['tgt']
        example['tgt'] = [curr_tgt[0]]
        return example

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

def iterator_from_fasta(src,tgt,vocab_fields,mode,is_train,max_tokens,rank=0,world_size=1):

    # build the ParallelCorpus
    corpus_name = "train" if is_train else "valid"
    corpus = ParallelFastaCorpus("train",src,tgt,mode)
   
    transforms = {"attach_class_label" : AttachClassLabel(opts=None),\
                    'omit_peptide' : OmitPeptide(opts=None)}

    if mode == "bioseq2seq":
        corpora_info = {corpus_name: {"weight": 1 , "transforms" : ["attach_class_label"]}}
    else:
        corpora_info = {corpus_name: {"weight": 1 , "transforms": ["attach_class_label", "omit_peptide"]}}

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
                                    offset=rank)
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


train_src = "new_data/mammalian_200-1200_train_RNA_balanced.fa" 
train_tgt = "new_data/mammalian_200-1200_train_PROTEIN_balanced.fa" 
val_src = "new_data/mammalian_200-1200_val_RNA_balanced.fa" 
val_tgt = "new_data/mammalian_200-1200_val_PROTEIN_balanced.fa" 

vocab_fields = build_standard_vocab()

train_iter = iterator_from_fasta(src=train_src,
                                tgt=train_tgt,
                                vocab_fields=vocab_fields,
                                mode='bioseq2seq',
                                is_train=True,
                                max_tokens=16000,
                                rank=0,
                                world_size=1) 
