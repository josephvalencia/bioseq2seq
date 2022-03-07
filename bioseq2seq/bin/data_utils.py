import torch as torch
import pandas as pd
import torchtext
import re
from torchtext..data import Dataset, Example,Batch,Field,RawField
from torchtext.legacy.data.iterator import RandomShuffler
import numpy as np
import random

global max_src_in_batch, max_tgt_in_batch

def basic_tokenize(original):
    return [c for c in original]

def src_tokenize(original):
    "Converts genome into list of nucleotides"
    return [c for c in original]

def tgt_tokenize(original):
    "Converts protein into list of amino acids prepended with class label "

    splits = re.match("(<\\w*>)(\\w*)",original)
    if not splits is None:
        label = splits.group(1)
        protein = splits.group(2)
    else:
        label = "<UNK>"
        protein = original
    return [label]+[c for c in protein]


def partition(dataset, split_ratios, random_state):

    """Create a random permutation of examples, then split them by split_ratios
    Arguments:
        dataset (torchtext.legacy.dataset): Dataset to partition
        split_ratios (list): split fractions for Dataset partitions.
        random_state (int) : Random seed for shuffler
    """
    N = len(dataset.examples)
    rnd = RandomShuffler(random_state)
    randperm = rnd(range(N))

    indices = []
    current_idx = 0

    for ratio in split_ratios[:-1]:
        partition_len = int(round(ratio*N))
        partition = randperm[current_idx:current_idx+partition_len]
        indices.append(partition)
        current_idx +=partition_len

    last_partition = randperm[current_idx:]
    indices.append(last_partition)

    data = tuple([dataset.examples[i] for i in index] for index in indices)
    splits = tuple(Dataset(d, dataset.fields) for d in data )
    return splits

def dataset_from_df(df_list,mode="bioseq2seq",saved_vocab = None):

    # Fields define tensor attributes
    if saved_vocab is None:
        RNA = Field(tokenize=src_tokenize,
                    use_vocab=True,
                    batch_first=False,
                    include_lengths=True,
                    init_token=rna_init)

        PROTEIN =  Field(tokenize=tgt_tokenize,
                        use_vocab=True,
                        batch_first=False,
                        is_target=True,
                        include_lengths=False,
                        init_token=prot_init,
                        eos_token=prot_eos)
    else:
        RNA = saved_vocab['src']
        PROTEIN = saved_vocab['tgt']

    # ID is string not tensor
    ID = RawField()
    splits = []

    # map column name to batch attribute and Field object
    if mode == "bioseq2seq":
        for df in df_list:
            df['Protein'] = df['Type']+df['Protein']
        
    # for EDC only the class is the target
    if mode == "ED_classify":
        fields = {'ID':('id', ID),'RNA':('src', RNA),'Type':('tgt', PROTEIN)}
    # for bioseq2seq target is protein prepended with class
    elif mode == "bioseq2seq":
        fields = {'ID':('id', ID),'RNA':('src', RNA),'Protein':('tgt', PROTEIN)}

    for df in df_list:
        dataset = TabularDataset(df,format='tsv',fields=fields)
        splits.append(dataset)

    return tuple(dataset)
    
def iterator_from_dataset(dataset, max_tokens, device, train):

    return TranslationIterator(BatchMaker(dataset,
                                          batch_size=max_tokens,
                                          device=device,
                                          repeat=train,
                                          sort_mode="source"))
