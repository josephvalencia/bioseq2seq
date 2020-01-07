import sys
import torch as torch 
import pandas as pd
import torchtext
import random

from tqdm import tqdm
from torchtext.data import Dataset, Example,Batch,Field,BucketIterator
from torchtext.data.iterator import RandomShuffler
import numpy as np
import random

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):

    '''Keep augmenting batch and calculate total number of tokens + padding.'''

    global max_src_in_batch, max_tgt_in_batch

    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 2)

    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch

    #return src_elements+tgt_elements
    return src_elements

class BatchMaker(torchtext.data.Iterator):

    def __init__(self,dataset,batch_size,device,repeat,train=True,sort_mode = "source"):

        if sort_mode == "source":
            sort_key = lambda x: len(x.src)
        elif sort_mode == "target":
            sort_key = lambda x: len(x.tgt)
        elif sort_mode =="total":
            sort_key = lambda x: len(x.tgt)+ len(x.src)

        super().__init__(dataset=dataset,batch_size=batch_size,device=device,
                         train=True,repeat=repeat,sort_key=sort_key)

        seed = 1.0
        state = random.seed(seed)
        self.random_shuffler = RandomShuffler(state)

    def create_batches(self):

        if self.train:

            def pool(d, random_shuffler):

                for p in torchtext.data.batch(d, self.batch_size * 100):

                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, batch_size_fn)

                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []

            for b in Batch(self.data(), self.batch_size,batch_size_fn):

                self.batches.append(sorted(b, key=self.sort_key))

class TransformerBatch(Batch):

        @classmethod
        def from_batch(cls,batch):

            batch_items = vars(batch)

            dataset = batch_items.pop('dataset',None)
            batch_size = batch_items.pop('batch_size',None)
            fields = batch_items.pop('fields',None)

            translation_batch = super().fromvars(dataset=dataset,batch_size = batch_size,train=True,**batch_items)

            return translation_batch

        def setup(self):
            '''
             Architecture-specific modification of batch attributes 
            '''
            x,x_lens = getattr(self,'src')

            tgt = getattr(self,'tgt')
            tgt = torch.unsqueeze(tgt,2)

            self.src = torch.unsqueeze(x,2),x_lens 
            self.tgt = tgt

class TranslationIterator:

    def __init__(self, iterator):

        self.iterator = iterator
        self.device = iterator.device
        self.fields = self.iterator.dataset.fields

    def __iter__(self):

        for batch in self.iterator:

            batch = TransformerBatch.from_batch(batch)
            batch.setup()
            yield batch

    def __len__(self):

        return len(self.iterator)

def filter_by_length(translation_table,max_len):

    translation_table['RNA_LEN'] = [len(x) for x in translation_table['RNA'].values]
    translation_table['Protein_LEN'] = [len(x) for x in translation_table['Protein'].values]
    interval = [0.1*x for x in range(1,10)]

    translation_table = translation_table[translation_table['RNA_LEN'] < max_len]
    return translation_table[['RNA','Protein']]

def tokenize(original):

    return [c for c in original]

def dataset_from_csv(translation_table,max_len,random_state):

    translation_table = filter_by_length(translation_table,max_len)
    RNA = Field(tokenize=tokenize,use_vocab=True,batch_first=False,include_lengths=True)
    PROTEIN =  Field(tokenize = tokenize, use_vocab=True,batch_first=False,is_target = True,include_lengths = False,init_token = "<sos>", eos_token = "<eos>")

    fields = {'RNA':('src', RNA), 'Protein':('tgt',PROTEIN)}

    reader = translation_table.to_dict(orient = 'records')
    examples = [Example.fromdict(line, fields) for line in tqdm(reader)]

    if isinstance(fields, dict):

        fields, field_dict = [], fields
        for field in field_dict.values():

            if isinstance(field, list):
                fields.extend(field)
            else:
                fields.append(field)

    dataset = Dataset(examples, fields)

    PROTEIN.build_vocab(dataset)
    RNA.build_vocab(dataset)

    return dataset.split(split_ratio = [0.8,0.1,0.1],random_state = random_state) # train,test,dev split

def iterator_from_dataset(dataset, max_tokens,device):

    return TranslationIterator(BatchMaker(dataset,batch_size = max_tokens,
                                          device = device,repeat=False,
                                          sort_mode ="source"))
