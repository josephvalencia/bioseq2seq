import torch as torch
import pandas as pd
import torchtext
import re
from torchtext.data import Dataset, Example,Batch,Field,RawField
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

    # return src_elements+tgt_elements
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

class TransformerBatch(torchtext.data.Batch):

        @classmethod
        def from_batch(cls,batch):
            '''Construct TransformerBatch from Batch '''

            batch_items = vars(batch)
            dataset = batch_items.pop('dataset',None)
            batch_size = batch_items.pop('batch_size',None)
            fields = batch_items.pop('fields',None)

            translation_batch = super().fromvars(dataset=dataset,batch_size = batch_size,train=True,**batch_items)

            return translation_batch

        def setup(self):
            '''Modify batch attributes to conform with OpenNMT architecture'''

            x,x_lens = getattr(self,'src')
            # add dummy dimension at axis = 2, src is a tuple
            self.src = torch.unsqueeze(x,2), x_lens
            self.tgt = torch.unsqueeze(getattr(self,'tgt'),2)
            self.id = getattr(self,'id')

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

        return len(self.iterator.dataset)

    def __test_batch_sizes__(self):
        '''Summarize batch sizes'''

        batch_sizes = []

        for batch in self.iterator:

            src_size = torch.sum(batch.src[1])
            tgt_size = batch.tgt.size(0)*batch.tgt.size(1)
            batch_sizes.append(batch.tgt.size()[1])
            total_tokens = src_size+tgt_size

            memory = torch.cuda.memory_allocated() / (1024*1024)

        df = pd.DataFrame()
        df['BATCH_SIZE'] = batch_sizes
        print(df.describe())

def filter_by_length(translation_table,max_len,min_len=0):
    '''Filter dataframe to RNA within (min_len,max_len)'''
    
    translation_table['RNA_LEN'] = [len(x) for x in translation_table['RNA'].values]
    translation_table['Protein_LEN'] = [len(x) for x in translation_table['Protein'].values]

    percentiles = [0.1 * x for x in range(1,10)]
    translation_table = translation_table[translation_table['RNA_LEN'] < max_len]

    if min_len > 0:
        translation_table =  translation_table[translation_table['RNA_LEN'] > min_len]
    
    print("total number =",len(translation_table)) 
    return translation_table[['ID','RNA', 'CDS', 'Type','Protein']]

def basic_tokenize(original):

    return [c for c in original]

def src_tokenize(original):
    "Converts genome into list of nucleotides"

    return [c for c in original]

def tgt_tokenize(original):
    "Converts protein into list of amino acids prepended with class label "

    splits = re.match("(<\w*>)(\w*)",original)

    if not splits is None:
        label = splits.group(1)
        protein = splits.group(2)
    else:
        label = "<UNK>"
        protein = original
    return [label]+[c for c in protein]

def train_test_val_split(translation_table,max_len,random_seed,min_len=0,splits=[0.8,0.1,0.1]):
    
    # keep entries with RNA length < max_len
    translation_table = filter_by_length(translation_table,max_len,min_len)
    # shuffle
    translation_table = translation_table.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    N = translation_table.shape[0]

    cumulative = [splits[0]]

    # splits to cumulative percentages
    for i in range(1,len(splits) -1):
        cumulative.append(cumulative[i-1]+splits[i])

    # train,test,val
    split_points = [int(round(x*N)) for x in cumulative]

    # split dataframe at split points
    train,test,val = np.split(translation_table,split_points)

    return train,test,val

def partition(dataset, split_ratios, random_state):

    """Create a random permutation of examples, then split them by split_ratios
    Arguments:
        dataset (torchtext.dataset): Dataset to partition
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

def dataset_from_df(df_list,mode="combined",saved_vocab = None):

    # Fields define tensor attributes
    if saved_vocab is None:

        RNA = Field(tokenize=src_tokenize,
                    use_vocab=True,
                    batch_first=False,
                    include_lengths=True)

        init = None if mode == "D_classify" else "<sos>"
        eos = None if mode == "D_classify" else "<eos>"

        PROTEIN =  Field(tokenize=tgt_tokenize,
                        use_vocab=True,
                        batch_first=False,
                        is_target=True,
                        include_lengths=False,
                        init_token=init,
                        eos_token=eos)
    else:
        RNA = saved_vocab['src']
        PROTEIN = saved_vocab['tgt']

    # GENCODE ID is string not tensor
    ID = RawField()
    splits = []
    
    for translation_table in df_list:
        # map column name to batch attribute and Field object
        if mode == "ED_classify" or mode == "D_classify":
            fields = {'ID':('id', ID),'RNA':('src', RNA),'Type':('tgt', PROTEIN)}
        elif mode == "translate":
            translation_table = translation_table[translation_table['Type'] == "<PC>"]
            fields = {'ID':('id', ID),'RNA':('src', RNA),'Protein':('tgt', PROTEIN)}
        elif mode == "combined":
            translation_table['Protein'] = translation_table['Type']+translation_table['Protein']
            fields = {'ID':('id', ID),'RNA':('src', RNA),'Protein':('tgt', PROTEIN)}

        # [{col:value}]
        reader = translation_table.to_dict(orient='records')
        examples = [Example.fromdict(line, fields) for line in reader]

        # Dataset expects fields as list
        field_list, field_dict = [], fields
        for field in field_dict.values():
            if isinstance(field, list):
                field_list.extend(field)
            else:
                field_list.append(field)

        dataset = Dataset(examples, field_list)
        splits.append(dataset)

    # Fields have a shared vocab over all datasets
    if saved_vocab is None:
        PROTEIN.build_vocab(*splits)
        RNA.build_vocab(*splits)

    #print("RNA:",RNA.vocab.stoi)
    #print("Protein:",PROTEIN.vocab.stoi)

    return tuple(splits)

def iterator_from_dataset(dataset, max_tokens, device, train):

    return TranslationIterator(BatchMaker(dataset,
                                          batch_size=max_tokens,
                                          device=device,
                                          repeat=train,
                                          sort_mode="source"))
