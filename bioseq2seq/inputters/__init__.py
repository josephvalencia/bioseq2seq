"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from bioseq2seq.inputters.inputter import get_fields, build_vocab, filter_example
from bioseq2seq.inputters.iterator import max_tok_len, OrderedIterator
from bioseq2seq.inputters.dataset_base import Dataset, DynamicDataset
from bioseq2seq.inputters.text_dataset import text_sort_key, TextDataReader
from bioseq2seq.inputters.datareader_base import DataReaderBase

str2reader = {
    "text": TextDataReader}
str2sortkey = {
    'text': text_sort_key}


__all__ = ['Dataset', 'get_fields', 'DataReaderBase', 'filter_example',
           'build_vocab', 'OrderedIterator', 'max_tok_len',
           'text_sort_key', 'TextDataReader', 'DynamicDataset']
