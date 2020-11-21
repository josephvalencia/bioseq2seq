"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from bioseq2seq.inputters.inputter import \
    get_fields, OrderedIterator, \
    build_vocab,filter_example
from bioseq2seq.inputters.dataset_base import Dataset
from bioseq2seq.inputters.text_dataset import text_sort_key, TextDataReader
from bioseq2seq.inputters.datareader_base import DataReaderBase
from bioseq2seq.inputters.batcher import train_test_val_split,filter_by_length,dataset_from_df,iterator_from_dataset

str2reader = {
    "text": TextDataReader}
str2sortkey = {
    'text': text_sort_key}


__all__ = ['Dataset', 'get_fields', 'DataReaderBase',
           'filter_example',
           'build_vocab', 'OrderedIterator',
           'text_sort_key', 'TextDataReader','train_test_val_split','filter_by_length','dataset_from_df','iterator_from_dataset']
