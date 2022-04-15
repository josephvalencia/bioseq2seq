#!/usr/bin/env python
import argparse
#import warnings
#warnings.filterwarnings('ignore')

from bioseq2seq.bin.train_utils import train_seq2seq, parse_train_args

if __name__ == "__main__":
    
    args = parse_train_args()
    train_seq2seq(args,tune=False)
    
