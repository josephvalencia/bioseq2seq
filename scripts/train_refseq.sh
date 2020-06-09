#!/usr/bin/env bash
source venv/bin/activate
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python $BIOHOME/bioseq2seq/bioseq2seq/bin/train.py --input $BIOHOME/Fa/refseq_combined.csv.gz --num_gpus 1 --mode combined --save-directory $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/  
