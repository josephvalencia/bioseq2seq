#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/refseq_combined.csv.gz --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Aug24_09-48-18/_step_100000.pt --output_name classify  --mode classify
