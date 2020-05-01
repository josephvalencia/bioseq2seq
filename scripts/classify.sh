#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/translate.py --input $BIOHOME/Fa/h_sapiens/v33/coding_noncoding.map --checkpoint $BIOHOME/bioseq2seq/checkpoints/coding_noncoding/Apr28_12-37-36/_step_80000.pt --output_name classify_new --mode classify
