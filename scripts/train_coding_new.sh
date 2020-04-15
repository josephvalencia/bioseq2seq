#!/usr/bin/env bash
export BIOHOME=/home/bb/valejose/home
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
source $BIOHOME/bioseq2seq/venv/bin/activate
python $BIOHOME/bioseq2seq/bioseq2seq/bin/train.py --input $BIOHOME/Fa/h_sapiens/v33/coding.map --save-directory $BIOHOME/bioseq2seq/checkpoints/codingnew/ --num_gpus 4 
