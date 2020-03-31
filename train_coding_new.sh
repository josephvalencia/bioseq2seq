#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python bioseq2seq/bin/train.py --input ../Fa/h_sapiens/v33/coding.map --save-directory codingnew/ --num_gpus 4 
