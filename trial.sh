#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python bioseq2seq/bio/train.py --input ../Fa/h_sapiens/translations.map --save-directory ./new_checkpoints --num_gpus 4
