#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python bioseq2seq/bin/train.py --input ../Fa/h_sapiens/v32/translations.map --save-directory original/ --num_gpus 4 
