#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH=/home/bb/valejose/home/bioseq2seq
python bioseq2seq/bio/train.py --input ../Fa/h_sapiens/v32/two_class.map --accum_steps 4 --save-directory ./checkpoints4/ --num_gpus 4
